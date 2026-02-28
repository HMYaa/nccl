/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include "transport.h"
#include "channel.h"
#include <assert.h>

__thread int ncclGroupDepth = 0; // depth of ncclGroupStart nesting
__thread ncclResult_t ncclGroupError = ncclSuccess;
__thread struct ncclComm* ncclGroupCommHead = nullptr;
__thread struct ncclComm* ncclGroupCommPreconnectHead = nullptr;
__thread struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> ncclAsyncJobs;
__thread struct ncclGroupJob *ncclGroupJobMainPtr = NULL;
__thread struct ncclGroupJob ncclGroupJobMain;
__thread int ncclGroupBlocking = -1; /* default mode */
__thread bool ncclGroupJobAbortFlag = false;

void* ncclAsyncJobMain(void* arg);
static ncclResult_t groupJobComplete(struct ncclGroupJob *job);

ncclResult_t ncclAsyncLaunch(
    struct ncclAsyncJob* job,
    ncclResult_t(*func)(struct ncclAsyncJob*),
    void(*undo)(struct ncclAsyncJob*),
    void(*destructor)(void*), ncclComm_t comm
  ) {
  ncclResult_t ret = ncclSuccess;

  if (ncclGroupDepth == 0) {
    ret = func(job);
    if (ret != ncclSuccess && undo) undo(job);
    if (destructor) destructor(job);
  } else {
    job->func = func;
    job->undo = undo;
    job->destructor = destructor;
    job->abortFlag = comm->abortFlag;
    job->state = ncclGroupJobRunning;
    job->comm = comm;
    /* check if there are blocking and nonblocking comms at the same time in group. */
    if (ncclGroupBlocking == -1) {
      /* first met communicator */
      ncclGroupBlocking = comm->blocking;
    } else if (ncclGroupBlocking != comm->blocking) {
      WARN("Blocking and nonblocking communicators are not allowed in the same group.");
      ret = ncclInvalidArgument;
    }
    ncclIntruQueueEnqueue(&ncclAsyncJobs, job);
  }

  return ret;
}

void* ncclAsyncJobMain(void* arg) {
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  job->result = job->func(job);
  if (job->result != ncclSuccess) {
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, job->result);
  }
  __atomic_store_n(&job->state, ncclGroupJobDone, __ATOMIC_RELEASE);
  return arg;
}

ncclResult_t ncclAsyncJobComplete(struct ncclAsyncJob* job) {
  ncclResult_t ret;
  SYSCHECK(pthread_join(job->thread, NULL), "pthread_join");
  if (job->result != ncclSuccess) {
    WARN("ncclAsyncJobComplete: job %p failed, job error %d", job, job->result);
  }
  ret = job->result;
  if (job->destructor) job->destructor((void*)job);
  return ret;
}

NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  ncclResult_t ret = ncclSuccess;
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  /* if previous group launch does not complete, don't launch this one. */
  if (ncclGroupJobMainPtr != NULL) {
    if (__atomic_load_n(&ncclGroupJobMainPtr->doneFlag, __ATOMIC_ACQUIRE) == false) {
      ret = ncclInvalidUsage;
      goto exit;
    } else {
      NCCLCHECKGOTO(groupJobComplete(ncclGroupJobMainPtr), ret, exit);
    }
  }
  NCCLCHECK(ncclGroupStartInternal());
  TRACE_CALL("ncclGroupStart()");

exit:
  return ret;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  ncclResult_t ret = ncclSuccess;
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, exit);
  TRACE_CALL("ncclGroupEnd()");
exit:
  return ret;
}

struct ncclPreconnectJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
};
ncclResult_t ncclPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  return ncclSuccess;
}

/**
 * doLaunches: 对「有待做任务的 comm 链表」执行 ncclLaunchPrepare -> 循环 (LaunchBefore/Launch/LaunchAfter) -> ncclLaunchFinish
 *
 * 按 clique 迭代：clique = 所有 intraComm0 相同的 comm（同一进程内、同一「全局实体」的一组分身）。
 * 对每个 clique：
 *   - 先对 clique 内每个 comm 调 ncclLaunchPrepare(comm)，把 tasks 变成 unlaunchedPlansHead；
 *   - 若 useBarrier，做 ncclCommIntraBarrierIn，以便多进程间同步「每人都有几轮 plan」；
 *   - 循环「轮次」：每轮对 clique 内每个 comm 从 unlaunchedPlansHead 弹一个 plan，
 *     LaunchBefore(uploadWork) -> LaunchKernel -> LaunchAfter(hostStreamPlanTask 等)；若无 plan 则 ncclLaunchFinish。
 * 这样保证：同一 clique 内多 comm 的 Kernel 发射顺序、与 Proxy 的配合，在多进程下一致。
 */
static ncclResult_t doLaunches(struct ncclComm* head) {
  ncclResult_t result = ncclSuccess;
  struct ncclComm* cliqueComm0 = head->intraComm0;
  struct ncclComm* cliqueHead = head;
  struct ncclComm* cliqueNextHead;
  bool useBarrier = ncclParamLaunchMode == ncclLaunchModeGroup;
  do {
    struct ncclComm* comm = cliqueHead;
    bool capturingYes = false, capturingNo = false;
    // 同一 clique 内：每个 comm 先 Prepare，把 tasks 转成 unlaunchedPlansHead
    do {
      (ncclCudaGraphValid(comm->tasks.capturingGraph) ? capturingYes : capturingNo) = true;
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
      NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure); // 排产
      if (useBarrier) ncclCommIntraBarrierIn(comm, 1);
      comm = comm->groupNext;
    } while (comm != nullptr && comm->intraComm0 == cliqueComm0);
    cliqueNextHead = comm;

    if (capturingYes && capturingNo) {
      WARN("Either none or all communicators in a ncclGroup() can be CUDA graph captured.");
      result = ncclInvalidUsage;
      goto failure;
    }

    // 多轮发射：每轮每个 comm 弹一个 plan 做 LaunchBefore/Launch/LaunchAfter，或做 LaunchFinish
    while (true) {
      bool moreRounds;
      comm = cliqueHead;
      do {
        struct ncclComm* next = comm->groupNext;
        if (useBarrier) {
          moreRounds = 0 != ncclCommIntraBarrierOut(comm);
        } else {
          moreRounds = comm->unlaunchedPlansHead != nullptr;
        }
        if (moreRounds) {
          struct ncclKernelPlan* plan = comm->unlaunchedPlansHead;
          if (plan != nullptr) {
            comm->unlaunchedPlansHead = plan->next;
            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan), result, failure);  // uploadWork
            NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure);  // 按排产开火
          }
          if (useBarrier) ncclCommIntraBarrierIn(comm, comm->unlaunchedPlansHead != nullptr ? 1 : 0);
          if (plan != nullptr) {
            NCCLCHECKGOTO(ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);  // 非 persistent 时 hostStreamPlanTask , 推任务
          }
        } else {
          CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
          NCCLCHECKGOTO(ncclLaunchFinish(comm), result, failure);
        }
        comm = next;
      } while (comm != cliqueNextHead);
      if (!moreRounds) break;
    }
    cliqueHead = cliqueNextHead;
  } while (cliqueHead != nullptr);
failure:
  return result;
}

static inline void groupResetJobState() {
  ncclGroupBlocking = -1;
  ncclGroupJobMainPtr = NULL;
  memset(&ncclGroupJobMain, 0, sizeof(struct ncclGroupJob));
  return;
}

static void groupCleanup(struct ncclComm** groupCommHeadPtr, struct ncclComm** groupCommPreconnectHeadPtr, struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next>* asyncJobsPtr, ncclResult_t* groupErrorPtr, ncclResult_t error) {
  struct ncclComm* comm = *groupCommHeadPtr;

  while (comm != nullptr) {
    struct ncclComm* next = comm->groupNext;
    (void) ncclGroupCommLeave(comm); // overwrites comm->groupNext
    // We don't know if preconnect succeeded or happened at all, so clear
    // the flags that let `taskAppend()` skip over checking if preconnect
    // is needed.
    comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
    for (int i = 0; i < comm->nRanks; i++) {
      comm->tasks.peers[i].sendSeen = false;
      comm->tasks.peers[i].recvSeen = false;
      comm->connectSend[i] = 0;
      comm->connectRecv[i] = 0;
    }
    comm->unlaunchedPlansHead = nullptr;
    // Reclaim abandoned kernel plan memory. Note ncclWork structs were already
    // reclaimed by a `ncclMemoryStackPop(&comm->memScoped)` during `ncclGroupCommLeave()`.
    while (!ncclIntruQueueEmpty(&comm->planQueue)) {
      struct ncclKernelPlan* plan = ncclIntruQueueDequeue(&comm->planQueue);
      // Persistent plans will be reclaimed via the callbackQueue when the
      // graph drops its UserObject reference.
      if (!plan->persistent) {
        for (int c = 0; c < MAXCHANNELS; c++) {
          while (!ncclIntruQueueEmpty(&plan->channels[c].proxyOpQueue)) {
            struct ncclProxyOp* pxop = ncclIntruQueueDequeue(&plan->channels[c].proxyOpQueue);
            ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, pxop);
          }
        }
        ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
      }
    }
    // Reset comm->tasks to empty.
    comm->tasks.nTasksColl = 0;
    comm->tasks.nTasksP2p = 0;
    comm->tasks.streams = nullptr;
    ncclIntruQueueConstruct(&comm->tasks.collQueue);
    comm->tasks.collBytesTotal = 0;
    for (int i = 0; i < comm->nRanks; i++) {
      ncclIntruQueueConstruct(&comm->tasks.peers[i].sendQueue);
      ncclIntruQueueConstruct(&comm->tasks.peers[i].recvQueue);
    }

    if (!comm->blocking)
      (void) ncclCommSetAsyncError(comm, error);
    comm = next;
  }

  /* reset everything */
  while (!ncclIntruQueueEmpty(asyncJobsPtr)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsPtr);
    *job->abortFlag = 1;
    if (job->comm && !job->comm->blocking)
      (void) ncclCommSetAsyncError(job->comm, error);
    if (job->undo) job->undo(job);
    if (job->destructor) job->destructor((void*)job);
  }

  *groupErrorPtr = ncclSuccess;
  *groupCommHeadPtr = nullptr;
  *groupCommPreconnectHeadPtr = nullptr;
  return;
}

/**
 * groupLaunch: ncclGroupEnd 深度变为 0 时触发的「组执行」入口
 *
 * 流程：1) 若有 preconnect 链表，为每个 comm 创建 ncclPreconnectJob 入 asyncJobs，并 pthread 跑完；
 *      2) 若有其它 async 任务（如 ncclCommInitRank 的 job），pthread 创建并等全部 Done；
 *      3) 若有 groupCommHead，调 doLaunches(groupCommHead) 对 comm 链做 Prepare -> 循环 Launch* -> Finish；
 *      4) 置 doneFlag，清 async 队列、groupCommHead、groupCommPreconnectHead，并 GroupCommLeave。
 * 阻塞/非阻塞：由 ncclGroupEndInternal 根据 ncclGroupBlocking、是否有 preconnect/async 决定
 *             是直接调 groupLaunch 还是起线程跑 groupLaunch。
 */
static ncclResult_t groupLaunch(struct ncclAsyncJob *job_) {
  int savedDev;
  ncclResult_t ret = ncclSuccess;
  bool jobsDone = false;
  bool errorJobAbortFlag = false;
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;
  struct ncclComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
  struct ncclComm *groupCommPreconnectHeadMain = *gjob->groupCommPreconnectHeadPtr;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = gjob->asyncJobsPtr;
  volatile bool *groupAbortFlag = gjob->abortFlagPtr;

  CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail);

  // 若有 preconnect：为每个 comm 建 PreconnectJob 并入 async 队列，后面统一 pthread 跑
  if (groupCommPreconnectHeadMain != nullptr) {
    struct ncclComm* comm = groupCommPreconnectHeadMain;
    do {
      struct ncclPreconnectJob* job;
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
      job->base.func = ncclPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->comm = comm;
      ncclIntruQueueEnqueue(asyncJobsMain, &job->base);

      struct ncclComm* next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  // 跑完所有 async 任务（preconnect、commInit 等）
  if (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueHead(asyncJobsMain);
    do {
      SYSCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job), ret, fail);
      job = job->next;
    } while (job != nullptr);

    do {
      jobsDone = true;
      job = ncclIntruQueueHead(asyncJobsMain);
      do {
        ncclGroupJobState_t state = __atomic_load_n(&job->state, __ATOMIC_ACQUIRE);
        if (state == ncclGroupJobRunning) {
          jobsDone = false;
        } else if (state == ncclGroupJobDone) {
          if (pthread_join(job->thread, nullptr) != 0) {
            WARN("Error waiting for pthread_join : %s", strerror(errno));
            ret = ncclSystemError;
          }
          job->state = ncclGroupJobJoined;
          if (job->result != ncclSuccess) {
            ret = job->result;
            errorJobAbortFlag = true;
          }
        } else {
          /* safety check */
          assert(state == ncclGroupJobJoined);
        }

        if (*groupAbortFlag == true || errorJobAbortFlag == true) {
          *job->abortFlag = 1;
          ret = ncclInternalError;
        }

        job = job->next;
      } while (job != nullptr);
    } while (jobsDone == false);

    if (ret != ncclSuccess) goto fail;
  }

  // 对待做任务的 comm 链表执行：ncclLaunchPrepare -> 循环 Launch* -> ncclLaunchFinish
  if (groupCommHeadMain != nullptr) {
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain), ret, fail);
  }

  __atomic_store_n(&gjob->doneFlag, true, __ATOMIC_RELEASE);  // 必须在清理和设置 comm 状态之前

  while (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain);
    if (job->comm && !job->comm->blocking)
      (void) ncclCommSetAsyncError(job->comm, ret);
    if (job->destructor) job->destructor((void*)job);
  }

  while (groupCommHeadMain != nullptr) {
    struct ncclComm* comm = groupCommHeadMain;
    struct ncclComm* next = comm->groupNext;
    (void) ncclGroupCommLeave(comm);
    if (!comm->blocking) {
      (void) ncclCommSetAsyncError(comm, ret);
    }
    groupCommHeadMain = next;
  }

  *gjob->groupErrorPtr = ncclSuccess;
  *gjob->groupCommHeadPtr = nullptr;
  *gjob->groupCommPreconnectHeadPtr = nullptr;

  CUDACHECK(cudaSetDevice(savedDev));

exit:
  return ret;
fail:
  groupCleanup(gjob->groupCommHeadPtr, gjob->groupCommPreconnectHeadPtr, gjob->asyncJobsPtr, gjob->groupErrorPtr, ret);
  goto exit;
}

/**
 * ncclGroupEndInternal: Group 深度减 1；当深度从 1 变为 0 时，触发本「组」的执行（groupLaunch）
 *
 * 深度>0：仅减深度后返回。
 * 深度变为 0：若无 groupCommHead 且无 async 且无 preconnect，直接返回；
 *            否则填充 ncclGroupJobMain，根据 blocking 与是否有 preconnect/async：
 *            - 非阻塞且 (preconnect 非空 或 async 非空)：起 pthread 跑 groupLaunch，返回 ncclInProgress；
 *            - 其它：本线程直接 groupLaunch，再 groupResetJobState。
 * 这样：单次 ncclAllReduce 时，ncclEnqueueCheck 里 GroupStart(深度=1) -> taskAppend -> GroupEnd(深度=0)
 *      会在这里触发 groupLaunch -> doLaunches -> ncclLaunchPrepare -> ncclLaunchKernel。
 */
ncclResult_t ncclGroupEndInternal() {
  ncclResult_t ret = ncclSuccess;

  if (ncclGroupDepth == 0) {
    WARN("ncclGroupEnd: not in a group call.");
    ret = ncclInvalidUsage;
    goto exit;
  }

  if ((--ncclGroupDepth) > 0) goto exit;  // 深度仍>0，只减深度

  if ((ret = ncclGroupError) != ncclSuccess) goto fail;

  if (ncclGroupCommHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    ncclGroupJobMain.groupCommHeadPtr = &ncclGroupCommHead;
    ncclGroupJobMain.groupCommPreconnectHeadPtr = &ncclGroupCommPreconnectHead;
    ncclGroupJobMain.groupErrorPtr = &ncclGroupError;
    ncclGroupJobMain.asyncJobsPtr = &ncclAsyncJobs;
    ncclGroupJobMain.abortFlagPtr = &ncclGroupJobAbortFlag;
    ncclGroupJobMain.doneFlag = false;
    ncclGroupJobMainPtr = &ncclGroupJobMain;
    assert(ncclGroupBlocking == 0 || ncclGroupBlocking == 1);
    // 非阻塞 且 需要 preconnect 或 有 async 任务：起线程跑 groupLaunch，先返回 ncclInProgress
    if (ncclGroupBlocking == 0 && (ncclGroupCommPreconnectHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs))) {
      if (!ncclIntruQueueEmpty(&ncclAsyncJobs)) {
        ncclAsyncJob* job = ncclIntruQueueHead(&ncclAsyncJobs);
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
          job = job->next;
        } while (job);
      }

      if (ncclGroupCommHead) {
        ncclComm_t comm = ncclGroupCommHead;
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
          comm = comm->groupNext;
        } while (comm);
      }
      ncclGroupJobMainPtr->base.func = groupLaunch;
      SYSCHECKGOTO(pthread_create(&ncclGroupJobMainPtr->base.thread, NULL, ncclAsyncJobMain, (void*)&ncclGroupJobMainPtr->base), ret, fail);
      ret = ncclInProgress;
    } else {
      NCCLCHECKGOTO(groupLaunch(&ncclGroupJobMainPtr->base), ret, fail);  // 阻塞：本线程直接执行
      groupResetJobState();
    }
  }

exit:
  return ret;
fail:
  groupCleanup(&ncclGroupCommHead, &ncclGroupCommPreconnectHead, &ncclAsyncJobs, &ncclGroupError, ret);
  groupResetJobState();
  goto exit;
}

static ncclResult_t groupJobComplete(struct ncclGroupJob* job) {
  ncclResult_t ret = ncclSuccess;
  if (job) {
    ret = ncclAsyncJobComplete(&job->base);
    groupResetJobState();
  }
  return ret;
}

void ncclGroupJobAbort() {
  ncclGroupJobAbortFlag = true;
  (void) groupJobComplete(ncclGroupJobMainPtr);
  /* reset group abort flag */
  ncclGroupJobAbortFlag = false;
}
