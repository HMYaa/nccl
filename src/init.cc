/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "coll_net.h"
#include "enqueue.h"
#include "graph.h"
#include "argcheck.h"
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define STR2(v) #v
#define STR(v) STR2(v)

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", 0);

static uint64_t hashUniqueId(ncclUniqueId const &id) {
  char const *bytes = (char const*)&id;
  uint64_t h = 0xdeadbeef;
  for(int i=0; i < (int)sizeof(ncclUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

// GDRCOPY support: Off by default
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);

// GDRCOPY support
gdr_t ncclGdrCopy = NULL;

ncclResult_t initGdrCopy() {
  if (ncclParamGdrCopyEnable() == 1) {
    ncclGdrCopy = ncclGdrInit();
  }
  return ncclSuccess;
}


NCCL_PARAM(L1SharedMemoryCarveout, "L1_SHARED_MEMORY_CARVEOUT", 0);

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static size_t maxLocalSizeBytes = 0;

static ncclResult_t ncclInit() {
  if (__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    initGdrCopy();
    maxLocalSizeBytes = ncclKernMaxLocalSize();
    int carveout = ncclParamL1SharedMemoryCarveout();
    if (carveout) ncclKernSetSharedMemoryCarveout(carveout);
    // Always initialize bootstrap network
    NCCLCHECK(bootstrapNetInit());
    NCCLCHECK(ncclNetPluginInit());

    __atomic_store_n(&initialized, true, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

/**
 * ncclGetUniqueId: 生成唯一通信ID
 * 
 * 功能：生成一个128字节的唯一ID，用于标识一个通信组
 * 所有要加入同一通信组的rank必须使用相同的uniqueId
 * 
 * 使用场景：
 * - 多进程场景：rank 0生成ID，然后通过某种机制（如文件、环境变量）分发给其他rank
 * - 单进程多GPU场景（ncclCommInitAll）：所有GPU共享同一个uniqueId
 */
NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  // 确保全局初始化已完成
  NCCLCHECK(ncclInit());
  // 参数校验
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  // 通过Bootstrap机制生成唯一ID（通常是基于时间戳和随机数）
  ncclResult_t res = bootstrapGetUniqueId(out);
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(*out));
  return res;
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  // Important that this does not trash intraComm0 & intraRefs.
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
}

#undef NCCL_NO_OPTIMIZE


static ncclResult_t ncclDestructorFnFree(struct ncclDestructor* dtor) {
  free(dtor->obj);
  return ncclSuccess;
}
void ncclCommPushFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaFree(struct ncclDestructor* dtor) {
  CUDACHECK(cudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaHostFree(struct ncclDestructor* dtor) {
  CUDACHECK(cudaFreeHost(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaHostFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaGdrFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclGdrCudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaGdrFree;
  dtor->obj = handle;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  // Stop all threads before we free anything.
  NCCLCHECK(ncclProxyDestroy(comm));

  delete[] comm->userRedOps;

  free(comm->connectSend);
  free(comm->connectRecv);

  free(comm->peerInfo);
  if (comm->topo)
    ncclTopoFree(comm->topo);
  if (comm->nodeRanks) {
    for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
    free(comm->nodeRanks);
  }
  free(comm->rankToNode);
  free(comm->rankToLocalRank);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks));

  if (comm->initState == ncclSuccess) {
    NCCLCHECK(ncclStrongStreamDestruct(&comm->hostStream));
    NCCLCHECK(ncclStrongStreamDestruct(&comm->deviceStream));
  }

  struct ncclDestructor* dtor = comm->destructorHead;
  while (dtor != nullptr) {
    NCCLCHECK(dtor->fn(dtor));
    dtor = dtor->next;
  }

  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  commPoison(comm); // Important that this does not interfere with anything used below.

  if (comm->initState == ncclSuccess) {
    struct ncclComm* intraComm0 = comm->intraComm0;
    if (0 == ncclAtomicRefCountDecrement(&intraComm0->intraRefs)) {
      // Wait for all service threads to be done. We could not
      // do it earlier because it could have blocked and prevented
      // other ranks in the process to call ncclCommDestroy
      comm = intraComm0;
      while (comm != nullptr) {
        if (comm->proxyState.thread) pthread_join(comm->proxyState.thread, nullptr);
        struct ncclComm* next = comm->intraNext;
        free(comm);
        comm = next;
      }
    }
  } else if (comm->proxyState.thread) {
    pthread_join(comm->proxyState.thread, nullptr);
    ncclCudaHostFree((void *)comm->abortFlag);
    free(comm);
  } else {
    ncclCudaHostFree((void *)comm->abortFlag);
    free(comm);
  }

  return ncclSuccess;
}

NCCL_PARAM(AggChannelSize, "AGG_CHANNEL_SIZE", -2);
NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
NCCL_PARAM(WorkFifoDepth, "WORK_FIFO_DEPTH", 64<<10);
enum ncclLaunchMode ncclParamLaunchMode;

NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 1);

// Detect DMA-BUF support
static ncclResult_t dmaBufSupported(struct ncclComm* comm) {
  if (ncclParamDmaBufEnable() == 0 || comm->ncclNet->regMrDmaBuf == NULL) return ncclInternalError;
#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion;
  CUCHECK(cuDriverGetVersion(&cudaDriverVersion));
  if (cudaDriverVersion < 11070) return ncclInternalError;
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));
  // Query device to see if DMA-BUF support is available
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0) return ncclInternalError;
  INFO(NCCL_INIT, "DMA-BUF is available on GPU device %d", comm->cudaDev);
  return ncclSuccess;
#endif
  return ncclInternalError;
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* comm must be ready, or error will be reported */
  ncclResult_t ret = ncclSuccess;

  if (*comm->abortFlag) {
    ncclGroupJobAbort();
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret != ncclSuccess) {
      /* if ret is not ncclInProgress, we just keep it. */
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess\n");
      if (ret == ncclInProgress) ret = ncclInvalidArgument;
      goto exit;
    }
  }

exit:
  return ret;
}

/**
 * commAlloc: 分配和初始化通信器的基础资源
 * 
 * 这是通信器初始化的第一步，负责分配所有基础数据结构：
 * - 内存管理（永久内存栈、作用域内存栈）
 * - 网络层初始化
 * - CUDA Stream创建
 * - 内存池初始化
 * - 通道初始化
 */
static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  struct ncclComm* comm;
  /* 目前我们在ncclCommInitRankDev中为异步函数支持分配comm。
   * 这个if结构是为了考虑commAlloc在其他情况下被调用的情况（除了ncclCommInitRankDev）。 */
  if (*comret == NULL) {
    /* 用户请求一个新的通信器 */
    NCCLCHECK(ncclCalloc(&comm, 1));
    NCCLCHECK(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1));
    NCCLCHECK(ncclCommSetAsyncError(comm, ncclInProgress));
  } else {
    /* 通信器已经在ncclCommInitRankDev中分配了 */
    comm = *comret;
  }

  // ========== 内存管理初始化 ==========
  // 永久内存栈：在整个通信器生命周期内存在
  ncclMemoryStackConstruct(&comm->memPermanent);
  // 作用域内存栈：在特定作用域内使用
  ncclMemoryStackConstruct(&comm->memScoped);
  comm->destructorHead = nullptr;
  
  // ========== 设置基本rank信息 ==========
  comm->rank = rank;
  comm->nRanks = ndev;

  // ========== 网络层初始化 ==========
  // 初始化网络抽象层（会检测可用的网络类型：IB、Socket等）
  NCCLCHECK(ncclNetInit(comm));
  INFO(NCCL_INIT, "Using network %s", ncclNetName(comm));

  // ========== CUDA Stream创建 ==========
  // 立即创建CUDA对象。如果设备有问题（失败原因#1），最好早点知道
  // deviceStream: 用于GPU Kernel执行
  // hostStream: 用于Host端操作
  NCCLCHECK(ncclStrongStreamConstruct(&comm->deviceStream));
  NCCLCHECK(ncclStrongStreamConstruct(&comm->hostStream));

  // ========== 获取GPU信息 ==========
  // 获取当前CUDA设备ID
  cudaGetDevice(&comm->cudaDev);
  // 获取GPU的PCIe Bus ID（用于拓扑检测和P2P通信）
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx", comm, rank, ndev, comm->cudaDev, comm->busId);

  // ========== 功能特性检测 ==========
  // 是否启用指针检查（调试用）
  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
  // 是否支持DMA Buffer（用于零拷贝）
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;
  // CollNet支持（集合网络，用于特定硬件加速）
  comm->collNetSupport = 0;

  // ========== 内存池初始化 ==========
  // 为不同类型的内存分配请求创建内存池，提高分配效率
  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);  // Kernel计划内存池
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);      // Proxy操作内存池
  ncclMemoryPoolConstruct(&comm->memPool_ncclPointerList);  // 指针列表内存池

  // ========== 通信器链表初始化 ==========
  // 用于组语义和预连接管理
  comm->groupNext = reinterpret_cast<struct ncclComm*>(0x1);
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
  // 通道大小（聚合通道的理想大小，用于充分利用带宽）
  comm->channelSize = ncclParamAggChannelSize();

  // ========== 通道连接状态位图 ==========
  // 使用位图跟踪每个rank的发送/接收通道连接状态
  // connectSend/connectRecv: 每个rank一个uint64_t，每个bit代表一个通道
  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  // ========== 通道初始化 ==========
  // 标记所有通道为未初始化状态（id = -1）
  for (int c=0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;

  // ========== 回调队列初始化 ==========
  // 多生产者单消费者（MPSC）队列，用于异步回调
  ncclIntruQueueMpscConstruct(&comm->callbackQueue);

  *comret = comm;
  return ncclSuccess;
}

/**
 * devCommSetup: 设置设备端（GPU）通信器
 * 
 * 功能：在GPU可访问的内存中创建通信器结构，使得GPU Kernel可以直接访问通信器信息
 * 
 * 主要工作：
 * 1. 获取CUDA Stream
 * 2. 在GPU内存中分配ncclDevCommAndChannels结构
 * 3. 将Host端的通信器信息复制到GPU内存
 * 4. 设置工作队列（workFifo）相关参数
 */
static ncclResult_t devCommSetup(ncclComm_t comm) {
  // 获取设备Stream（用于异步内存分配和复制）
  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&comm->deviceStream));

  int nRanks = comm->nRanks;
  struct ncclDevCommAndChannels *devCommAndChans, tmpCommAndChans;
  
  // ========== 在GPU内存中分配设备端通信器结构 ==========
  // 这个结构体包含通信器信息和所有通道信息，GPU Kernel可以直接访问
  NCCLCHECK(ncclCudaCallocAsync(&devCommAndChans, 1, comm->deviceStream.stream));
  // 注册到清理列表，通信器销毁时自动释放
  ncclCommPushCudaFree(comm, devCommAndChans);
  comm->devComm = &devCommAndChans->comm;
  
  // ========== 准备要复制到GPU的数据 ==========
  // 在Host端准备临时结构，然后复制到GPU
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.abortFlag = comm->abortFlag;  // GPU可以读取此标志检查是否需要中止
  // 复制各协议的缓冲区大小（Simple/LL/LL128）
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  // 设置通道指针（指向GPU内存中的通道数组）
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  comm->workFifoDepth = ncclParamWorkFifoDepth();
  if (0 != (comm->workFifoDepth & (comm->workFifoDepth-1))) {
    WARN("NCCL_WORK_FIFO_DEPTH=%d is being ignored because it is not a power of 2.", comm->workFifoDepth);
    comm->workFifoDepth = 64<<10;
  }
  tmpCommAndChans.comm.workFifoDepth = comm->workFifoDepth;

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // The workFifoHeap lives in GDR mapped CUDA memory.
    NCCLCHECK(ncclGdrCudaCalloc(&comm->workFifoHeap, &comm->devWorkFifoHeap, comm->workFifoDepth, &comm->workFifoHeapGdrHandle));
    ncclCommPushCudaGdrFree(comm, comm->workFifoHeapGdrHandle);
  } else {
    // The workFifoHeap lives in cudaHost memory.
    comm->workFifoHeapGdrHandle = nullptr;
    NCCLCHECK(ncclCudaHostCalloc(&comm->workFifoHeap, comm->workFifoDepth));
    ncclCommPushCudaHostFree(comm, comm->workFifoHeap);
    comm->devWorkFifoHeap = comm->workFifoHeap;
  }
  tmpCommAndChans.comm.workFifoHeap = comm->devWorkFifoHeap;

  NCCLCHECK(ncclCudaHostCalloc(&comm->workFifoDone, MAXCHANNELS));
  ncclCommPushCudaHostFree(comm, comm->workFifoDone);
  comm->workFifoSent = 0;
  comm->workFifoAckdMin = 0;

  for (int c=0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].workFifoDone = &comm->workFifoDone[c];

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECK(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, comm->deviceStream.stream));
    }
  }

  NCCLCHECK(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, comm->deviceStream.stream));
  CUDACHECK(cudaStreamSynchronize(comm->deviceStream.stream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNull(), &comm->deviceStream));
  return ncclSuccess;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
static void showVersion() {
  static int shown = 0;
  if (shown == 0 && ncclDebugLevel >= NCCL_LOG_VERSION) {
    printf("%s\n", VERSION_STRING);
    fflush(stdout);
    if (ncclDebugFile != stdout)
      INFO(NCCL_ALL,"%s", VERSION_STRING); // Also log NCCL version in one of the files
    shown = 1;
  }
}

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  CUDACHECK(cudaGetDevice(&info->cudaDev));
  info->hostHash=getHostHash()+commHash;
  info->pidHash=getPidHash()+commHash;

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  struct stat statbuf;
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = statbuf.st_dev;

  info->busId = comm->busId;

  NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));
  info->comm = comm;
  info->cudaCompCap = ncclCudaCompCap();
  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Find our ring-distance from rank zero and reorganize ranks to start with rank.
  int ixZero=0, ixRank=0;
  for (int i=0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank-ixZero + nranks)%nranks;
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];
  }
  return ncclSuccess;
}

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_BUFFSIZE_ARM (1 << 20) /* 1MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));

  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  if (cpuArch == NCCL_TOPO_CPU_ARCH_ARM) defaults[NCCL_PROTO_SIMPLE] = DEFAULT_BUFFSIZE_ARM;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  comm->p2pNetChunkSize = ncclParamP2pNetChunkSize();
  return ncclSuccess;
}

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 1);
NCCL_PARAM(AllocP2pNetLLBuffers, "NCCL_ALLOC_P2P_NET_LL_BUFFERS", 0);

/**
 * initTransportsRank: 初始化传输层（核心初始化函数）
 * 
 * 这是通信器初始化最复杂的部分，负责建立所有rank之间的网络连接
 * 
 * 主要步骤：
 * 1. Bootstrap初始化：建立进程间通信机制
 * 2. AllGather1：收集所有rank的peerInfo（hostHash、busId、compCap等）
 * 3. 拓扑检测：检测GPU与NIC之间的连接关系
 * 4. CPU亲和性设置：绑定到与GPU相近的CPU
 * 5. 启动Proxy线程：用于后续的数据移动
 * 6. AllGather2：收集通道数、图信息、拓扑rank
 * 7. 建立传输连接：初始化IB Verbs/Socket等传输层
 * 
 * 注意：我们使用2次AllGather操作
 *   - AllGather1: { peerInfo, comm, compCap }
 *   - AllGather2: { nChannels, graphInfo, topoRanks }
 */
static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  TRACE(NCCL_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  
  // ========== 步骤1: Bootstrap初始化 ==========
  // 建立进程间通信机制（用于多进程场景下的rank发现和同步）
  // 在单进程多GPU场景下，这仍然需要用于rank间的同步
  NCCLCHECK(bootstrapInit(commId, comm));

  // ========== 步骤2: AllGather1 - 收集所有rank的peerInfo ==========
  // 分配peerInfo数组（nranks+1，额外一个用于CollNet root）
  NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks+1)); // Extra rank to represent CollNet root
  
  // 填充当前rank的peerInfo（hostHash、busId、compCap等）
  NCCLCHECK(fillInfo(comm, comm->peerInfo+rank, commHash));
  
  // 通过Bootstrap进行AllGather，收集所有rank的peerInfo
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)));

  // 检查是否有重复的GPU（同一host上相同busId的GPU）
  for (int i = 0; i < nranks; i++) {
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      return ncclInvalidUsage;
    }
  }

  // ========== 步骤3: 拓扑检测和系统图创建 ==========
  //
  // 【步骤3 整体接口】
  // 输入(in):  comm 已分配；comm->peerInfo[0..nRanks-1] 已由 AllGather1 填好（含 busId, hostHash, gdrSupport）。
  // 输出(out): comm->topo 指向可用的 ncclTopoSystem；其内仅含与本 rank 可达的 GPU 及本机 NIC，路径已算好。
  // 前置条件: initTransportsRank 步骤1、2 已完成（bootstrapInit + AllGather1 peerInfo）。
  // 后置条件: comm->topo 可供 ncclTopoCompute（Ring/Tree/CollNet）、ncclTransportP2pSetup 等使用。
  //
  // 3.1 构建拓扑图
  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // 3.2 计算路径（不可达 peer 标记为 count=0，供 Trim 使用）
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  // 3.3 修剪拓扑（去掉不可达 GPU 与单机未用 NIC）
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // 3.4 修剪后重算路径
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm));
  // 3.5 初始化拓扑搜索器
  NCCLCHECK(ncclTopoSearchInit(comm->topo));
  // 3.6 打印拓扑（调试）
  NCCLCHECK(ncclTopoPrint(comm->topo));

  // ========== 步骤4: CPU亲和性设置 ==========
  // 设置CPU亲和性到与GPU相近的CPU，这样我们分配的所有Host内存都是本地的
  // 这可以提高NUMA性能（减少跨NUMA节点的内存访问）
  NCCLCHECK(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity));
  cpu_set_t affinitySave;
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }
  ncclResult_t ret;

  // ========== 步骤5: 启动Proxy服务线程 ==========
  // Proxy线程负责：
  // - 处理网络通信（调用IB Verbs API如ibv_post_send）
  // - 管理数据移动（Host <-> GPU <-> Network）
  // - 处理异步操作
  NCCLCHECK(ncclProxyCreate(comm));

  // Get rings and trees
  struct ncclTopoGraph ringGraph;
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECK(ncclTopoCompute(comm->topo, &ringGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &ringGraph));

  struct ncclTopoGraph treeGraph;
  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.collNet = 0;
  treeGraph.minChannels = 1;
  treeGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &treeGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &treeGraph));

  struct ncclTopoGraph collNetGraph;
  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &collNetGraph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &collNetGraph));

  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* graphs[3] = { &ringGraph, &treeGraph, &collNetGraph };
    NCCLCHECK(ncclTopoDumpGraphs(comm->topo, 3, graphs));
  }

  // Determine local CollNet support before all-gather
  if (collNetSupport(comm)) {
    char *collNetEnable = getenv("NCCL_COLLNET_ENABLE");
    if (collNetEnable != NULL) {
      INFO(NCCL_ALL, "NCCL_COLLNET_ENABLE set by environment to %s.", collNetEnable);
      if (strcmp(collNetEnable, "1") == 0) {
        comm->collNetSupport = 1;
      }
    }
  }
  if (comm->collNetSupport == 1 && collNetGraph.nChannels <= 0) comm->collNetSupport = 0;

  // AllGather3 - begin
  struct ncclGraphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float bwIntra;
    float bwInter;
    int typeIntra;
    int typeInter;
  };

  struct {
    int netDev;
    int collNetSupport;
    struct ncclGraphInfo tree;
    struct ncclGraphInfo ring;
    struct ncclGraphInfo collNet;
    struct ncclTopoRanks topoRanks;
  } *allGather3Data;

  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));

  NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, &allGather3Data[rank].netDev));
  allGather3Data[rank].tree.pattern = treeGraph.pattern;
  allGather3Data[rank].tree.nChannels = treeGraph.nChannels;
  allGather3Data[rank].tree.sameChannels = treeGraph.sameChannels;
  allGather3Data[rank].tree.bwIntra = treeGraph.bwIntra;
  allGather3Data[rank].tree.bwInter = treeGraph.bwInter;
  allGather3Data[rank].tree.typeIntra = treeGraph.typeIntra;
  allGather3Data[rank].tree.typeInter = treeGraph.typeInter;
  allGather3Data[rank].ring.pattern = ringGraph.pattern;
  allGather3Data[rank].ring.nChannels = ringGraph.nChannels;
  allGather3Data[rank].ring.sameChannels = ringGraph.sameChannels;
  allGather3Data[rank].ring.bwIntra = ringGraph.bwIntra;
  allGather3Data[rank].ring.bwInter = ringGraph.bwInter;
  allGather3Data[rank].ring.typeIntra = ringGraph.typeIntra;
  allGather3Data[rank].ring.typeInter = ringGraph.typeInter;
  allGather3Data[rank].collNet.pattern = collNetGraph.pattern;
  allGather3Data[rank].collNet.nChannels = collNetGraph.nChannels;
  allGather3Data[rank].collNet.sameChannels = collNetGraph.sameChannels;
  allGather3Data[rank].collNet.bwIntra = collNetGraph.bwIntra;
  allGather3Data[rank].collNet.bwInter = collNetGraph.bwInter;
  allGather3Data[rank].collNet.typeIntra = collNetGraph.typeIntra;
  allGather3Data[rank].collNet.typeInter = collNetGraph.typeInter;
  allGather3Data[rank].collNetSupport = comm->collNetSupport;

  comm->nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
  NCCLCHECK(ncclTopoPreset(comm, &treeGraph, &ringGraph, &allGather3Data[rank].topoRanks));

  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Determine nNodes, firstRanks, ...
  int *nodesFirstRank, *nodesTreePatterns;
  NCCLCHECK(ncclCalloc(&nodesFirstRank, nranks));
  NCCLCHECK(ncclCalloc(&nodesTreePatterns, nranks));
  NCCLCHECK(ncclCalloc(&comm->rankToNode, comm->nRanks));
  for (int r=0; r<nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[r].tree.pattern;
    }
    comm->rankToNode[r] = node;
  }
  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECK(ncclCalloc(&comm->nodeRanks, comm->nNodes));
  NCCLCHECK(ncclCalloc(&comm->rankToLocalRank, comm->nRanks));
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECK(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks));
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    return ncclInternalError;
  }

  int nChannelsOrig = comm->nChannels;
  struct ncclTopoRanks** allTopoRanks;
  NCCLCHECK(ncclCalloc(&allTopoRanks, comm->nRanks));
  for (int i=0; i<nranks; i++) {
    comm->peerInfo[i].netDev = allGather3Data[i].netDev;
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph.nChannels = std::min(allGather3Data[i].tree.nChannels, treeGraph.nChannels);
    treeGraph.sameChannels = std::min(allGather3Data[i].tree.sameChannels, treeGraph.sameChannels);
    treeGraph.bwIntra = std::min(allGather3Data[i].tree.bwIntra, treeGraph.bwIntra);
    treeGraph.bwInter = std::min(allGather3Data[i].tree.bwInter, treeGraph.bwInter);
    treeGraph.typeIntra = std::max(allGather3Data[i].tree.typeIntra, treeGraph.typeIntra);
    treeGraph.typeInter = std::max(allGather3Data[i].tree.typeInter, treeGraph.typeInter);
    ringGraph.nChannels = std::min(allGather3Data[i].ring.nChannels, ringGraph.nChannels);
    ringGraph.sameChannels = std::min(allGather3Data[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.bwIntra = std::min(allGather3Data[i].ring.bwIntra, ringGraph.bwIntra);
    ringGraph.bwInter = std::min(allGather3Data[i].ring.bwInter, ringGraph.bwInter);
    ringGraph.typeIntra = std::max(allGather3Data[i].ring.typeIntra, ringGraph.typeIntra);
    ringGraph.typeInter = std::max(allGather3Data[i].ring.typeInter, ringGraph.typeInter);
    collNetGraph.nChannels = std::min(allGather3Data[i].collNet.nChannels, collNetGraph.nChannels);
    collNetGraph.sameChannels = std::min(allGather3Data[i].collNet.sameChannels, collNetGraph.sameChannels);
    collNetGraph.bwIntra = std::min(allGather3Data[i].collNet.bwIntra, collNetGraph.bwIntra);
    collNetGraph.bwInter = std::min(allGather3Data[i].collNet.bwInter, collNetGraph.bwInter);
    collNetGraph.typeIntra = std::max(allGather3Data[i].collNet.typeIntra, collNetGraph.typeIntra);
    collNetGraph.typeInter = std::max(allGather3Data[i].collNet.typeInter, collNetGraph.typeInter);
    comm->collNetSupport = std::min(allGather3Data[i].collNetSupport, comm->collNetSupport);
  }

  comm->nChannels = treeGraph.nChannels = ringGraph.nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->collNetSupport == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->collNetSupport = 0;
    }
    for (int n=0; n<comm->nNodes; n++) {
      if (comm->nodeRanks[n].localRanks > NCCL_MAX_DIRECT_ARITY+1) {
        WARN("CollNet currently only supports up to %d GPUs per node, disabling CollNet", NCCL_MAX_DIRECT_ARITY+1);
        comm->collNetSupport = 0;
        break;
      }
    }
  }

  int *rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));
  NCCLCHECK(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, &collNetGraph));

  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather3Data);

  // AllGather3 - end

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %02d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank, comm->channels[c].ring.next);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  NCCLCHECK(computeBuffSizes(comm));

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, affinity_restore);
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, 0), ret, affinity_restore);
  free(rings);
  INFO(NCCL_INIT, "Connected all rings");

  // Connect Trees
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, affinity_restore);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, affinity_restore);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &treeGraph, 0), ret, affinity_restore);
  INFO(NCCL_INIT, "Connected all trees");

  // Check if we can setup CollNet
  if (comm->collNetSupport > 0) {
    int collNetSetupFail = 0;
    int highestTypes[NCCL_MAX_LOCAL_RANKS] = {TRANSPORT_P2P};
    // Find all head ranks
    int nHeads = collNetGraph.nChannels;
    int *heads;
    NCCLCHECK(ncclCalloc(&heads, nHeads));
    // Head GPU index is always 0
    for (int c=0; c<nHeads; c++) {
      heads[c] = collNetGraph.intra[c*comm->localRanks+0];
    }
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      for (int h=0; h<nHeads; h++) {
        const int head = heads[h];
        collNetSetupFail = ncclTransportCollNetSetup(comm, &collNetGraph, channel, head, head, h, collNetRecv);
        if (!collNetSetupFail) collNetSetupFail = ncclTransportCollNetSetup(comm, &collNetGraph, channel, head, head, h, collNetSend);
      }
      // Verify CollNet setup across ranks after trying the first channel
      if (c == 0) {
        NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, collnet_cleanup);
      }
    }
    // Verify CollNet setup across ranks after trying all channels
    NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, collnet_cleanup);
    TRACE(NCCL_INIT, "rank %d Connected inter-node CollNet", rank);

    char line[1024];
    line[0]='\0';
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclTree* chain = &comm->channels[c].collnetChain;
      snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d->%d->%d",
          c, chain->down[0], rank, chain->up);
    }
    line[1023] = '\0';
    INFO(NCCL_INIT, "Collnet Chains %s", line);
    // Connect Collnet + chain
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->collnetChain.up, 1, channel->collnetChain.down, 0), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 0), ret, collnet_cleanup);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, channel->collnetChain.down, 1, &channel->collnetChain.up, 1), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 1), ret, collnet_cleanup);
    INFO(NCCL_INIT, "Connected collnet + chain");

    // Connect intra-node CollNet + Direct
    int highestTransportType0, highestTransportType1;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelRecv = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.up, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.down, 0), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 0, &highestTransportType0), ret, collnet_cleanup);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channelSend = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.down, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.up, 1), ret, collnet_cleanup);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &collNetGraph, 1, &highestTransportType1), ret, collnet_cleanup);

    // Exchange highest intra-node transport type among ranks
    // because we need to know whether all ranks can p2p each other to determine whether we can directly read/write registered user buffer
    comm->intraHighestTransportType = highestTypes[comm->localRank] = highestTransportType0 > highestTransportType1 ? highestTransportType0 : highestTransportType1;
    NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, highestTypes, sizeof(int)));
    for (int i=0; i<comm->localRanks; i++) {
      if (highestTypes[i] > comm->intraHighestTransportType)
        comm->intraHighestTransportType = highestTypes[i];
    }
    INFO(NCCL_INIT, "rank %d Connected CollNet", rank);

collnet_cleanup:
    free(heads);
    if (ret != ncclSuccess) {
      NCCLCHECK(ncclTransportCollNetFree(comm));
      comm->collNetSupport = 0;
      ret = ncclSuccess;
    }
  }
  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  do {
    int myCompCap = comm->peerInfo[rank].cudaCompCap;
    int minCompCap = myCompCap, maxCompCap = myCompCap;
    for (int i = 0; i < nranks; i++) {
      minCompCap = std::min(comm->peerInfo[i].cudaCompCap, minCompCap);
      maxCompCap = std::max(comm->peerInfo[i].cudaCompCap, maxCompCap);
    }
    NCCLCHECK(ncclTopoTuneModel(comm, minCompCap, maxCompCap, &treeGraph, &ringGraph, &collNetGraph));
  } while(0);

  // Compute nChannels per peer for p2p
  NCCLCHECK(ncclTopoComputeP2pChannels(comm));

  do { // Setup p2p structures in comm->tasks
    struct ncclTasks* tasks = &comm->tasks;
    int nRanks = comm->nRanks;
    int node = comm->node;
    int nNodes = comm->nNodes;
    struct ncclNodeRanks *nodeRanks = comm->nodeRanks;
    int localRank = comm->localRank;
    tasks->peers = ncclMemoryStackAlloc<ncclTasks::Peer>(&comm->memPermanent, nRanks);
    tasks->p2pSendOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
    tasks->p2pRecvOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
    int s=0, r=0;
    // schedule delta 0, +1, -1, +2, -2, ...
    // also make sure we don't do 0 twice, nor +n/2 and -n/2 if n is even.
    for (int d=0; d <= nNodes/4; d++) {
      int deltas[4] = { d, (nNodes-d)%nNodes, nNodes/2-d, (nNodes-(nNodes/2-d))%nNodes };
      int index = 0;
      int delta = deltas[index];
    sched_delta:
      int recvNode = (node+nNodes-delta)%nNodes;
      int sendNode = (node+delta)%nNodes;
      int steps = comm->maxLocalRanks;
      for (int step=0; step < steps; step++) {
        int recvIndex = (localRank-step+steps)%steps;
        if (recvIndex < nodeRanks[recvNode].localRanks) {
          tasks->p2pRecvOrder[r] = nodeRanks[recvNode].localRankToRank[recvIndex];
          r++;
        }
        int sendIndex = (localRank+step)%steps;
        if (sendIndex < nodeRanks[sendNode].localRanks) {
          tasks->p2pSendOrder[s] = nodeRanks[sendNode].localRankToRank[sendIndex];
          s++;
        }
      }
      index++;
      if (index == 1 && deltas[1] == deltas[0]) index++;
      if (index == 2 && deltas[2] == deltas[0]) index++;
      if (index == 3 && deltas[3] == deltas[2]) index++;
      if (index == 3 && deltas[3] == deltas[1]) index++;
      if (index < 4) {
        delta = deltas[index];
        goto sched_delta;
      }
    }
    assert(s == nRanks && r == nRanks);
  } while (0);

  if (ncclParamNvbPreconnect()) {
    // Connect p2p when using NVB path
    int nvbNpeers;
    int* nvbPeers;
    NCCLCHECK(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers));
    for (int r=0; r<nvbNpeers; r++) {
      int peer = nvbPeers[r];
      int channelId;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECK(ncclChannelCompute(comm, peer, c, ncclFuncSend, &channelId));
        if (comm->channels[channelId].peers[peer].send[1].connected == 0) {
          comm->connectSend[peer] |= (1<<channelId);
        }
      }
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECK(ncclChannelCompute(comm, peer, c, ncclFuncRecv, &channelId));
        if (comm->channels[channelId].peers[peer].recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1<<channelId);
        }
      }
    }
    NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
    free(nvbPeers);
  }

  // Connect to local net proxy
  struct ncclProxyConnector proxyConn;
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn));
  NCCLCHECK(ncclProxyCall(&proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0));

  // Then to remote ones when using PXN
  if (ncclPxnDisable(comm) == 0) {
    int nranks;
    int* pxnPeers;
    NCCLCHECK(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks));
    for (int r=0; r<nranks; r++) {
      NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn));
      NCCLCHECK(ncclProxyCall(&proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0));
    }
    free(pxnPeers);
  }

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
    for (int i = 0; i < nranks; i++) {
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash)
          && (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }
    }
    TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      return ncclInternalError;
    }
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    assert(intraProcRank==0 ? comm==comm0 : true);
    comm->intraComm0 = comm0;
    comm->intraRefs = intraProcRank==0 ? intraProcRanks : 0;
    comm->intraRank = intraProcRank;
    comm->intraRanks = intraProcRanks;
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while(0);

  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    char* str = getenv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  /* Local intra-node barrier */
  NCCLCHECK(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]));

  // Unlink proxy shm to make sure it will be properly cleaned up.
  NCCLCHECK(ncclProxyShmUnlink(comm));

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
affinity_restore:
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  if (ret != ncclSuccess) return ret;

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);
  return ncclSuccess;
}

NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);

struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t* newcomm;
  int nranks, myrank;
  ncclUniqueId commId;
  int cudaDev;
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

/**
 * ncclCommInitRankFunc: 异步初始化函数的实际执行体
 * 
 * 这是真正执行通信器初始化的函数，在ncclAsyncLaunch中被调用
 * 主要完成三个核心步骤：
 * 1. commAlloc: 分配通信器资源
 * 2. initTransportsRank: 初始化传输层（建立IB/Socket连接）
 * 3. devCommSetup: 设置设备端通信器
 */
static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t* newcomm = job->newcomm;
  ncclComm_t comm = *newcomm;
  int nranks = job->nranks;
  ncclUniqueId commId = job->commId; // C++ struct assignment
  int myrank = job->myrank;
  int cudaDev = job->cudaDev;
  ncclResult_t res = ncclSuccess;

  // ========== 步骤1: 设置CUDA设备 ==========
  // 切换到指定的CUDA设备
  CUDACHECK(cudaSetDevice(cudaDev));
  
  // 设置Kernel栈大小限制，避免CUDA内存重配置（参考NVSHMEM问题）
  // 这可以避免在加载时触发CUDA内存重新配置
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zi", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }
  
  // ========== 步骤2: 分配通信器资源 ==========
  // commAlloc主要工作：
  // - 初始化ncclComm结构体（rank、nRanks等）
  // - 初始化网络层（ncclNetInit）
  // - 创建CUDA Stream（deviceStream、hostStream）
  // - 获取GPU Bus ID
  // - 初始化内存池和通道
  NCCLCHECKGOTO(commAlloc(newcomm, nranks, myrank), res, cleanup);
  
  // ========== 步骤3: 初始化传输层 ==========
  // initTransportsRank是核心函数，负责：
  // - Bootstrap初始化（建立进程间通信）
  // - AllGather收集所有rank的peerInfo
  // - 拓扑检测（GPU与NIC之间的路径）
  // - 启动Proxy线程
  // - 建立传输连接（IB Verbs/Socket等）
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);
  
  // ========== 步骤4: 设备端通信器设置 ==========
  // devCommSetup主要工作：
  // - 获取CUDA Stream
  // - 分配设备端ncclDevCommAndChannels结构
  // - 将通信器信息复制到GPU可访问内存
  NCCLCHECKGOTO(devCommSetup(*newcomm), res, cleanup);

  // ========== 步骤5: 标记初始化完成 ==========
  // 更新通信器状态为成功
  comm->initState = ncclSuccess;

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - Init COMPLETE", *newcomm, myrank, nranks, (*newcomm)->cudaDev, (*newcomm)->busId);
  TRACE_CALL("ncclCommInitRank(%p,%d,0x%llx,%d,%d)", *newcomm, nranks, (unsigned long long)hashUniqueId(commId), myrank, (*newcomm)->cudaDev);
  return ncclSuccess;
  
cleanup:
  // 如果初始化失败，记录错误状态
  comm->initState = res;
  return res;
}

/**
 * parseCommConfig: 解析并设置通信器配置
 * 
 * 功能：从config参数中读取配置并设置到通信器中
 * 
 * 当前支持的配置：
 *   - blocking: 是否使用阻塞模式
 *     * blocking=1: 阻塞模式，操作会等待完成
 *     * blocking=0: 非阻塞模式，操作异步执行
 * 
 * 默认配置：blocking=1（阻塞模式）
 */
static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;

  /* 首先设置配置 */
  if (config) {
    // 如果提供了配置，使用配置中的值
    comm->blocking = config->blocking;
  } else {
    /* 通信器的默认设置 */
    // 默认使用阻塞模式
    comm->blocking = 1;
  }

  return ret;
}

static void ncclCommInitRankUndo(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclCommDestroy(*job->newcomm);
  *job->newcomm = nullptr;
}

/**
 * ncclCommInitRankDev: 单个rank的通信器初始化（设备级别）
 * 
 * 这是每个rank初始化的核心函数，负责：
 * 1. 创建通信器对象
 * 2. 设置初始状态
 * 3. 异步启动实际的初始化任务
 * 
 * 参数：
 *   - newcomm: 输出参数，返回创建的通信器
 *   - nranks: 通信组中的总rank数
 *   - commId: 唯一通信ID（所有rank共享）
 *   - myrank: 当前rank的ID
 *   - cudaDev: CUDA设备ID
 *   - config: 可选的配置参数
 */
static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev, ncclConfig_t *config) {
  ncclResult_t res;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob *job = NULL;
  
  // ========== 步骤1: 环境变量检查 ==========
  // 如果设置了NCCL_COMM_ID环境变量，使用环境变量中的ID（通常用于调试）
  char* env = getenv("NCCL_COMM_ID");
  if (env && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    NCCLCHECKGOTO(bootstrapCreateRoot(&commId, true), res, fail);
  }

  // ========== 步骤2: 全局初始化 ==========
  // ncclInit()是全局初始化函数，只会执行一次（使用原子变量和互斥锁保护）
  // 初始化内容：GDR、网络插件、Bootstrap网络等
  NCCLCHECKGOTO(ncclInit(), res, fail);
  if (myrank == 0) showVersion();

  // 确保CUDA运行时已初始化（通过调用cudaFree(NULL)）
  CUDACHECKGOTO(cudaFree(NULL), res, fail);

  // ========== 步骤3: 参数校验 ==========
  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  // ========== 步骤4: 创建通信器对象 ==========
  // 分配通信器结构体内存
  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);
  
  // 分配GPU可访问的主机内存，用于abortFlag（GPU可以读取此标志来检查是否需要中止）
  // 
  // 为什么必须用 ncclCudaHostCalloc 而不是 malloc？
  // 1. GPU Kernel在执行时会定期检查 abortFlag（见 src/collectives/device/common.h:198）
  // 2. GPU必须能直接通过PCIe读取这个标志，不能等CPU复制（实时性要求）
  // 3. ncclCudaHostCalloc分配的是Pinned Memory（固定内存），GPU可以直接访问
  // 4. 如果用malloc，GPU无法直接访问，需要cudaMemcpy复制，延迟高且可能错过中止信号
  // 
  // 类比：这是CPU和GPU之间的"共享公告板"，双方都能直接看到，而不是"私人笔记本"
  NCCLCHECKGOTO(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1), res, fail);
  *comm->abortFlag = 0;  // 初始化为0（无中止）
  
  // 解析并设置通信器配置（如blocking模式）
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  
  // 初始状态设为错误，只有初始化成功后才改为ncclSuccess
  comm->initState = ncclInternalError;
  *newcomm = comm;

  // ========== 步骤5: 创建异步初始化任务 ==========
  // 分配异步任务结构体
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->newcomm = newcomm;
  job->nranks = nranks;
  job->commId = commId; // C++ struct assignment
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  
  // 异步启动初始化任务
  // 如果不在组内（ncclGroupDepth == 0）：立即执行ncclCommInitRankFunc
  // 如果在组内（ncclGroupDepth > 0）：将任务加入队列，等待ncclGroupEnd时统一执行
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitRankFunc, NULL, free, comm), res, fail);

exit:
  return ncclGroupErrCheck(res);
fail:
  goto exit;
}

/**
 * ncclCommInitRank: 多进程/多线程初始化函数
 * 
 * 功能：为单个rank创建通信器（多进程或多线程场景）
 * 
 * 与ncclCommInitAll的区别：
 * - ncclCommInitAll: 单进程多GPU，自动处理所有GPU
 * - ncclCommInitRank: 多进程/多线程，每个进程/线程只初始化一个rank
 * 
 * 使用场景：
 * - 多进程训练（每个进程一个GPU）
 * - 多线程训练（每个线程一个GPU）
 * - 需要手动管理每个rank的初始化
 * 
 * 注意：所有rank必须使用相同的commId和nranks
 */
NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  // 加载CUDA驱动和dlsym钩子（旧驱动可能失败）
  (void) cudaLibraryInit();

  // 获取当前CUDA设备（每个进程/线程应该已经设置了对应的设备）
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  
  // 调用核心初始化函数（使用当前设备）
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, NULL));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
/**
 * ncclCommInitAll: 单进程多GPU初始化函数
 * 
 * 功能：在单个进程内为多个GPU创建通信器（communicator）
 * 适用场景：单机多卡训练，所有GPU在同一进程内
 * 
 * 参数：
 *   - comms: 输出参数，返回ndev个通信器对象的数组
 *   - ndev: 要初始化的GPU数量
 *   - devlist: 可选的GPU设备ID列表，如果为NULL则使用前ndev个GPU
 */
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int *gpuFlags = NULL;
  
  // ========== 阶段1: 前置检查与初始化 ==========
  // 1. CUDA库初始化 - 加载CUDA驱动和dlsym钩子（旧驱动可能失败）
  (void) cudaLibraryInit();
  
  // 2. 参数校验 - 检查comms指针是否有效
  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }
  
  // 3. 获取系统GPU总数 - 用于后续的设备ID校验
  CUDACHECKGOTO(cudaGetDeviceCount(&totalnDev), ret, fail);
  
  // 4. 如果提供了devlist，进行设备列表校验
  if (devlist) {
    // 分配标志数组，用于检查重复和无效设备
    NCCLCHECKGOTO(ncclCalloc(&gpuFlags, totalnDev), ret, fail);
    for (int i = 0; i < ndev; ++i) {
      /* invalid device check. */
      // 检查设备ID是否有效（必须在[0, totalnDev)范围内）
      if (devlist[i] < 0 || devlist[i] >= totalnDev) {
        ret = ncclUnhandledCudaError;
        goto fail;
      }

      /* duplicate device check. */
      // 检查是否有重复设备（同一个GPU被指定多次）
      if (gpuFlags[devlist[i]] != 0) {
        ret = ncclInvalidUsage;
        goto fail;
      }

      gpuFlags[devlist[i]] = 1;  // 标记该设备已被使用
    }
    free(gpuFlags);
  }

  // ========== 阶段2: 生成唯一通信ID ==========
  // 生成一个128字节的唯一ID，用于标识这个通信组
  // 在单进程多GPU场景下，所有GPU共享同一个uniqueId
  ncclUniqueId uniqueId;
  NCCLCHECKGOTO(ncclGetUniqueId(&uniqueId), ret, fail);
  
  // ========== 阶段3: 组语义包装 - 同步初始化多个通信器 ==========
  // 使用ncclGroupStart/End确保所有通信器同步初始化
  // 这是因为在单线程中初始化多个通信器需要同步机制
  NCCLCHECKGOTO(ncclGroupStart(), ret, fail);
  
  // 为每个GPU创建通信器
  // 注意：这里使用组语义，所有初始化任务会被加入队列，在ncclGroupEnd时统一执行
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    // 如果devlist为NULL，使用设备i；否则使用devlist[i]
    ncclCommInitRankDev(comms+i, ndev, uniqueId, i, devlist ? devlist[i] : i, NULL);
  }
  
  // 执行所有初始化任务（如果不在组内，ncclAsyncLaunch会立即执行）
  NCCLCHECKGOTO(ncclGroupEnd(), ret, fail);

exit:
  return ret;
fail:
  if (gpuFlags) free(gpuFlags);
  goto exit;
}

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr;
  size_t realSize;
  int blockingEnv;

  NCCLCHECK(ncclGroupStartInternal());
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != 0xcafebeef) {
      WARN("ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER");
      ret = ncclInvalidArgument;
      goto exit;
    }
  }

  /* check input config attributes */
  if (internalConfigPtr->blocking != 0 && internalConfigPtr->blocking != 1) {
    WARN("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto exit;
  }

  /* overwrite configuration from env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv != 0 && blockingEnv != 1) {
    WARN("Invalid NCCL_COMM_BLOCKING value %d", blockingEnv);
  }
  if (blockingEnv == 1) internalConfigPtr->blocking = blockingEnv;

  (void) cudaLibraryInit();
  CUDACHECKGOTO(cudaGetDevice(&cudaDev), ret, exit);
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, internalConfigPtr), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm && !(*newcomm)->blocking) (void) ncclCommGetAsyncError(*newcomm, &ret);
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  int savedDevice;
  CUDACHECK(cudaGetDevice(&savedDevice));
  int commDevice = comm->cudaDev;
  ncclResult_t ret;

  CUDACHECKGOTO(cudaGetDevice(&savedDevice), ret, fail);
  if (savedDevice != commDevice) {
    CUDACHECKGOTO(cudaSetDevice(commDevice), ret, fail);
  }

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->hostStream), ret, fail);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->deviceStream), ret, fail);
  }
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);
  // And keep polling until all graphs referencing us die.
  while (comm->persistentRefs != 0) {
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
  }

  if (savedDevice != commDevice) {
    CUDACHECKGOTO(cudaSetDevice(savedDevice), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  int savedDevice;
  int commDevice = comm->cudaDev;

  CUDACHECK(cudaGetDevice(&savedDevice));
  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(savedDevice));
  }

  return ncclSuccess;
}

static ncclResult_t commFinalize(ncclComm_t comm, bool userCalled) {
  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  comm->finalizeCalled = true;
  /* launch async thread to finalize comm. */
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;

  if (userCalled) {
    NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, commDestroySync, NULL, free, comm), ret, fail);
  } else {
    NCCLCHECKGOTO(commDestroySync(&job->base), ret, fail);
    free(job);
  }

exit:
  return ncclGroupErrCheck(ret);
fail:
  if (job) free(job);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* finalize comm. */
  ret = commFinalize(comm, true);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm && !comm->blocking) { NCCLCHECK(ncclCommGetAsyncError(comm, &ret)) };
  return ret;
fail:
  if (comm && !comm->blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  ncclResult_t state;
  int curRank; /* Debug info */

  NCCLCHECKGOTO(ncclCommGetAsyncError(comm, &state), ret, fail);
  TRACE(NCCL_INIT, "commReclaim: reclaim comm %p rank %d state %d", comm, comm->rank, state);
  if (state == ncclSuccess && *comm->abortFlag == 0 && comm->finalizeCalled == false) {
    /* user does not call ncclCommFinalize and this is a normal comm destroy. ncclCommDestroy
     * should be nonblocking until last call of ncclCommDestroy. */
    NCCLCHECKGOTO(commFinalize(comm, false), ret, fail);
  }

  if (comm->initState != ncclSuccess) {
    /* if init errors happen, no finalize thread should have been launched. Main thread can reclaim
     * everything since no NCCL kernel was issued. */
    struct ncclCommFinalizeAsyncJob job;

    job.comm = comm;
    curRank = comm->rank;
    /* comm aborts, commDestroySync should not be blocked. */
    if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess) {
      WARN("commReclaim: comm %p (rank = %d) in abort, error %d", comm, curRank, ret);
    }

    if ((ret = commCleanup(comm)) != ncclSuccess) {
      WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", comm, curRank, ret);
    }
  } else {
    int curRankCnt;
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);
    if (curRankCnt == intraRanks) {
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (comm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob job;
          job.comm = curIntraComm;
          /* every comm aborts, commDestroySync should not be blocked. */
          if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess)
            WARN("commReclaim: comm %p (rank = %d) in abort, error %d", curIntraComm, curRank, ret);
        }

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  if (comm == NULL)
    return ncclSuccess;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  int64_t busId = comm->busId;
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, busId);
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  NCCLCHECK(commReclaim(comm));
  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - Destroy COMPLETE", comm, rank, nranks, cudaDev, busId);

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  if (comm == NULL)
    return ncclSuccess;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  int64_t busId = comm->busId;
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, busId);

  // Ask anything that might still be running on the device to quit
  *comm->abortFlag = 1;
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  ncclCommEnsureReady(comm);

  (void) commReclaim(comm);
  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - Abort COMPLETE", comm, rank, nranks, cudaDev, busId);

  return ncclSuccess;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error";
    case ncclSystemError            : return "unhandled system error";
    case ncclInternalError          : return "internal error";
    case ncclInvalidArgument        : return "invalid argument";
    case ncclInvalidUsage           : return "invalid usage";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);
const char* ncclGetLastError(ncclComm_t comm) {
  return ncclLastError;
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(PtrCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}
