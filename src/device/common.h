/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "device.h"
#include "op128.h"
#include "reduce_kernel.h"
#include "network/unpack/unpack_defs.h"

#define COLL_UNROLL (ncclCollUnroll())

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

typedef void (*ncclDevFuncPtr_t)();
#if defined(NCCL_OS_WINDOWS)
/* MSVC C2133: extern array of unknown size needs a complete type; use pointer instead. */
extern __device__ ncclDevFuncPtr_t const* ncclDevFuncTable;
#else
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable[];
#endif

struct ncclShmemGroup {
  ncclConnInfo* recvConns[NCCL_MAX_ARITY];
  ncclConnInfo* sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* srcs[NCCL_MAX_ARITY + 1];
  void* dsts[NCCL_MAX_ARITY + 1];
  union {
    unpackGroupShmem unpack;
  } devicePlugin;
  int32_t dstSizes[NCCL_MAX_ARITY + 1];
  uint64_t redOpArgs;
};

struct ncclShmemData {
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclKernelComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  uint64_t workCounter;
  bool profilerEnabled;
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];

  alignas(16) char workStorage[ncclMaxDevWorkBatchBytes()];

  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
};

extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
extern __shared__ ulong2
  ncclShmemPerWarp[ncclShmemScratchWarpSize() * (NCCL_MAX_NTHREADS / WARP_SIZE) / sizeof(ulong2)];
#endif

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)ncclShmemPerWarp + warp * ncclShmemScratchWarpSize();
}

__device__ inline void barrier_sync(int name) {
#if 0
  asm volatile("barrier.sync %0;" :: "r"(name) : "memory");
#else
  asm volatile("barrier.sync.aligned %0;" ::"r"(name) : "memory");
#endif
}
__device__ inline void barrier_sync(int name, int nThreads) {
#if 0
  asm volatile("barrier.sync %0, %1;" :: "r"(name), "r"(nThreads) : "memory");
#else
  asm volatile("barrier.sync.aligned %0, %1;" ::"r"(name), "r"(nThreads) : "memory");
#endif
}
__device__ inline void barrier_sync_aligned(int name) {
  asm volatile("barrier.sync.aligned %0;" ::"r"(name) : "memory");
}
__device__ inline void barrier_sync_aligned(int name, int nThreads) {
  asm volatile("barrier.sync.aligned %0, %1;" ::"r"(name), "r"(nThreads) : "memory");
}

__device__ inline bool barrier_red_or(bool vote, int name) {
  int ans;
  asm volatile("{ .reg .pred p;"
               "  setp.ne.s32 p, %1, 0;"
               "  barrier.red.or.pred p, %2, p; "
               "  selp.s32 %0, 1, 0, p; }"
               : "=r"(ans)
               : "r"((int)vote), "r"(name)
               : "memory");
  return bool(ans);
}
__device__ inline bool barrier_red_or(bool vote, int name, int nThreads) {
  int ans;
  asm volatile("{ .reg .pred p;"
               "  setp.ne.s32 p, %1, 0;"
               "  barrier.red.or.pred p, %2, %3, p; "
               "  selp.s32 %0, 1, 0, p; }"
               : "=r"(ans)
               : "r"((int)vote), "r"(name), "r"(nThreads)
               : "memory");
  return bool(ans);
}

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16 * tid;
  if (offset < bytes) {
    uint64_t a = 0, b = 0;
    asm volatile("ld.v2.u64 {%0,%1},[%2];" : "=l"(a), "=l"(b) : "l"((char const*)src + offset) : "memory");
    uint32_t udst = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("st.shared.v2.u64 [%0],{%1,%2};" ::"r"(udst + offset), "l"(a), "l"(b) : "memory");
  }
}

// 根据 batch 目录把选中的 Work 描述紧凑加载到本 block 的 shared memory；本函数只准备工作，不执行通信。
// 输入：加载子组的逻辑 tid/线程数 tn、GPU 参数包 args、起始 batch 描述下标 batchIx。
// 输出：ncclShmem.workStorage，以及 workSize/nWorks/funcId/nextBatchIx 等执行控制字段。
// 调用前 ncclShmem.args 已就绪；返回后由 ncclKernelMain 的 __syncthreads() 发布本批加载结果。
//
// 例：offsetBitset 的第 0、2、5 位为 1，只取 offsetBase 起点之后这三份 Work：
//   source index:       0     1     2     3     4     5
//   source Work:      [ W0 ][ W1 ][ W2 ][ W3 ][ W4 ][ W5 ]
//   selected:           ^           ^                 ^
//
//   fnsOfBitset:      [  0 ][  2 ][  5 ]  <- 目标编号 dstWork 映射到源编号 srcWork
//                       |     |     |
//                       v     v     v
//   workStorage:      [ W0 ][ W2 ][ W5 ]
//   destination index:  0     1     2
//   Work 内容是 ncclDevWork* 执行描述，用户输入/输出 buffer 仍由描述中的指针访问。
// Must run with at least 64 threads
__device__ __forceinline__ void loadWorkBatchToShmem(int tid, int tn, struct ncclDevKernelArgs const* args,
                                                     int batchIx) {
  // tid 是加载子组内的逻辑编号，可能从 threadIdx.x 减去前两个 warp 得到；扩展加载时还会轮转。
  int lane = tid % WARP_SIZE;
  int workCursor = 0; // 已加载的 Work 数量，单位是 Work；融合扩展描述时用于追加目标位置。
  while (true) {
    // batch 描述始终紧跟参数头；其 Work 内容可能在 Args、FIFO 或 Persistent 存储中。
    struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(args + 1))[batchIx];

    // ============ 1. 将稀疏 bitset 转换为紧凑目标到源位置的映射 ============
    // fnsOfBitset[n] = index of n'th set bit in batch.offsetBitset.
    // PTX has instruction "fns" (find n-th set) but it expands to a lot of SASS,
    // since we know all lanes will be querying the same bitmask we can compute
    // much faster using shared memory.
    // 每个物理 warp 在自己的 scratch 内构建相同映射；threadIdx.x 保证逻辑 tid 轮转后仍不串用 scratch。
    uint8_t* fnsOfBitset = (uint8_t*)ncclScratchForWarp(threadIdx.x / WARP_SIZE);
    __syncwarp(); // 本 warp 到齐后再复用 scratch，避免覆盖上一轮仍在读取的映射。
    if (uint32_t(batch.offsetBitset) & (1u << lane)) {
      // 低 32 位：选中位置前面有多少个置位 bit，就写到紧凑表的哪个下标。
      // 例：bit 5 前面有 bit 0/2，因此 fnsOfBitset[2]=5。
      int nWorksBelow = __popc(uint32_t(batch.offsetBitset) & ((1u << lane) - 1));
      fnsOfBitset[nWorksBelow] = lane;
    }
    int nWorksLow32 = __popc(uint32_t(batch.offsetBitset)); // just of low 32 bits
    if (uint32_t(batch.offsetBitset >> 32) & (1u << lane)) {
      // 高 32 位由同一组 lane 再处理；目标排名加上低半部数量，源位置加 32。
      int nWorksBelow = nWorksLow32;
      nWorksBelow += __popc(uint32_t(batch.offsetBitset >> 32) & ((1u << lane) - 1));
      fnsOfBitset[nWorksBelow] = 32 + lane;
    }
    int nWorks = nWorksLow32 + __popc(uint32_t(batch.offsetBitset >> 32)); // add high 32 bits
    __syncwarp(); // 映射表在本 warp 内已完整写入，下面的复制线程才可查表。

    // ============ 2. 根据 Work 类型，将复制任务拆成 16B 包并分配给线程 ============
    // 假设 workSize=64B（仅示意），一份 Work 有四包，选中三份共需 12 个复制线程：
    //   tid:         0  1  2  3 | 4  5  6  7 | 8  9 10 11
    //   dstWork:     0  0  0  0 | 1  1  1  1 | 2  2  2  2
    //   packInWork:  0  1  2  3 | 0  1  2  3 | 0  1  2  3
    //   source:          W0     |     W2     |     W5
    // 每个复制线程搬一包，多个线程协作复制一份 Work；16B 是复制粒度，不是原子发布保证。
    int workSize;
    int nPacks; // total number of packs loaded, each pack is 16 bytes
    int packInWork; // my pack index within work struct
    int dstWork; // my work index in contiguous destination shmem
    switch (batch.workType) {
    case (int)ncclDevWorkTypeP2p:
      workSize = sizeof(struct ncclDevWorkP2p);
      nPacks = nWorks * (workSize / 16);
      packInWork = tid % (workSize / 16);
      dstWork = tid / (workSize / 16);
      break;
    case (int)ncclDevWorkTypeColl:
      workSize = sizeof(struct ncclDevWorkColl);
      nPacks = nWorks * (workSize / 16);
      packInWork = tid % (workSize / 16);
      dstWork = tid / (workSize / 16);
      break;
    case (int)ncclDevWorkTypeBcast:
      workSize = sizeof(struct ncclDevWorkBcast);
      nPacks = nWorks * (workSize / 16);
      packInWork = tid % (workSize / 16);
      dstWork = tid / (workSize / 16);
      break;
    case (int)ncclDevWorkTypeCollReg:
    default:
      workSize = sizeof(struct ncclDevWorkCollReg);
      nPacks = nWorks * (workSize / 16);
      packInWork = tid % (workSize / 16);
      dstWork = tid / (workSize / 16);
      break;
    }
    if (tid == 0) {
      // 后续执行函数按此大小遍历 workStorage 中的连续 Work。
      ncclShmem.workSize = workSize;
    }
    // We deliberately replicate these div and mod calculations into the case
    // blocks above so that they get constant divisor optimizations by the compiler.
    //   packInWork = tid%(workSize/16);
    //   dstWork = tid/(workSize/16);

    // ============ 3. 定位源 Work 的包，复制到紧凑 shared memory 位置 ============
    // We can only assume we have 64 threads, which means we can read at most 1024 bytes
    // here which is the per batch maximum.
    if (tid < nPacks) {
      int srcWork = fnsOfBitset[dstWork]; // find n'th set bit in batch.offsetBitset
      ulonglong2 tmp;
      // 源字节偏移 = offsetBase + srcWork*workSize + packInWork*16。
      // Args 直接从参数空间加载；FIFO/Persistent 从 workBuf 加载，FIFO 还需 mask 回绕。
      // 两条加载路径刻意分开，保留 ld.param；共用通用源指针可能触发参数结构的 per-thread spill。
      // The loads done in these two cases must be kept separate since we are
      // relying on the compiler to use "ld.param" in the first one. The parameter
      // space is not generically addressable, so any attempt to load through
      // a pointer that *might* be parameter space backed will cause the
      // compiler to spill the parameter struct (4K!) to each thread's local space
      // before creating a pointer (to the spill) and decimate perf.
      //
      // An example of what not to do would be the following:
      //
      // if (condition) {
      //   // The compiler could spill parameter_variable to local space and take
      //   // the address of that, since when src is loaded below it could also
      //   // be global space.
      //   src = &parameter_variable;
      // } else {
      //   src = &global_variable;
      // }
      // memcpy(dst, src, n);
      if (ncclShmem.args.workStorageType == ncclDevWorkStorageTypeArgs) {
        char* src = (char*)args + (batch.offsetBase + srcWork * workSize + packInWork * 16);
        tmp = *(ulonglong2*)src; // becomes ld.param.v2.u64
      } else {
        char* src = (char*)ncclShmem.args.workBuf +
                    ((batch.offsetBase + srcWork * workSize + packInWork * 16) & ncclShmem.args.workMask);
        tmp = *(ulonglong2*)src; // becomes ld.v2.u64
      }
      // 目标按 dstWork 紧凑排列；workCursor 使扩展描述的 Work 接在已加载内容之后。
      char* dst = ncclShmem.workStorage;
      dst += (workCursor + dstWork) * workSize + packInWork * 16;
      *(ulonglong2*)dst = tmp;
    }
    workCursor += nWorks; // 当前描述加载完成，累计本次逻辑批次的 Work 数量。

    // ============ 4. 融合扩展描述，或完成加载并给出下一逻辑批次入口 ============
    // 描述链： A(nextExtends=1) --> A'(nextExtends=0) --> B
    //           load A Works       append A' Works      留给下次调用
    // shared:  [ A Works ][ A' Works ]，总数存入 ncclShmem.nWorks；返回后一起交给 batch 执行函数。
    // nextJump 负责找后继描述；nextExtends 决定后继是本次融合加载，还是下一次执行的批次。
    if (batch.nextExtends) {
      batchIx += batch.nextJump;
      // 在同一加载子组内轮转逻辑编号，把下一描述的前 64 个复制位置交给下一组两个 warp。
      // 物理 threadIdx.x 不变；tn=64 时轮转后逻辑编号也不变。
      tid -= 64; // Rotate threads so we use the next two warps for next batch struct.
      if (tid < 0) tid += tn;
    } else {
      if (tid == 0) {
        // 记录扩展链最后一个描述的下标；下一批入口从此处计算，nextJump=0 转成结束标记 -1。
        ncclShmem.batchIx = batchIx;
        ncclShmem.nextBatchIx = (batch.nextJump == 0) ? -1 : (int)(batchIx + batch.nextJump);
        ncclShmem.workType = (enum ncclDevWorkType)batch.workType;
        ncclShmem.nWorks = workCursor; // 整条扩展链的 Work 总数；funcId/workType 在同一融合批内一致。
        ncclShmem.funcId = batch.funcId;
      }
      break; // 本次加载结束；由调用者做 block barrier，再读取 Work 并执行。
    }
  }
}

__device__ __forceinline__ unsigned long long int globaltimer() {
  unsigned long long int timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

template <ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkColl {
  __device__ void run(int tid, int tn, struct ncclDevWorkColl* work) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template <ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch;

// Specialized for P2p in sendrecv.h
template <typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>;

template <typename T, typename RedOp, int Proto>
struct RunWorkBatch<ncclFuncAllGatherV, T, RedOp, NCCL_ALGO_RING, Proto>;

#define START 0
#define STOP 1
#define FINI 2

__device__ __forceinline__ bool profilerEnabled(int workItemIdx) {
  return (ncclShmem.workType == ncclDevWorkTypeP2p) ?
           ((struct ncclDevWorkP2p*)ncclShmem.workStorage)[workItemIdx].profilerEnabled :
           ((struct ncclDevWorkColl*)ncclShmem.workStorage)[workItemIdx].profilerEnabled;
}

__device__ __forceinline__ void profilerPhase(int phaseId) {
  uint64_t ts = globaltimer();
  int idx = 0;
  uint64_t wc = ncclShmem.channel.workCounter + 1;
  for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
    if (!profilerEnabled(idx++)) continue;
    int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;
    ncclShmem.comm.workPhases[ncclShmem.channelId].data[slot].timestamps[phaseId] = ts;
  }
}

// Specialized here for non-P2p (Coll and CollReg)
// 当前 block（= 一个 channel）的 collective 批次派发器：把 loadWorkBatchToShmem 已紧凑放入
// ncclShmem.workStorage 的 nWorks 份 Work，按下标顺序逐份交给 RunWorkColl 执行。
//
// ---- (a) 编译期 vs 运行期 ----
// Fn/T/RedOp/Algo/Proto 是模板参数，由 ncclKernelMain 按 batch.funcId 选中的实例化固定，
// 即「做什么集合、什么类型、什么归约、走哪种算法/协议」在进入本函数前已确定；
// 每份 Work 只携带运行期数据：sendbuff/recvbuff、各 channel 的数据量划分、redOpArg、nWarps。
//
// ---- (b) 执行模型 ----
// 同一批内的 Coll Work 在本 block 内串行执行（P2p 走 sendrecv.h 的特化，按 warp 并行）。
// 每份 Work 只用前 nWarps 个完整 warp；blockDim.x 按整个 plan 中最大的 nWarps 设定，
// 因此 nWarps 较小的 Work 执行时，其余线程空转等待。
// 本函数只负责「选线程、按顺序派发」；数据搬运、归约和与 peer 的握手在 RunWorkColl → Primitives 中。
//
// ---- (c) 例：blockDim.x=256，三份 Work 分别需要 4/8/8 个 warp ----
//   workStorage: [ Work[0]:4 warps ][ Work[1]:8 warps ][ Work[2]:8 warps ]
//
//                        Work[0]   __syncthreads   Work[1]      Work[2]
//   threads   0..127:     [ run  ] -------|-------- [ run  ] --- [ run  ]
//   threads 128..255:     [ skip ] -------|-------- [ run  ] --- [ run  ]
//
//   Work[0] -> Work[1]：参与线程从 128 变为 256，必须全 block 对齐。原因是 Primitives 内部用
//     named barrier（编号 15-group，计数 = 参与线程数）同步：若 128..255 抢先进入 Work[1]，
//     会以计数 256 到达与 Work[0] 同编号的 barrier，与仍在 Work[0] 中按计数 128 等待的线程混算；
//     同时还会覆写 Work[0] 仍在使用的 ncclShmem.groups[] 状态。
//   Work[1] -> Work[2]：参与线程集合不变，Primitives 内部同计数的 barrier 能自行保持同步，省掉一次
//     全 block barrier。
//
// 调用层次：ncclKernelMain -> RunWorkBatch::run -> RunWorkColl::run -> runRing/runTree 等算法实现。
// 本函数只处理当前批；返回后由 ncclKernelMain 沿 nextBatchIx 加载并执行下一批。
template <ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run() {
    int tid = threadIdx.x;
    int tn = blockDim.x; // 整个 block 的线程数；当前 Work 的参与线程数在后面计算为 subtn。

    // ============ 1. 按需准备归约算子的额外参数 ============
    // 只有带参数的 RedOp（如 PreMulSum 的乘数）才进入；编译期常量，普通 Sum/Max 等整段被裁掉。
    // 目的：把「参数在 device 内存中的地址」提前解引用成值，后续每次 reduce 直接用值，不再访存。
    if (RedOpArg<RedOp>::ArgUsed) {
      int nWorks = ncclShmem.nWorks;
      // 一个线程负责一份 Work（w = tid, tid+tn, ...）；与后面「多线程协作执行一份 Work」是两种分工。
      for (int w = tid; w < nWorks; w += tn) {
        struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem.workStorage + w * ncclShmem.workSize);
        if (work->redOpArgIsPtr) {
          // redOpArg 原地改写：地址 -> loadArg 读出的值（位模式）。只改 shared 中的副本，
          // 参数包/FIFO 里的原始 Work 不变；redOpArgIsPtr=0 时 redOpArg 本身就是立即值，直接沿用。
          work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
        }
      }
      __syncthreads(); // 写 redOpArg 的线程与后面执行该 Work 的线程不同，需全 block 可见后再执行。
    }

    // Block-uniform gate so a profiler-off launch keeps the baseline cost (no extra
    // sync/stamps). profilerEnabled(0) is uniform; nWorks>0 guards workStorage[0].
    // 以首份 Work 的 profilerEnabled 作为整批开关：所有线程读同一份 shared 数据，结果一致，
    // 因此末尾 if (profOn) 内的 __syncthreads 要么全 block 都执行、要么都不执行，不会死锁。
    bool profOn = (ncclShmem.nWorks > 0) && profilerEnabled(0);
    if (profOn && threadIdx.x == 0) profilerPhase(NCCL_KERNEL_PHASE_AFTER_OPEN);  // end of initial sync

    // ============ 2. 按 Work 顺序选择参与线程，执行当前 channel 对应的工作 ============
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int w = 0; w < ncclShmem.nWorks; w++) {
      // 按 ncclShmem.workSize 步进：ncclDevWorkCollReg 首成员就是 ncclDevWorkColl，两者都可按 Coll 指针读。
      // 所有线程（包括本轮不参与的）都走这段，保证下面的 __syncthreads 全 block 到齐。
      struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w * ncclShmem.workSize);
      if (w != 0) {
        struct ncclDevWorkColl* workPrev =
          (struct ncclDevWorkColl*)(ncclShmem.workStorage + (w - 1) * ncclShmem.workSize);
        // 参与线程数变化时全 block 对齐，防止 named barrier 计数混算和 ncclShmem.groups 被提前覆写，见 (c)。
        if (work->nWarps != workPrev->nWarps) __syncthreads();
      }
      int subtn = work->nWarps * WARP_SIZE; // 当前 Work 的参与线程数，按整 warp 划分，warp 内不分化。
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      // 进入 RunWorkColl 的模板特化（F12 落到上面的空主模板，真实实现在 all_reduce.h 等），
      // 如 AllReduce + RING + SIMPLE 进入 runRing。传入的 tn 是 subtn，不是 blockDim.x。
      // tid>=subtn 的线程跳过本份 Work，但仍继续循环，以便参与下一次切换时的 __syncthreads。
      if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work);
    }
    // ============ 3. 按需记录批次计算结束阶段 ============
    // End of compute. Sync so thread 0's stamp reflects the last worker finishing.
    if (profOn) {
      __syncthreads(); // 等最晚的执行线程到齐，再让线程 0 记录结束阶段时间戳。
      if (threadIdx.x == 0) profilerPhase(NCCL_KERNEL_PHASE_BEFORE_CLOSE);
    }
  }
};

__device__ __forceinline__ void profiler(int action) {
  if (threadIdx.x == 0) {
    int idx = 0;
    uint64_t wc = ncclShmem.channel.workCounter + 1;
    if (action == START) {
      // workStarted timestamp+counter share one 16B slot (single cache line), so no
      // fence is needed; the BEGIN phase stamp is ordered by STOP's fence below.
      for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        uint64_t ts = globaltimer();
        int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;
        ncclShmem.comm.workPhases[ncclShmem.channelId].data[slot].timestamps[NCCL_KERNEL_PHASE_BEGIN] = ts;
        ncclShmem.comm.workStarted[ncclShmem.channelId].data[slot].timestamp = ts;
        ncclShmem.comm.workStarted[ncclShmem.channelId].data[slot].counter = wc;
      }
    } else {
      bool fenceNeeded = false;
      for (; wc <= ncclShmem.channel.workCounter + ncclShmem.nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        uint64_t ts = globaltimer();
        int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;
        ncclShmem.comm.workCompleted[ncclShmem.channelId].data[slot].timestamp = ts;
        ncclShmem.comm.workPhases[ncclShmem.channelId].data[slot].timestamps[NCCL_KERNEL_PHASE_END] = ts;
        fenceNeeded = true;
      }
      // workPhases stamps span the kernel and straddle cache lines, so fence once
      // before publishing the counters to order all phase stamps ahead of them.
      if (fenceNeeded) {
        __threadfence_system();
        idx = 0;
        for (uint64_t wc2 = ncclShmem.channel.workCounter + 1; wc2 <= ncclShmem.channel.workCounter + ncclShmem.nWorks;
             wc2++) {
          if (!profilerEnabled(idx++)) continue;
          int slot = wc2 % MAX_PROFILER_EVENTS_PER_CHANNEL;
          ncclShmem.comm.workPhases[ncclShmem.channelId].data[slot].counter = wc2;
          ncclShmem.comm.workCompleted[ncclShmem.channelId].data[slot].counter = wc2;
        }
      }
      ncclShmem.channel.workCounter += ncclShmem.nWorks;
      if (action == FINI)
        ((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId].workCounter =
          ncclShmem.channel.workCounter;
    }
  }
}

// 每个 block 负责一个 channel：建立组内执行上下文，再沿该 channel 的 batch 链执行 Work。
// args 指向 GPU 已收到的参数包；ncclShmem 是本 block 的共享工作台，各 block 之间互不共用。
// 参数头/comm/channel 是上下文，workStorage 保存当前 Work 描述，用户数据仍通过 Work 中的指针访问。
//
// 例：channelMask={2,5}，finishPlan 已将 batch 排成 [A,D,B,C]（无扩展 batch 的示例）：
//   batch index:    0    1    2    3
//   batch:          A    D    B    C
//   channel:        2    5    2    2    <- 仅标注归属，batch 本身没有 channelId 字段
//   nextJump:       2    0    1    0
//
//   block 0 --> channel 2 --> load A --> run A --> load B --> run B --> load C --> run C
//   block 1 --> channel 5 --> load D --> run D --> finish
//   各 block 独立沿自己的链执行；这些箭头表示先后关系，不要求不同 block 同时被调度。
//
// 本 block 的工作台：
//   [args + channelId] -> [comm + channel] -> [workStorage + funcId + nWorks]
//                                                     |
//                                               execute batch
//                                                     |
//                        [nextBatchIx] <--------------+
//                              |
//                -1: finish    +-- otherwise: load next batch and repeat
template <int SpecializedFnId, typename SpecializedRunWorkBatch>
__device__ __forceinline__ void ncclKernelMain(struct ncclDevKernelArgs const* args) {
  int tid = threadIdx.x;
  int tn = blockDim.x;

  // ============ 1. 参数头入 shared memory，确定本 block 的真实 channelId ============
  // Copy kernel args to shmem and then only read those. Otherwise the compiler
  // will end up putting the args into thread local stack which is very wasteful.
  if (tid < sizeof(ncclDevKernelArgs) / sizeof(uint32_t)) {
    ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid]; // 每线程复制参数头的一个 4B 字；batch/Work 后面再读。
  }

  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  // PTX has the fns instruction which does this but is extremely slow. We can
  // do better when we know all threads are querying the same bitmask.
  if (tid < MAXCHANNELS && (args->channelMask & (1ull << tid))) {
    // tid 是候选 channelId；n 是它在活跃 channel 中的位置，即比 tid 小的置位 bit 数。
    // 例：mask={2,5}，tid=2 时 n=0，tid=5 时 n=1；各 block 只接受 n==blockIdx.x 的候选。
    int n = __popcll(args->channelMask & ((1ull << tid) - 1)); // 活跃 channel 的紧凑序号，从 0 开始。
    if (blockIdx.x == n) ncclShmem.channelId = tid; // 一个匹配线程填写 channelId，整个 block 随后共用它。
  }
  // 发布参数头和 channelId；这是 block 内 barrier，其他 block/rank 不参与。
  __syncthreads(); // publish ncclShmem.{args, channelId}
  /* set abort flag to 0 */
  if (tid == 0) {
    ncclShmem.aborted = 0;
    // 继承本 channel 的观测计数；profiler(FINI) 会把推进后的 workCounter 写回设备侧 channel。
    ncclShmem.channel.workCounter =
      ((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId].workCounter;
  }

  // ============ 2. 各 warp 分工加载上下文和首 batch 的 Work ============
  // 首轮分工（WARP_SIZE=32；各项并行加载，最后统一 barrier）：
  //   threads  0..31  / warp 0  : device comm       -> ncclShmem.comm
  //   threads 32..63  / warp 1  : device channel[c] -> ncclShmem.channel
  //   threads 64..end / warp 2+ : Args/workBuf Work -> ncclShmem.workStorage
  //                                           |
  //                                    __syncthreads()
  //                                           |
  //                               所有线程读取同一份执行上下文
  // Use first 2 warps to load comm and channel, and remaining load work batch.
  switch (tid / WARP_SIZE) {
  case 0: // warp 0 负责加载 communicator 信息到 ncclShmem.comm
    {
      void* dst = &ncclShmem.comm;
      void* src = ncclShmem.args.comm;
      int bytes = sizeof(ncclKernelComm);
      static_assert(sizeof(ncclKernelComm) <= 16 * WARP_SIZE,
                    "ncclKernelComm cannot be loaded by a single warp in one insn.");
      // 每个参与线程复制 16B；整个结构的大小保证一个 warp 足够覆盖。
      copyToShmem16(tid, dst, src, bytes);
    }
    break;
  case 1: // warp 1 负责加载本 channel 信息到 ncclShmem.channel
    { // Get address of channel without incurring indirect load from ncclKernelComm::channels
      void* dst = &ncclShmem.channel;
      void* src = &((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];
      int bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16 * WARP_SIZE,
                    "ncclDevChannel cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid - WARP_SIZE, dst, src, bytes);
    }
    break;
  default:
    { // 其他 warp 负责加载首 batch 引用的 Work 到 ncclShmem.workStorage
      // 为加载子组重新从 0 编号；helper 至少需要 64 个线程。
      int subtid = tid - 2 * WARP_SIZE;
      int subtn = tn - 2 * WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      // 首 batch 下标直接取 blockIdx.x，这是 finishPlan 数组前缀布局的消费端。
      // helper 加载 Work 并填写 funcId/nWorks/nextBatchIx；nextExtends 可将扩展描述融合加载。
      loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
    }
    break;
  }
  __syncthreads(); // 等 comm/channel/Work 都加载完成，再开始执行。

  // ============ 3. 执行当前批次，再切换下一批；直到链尾或中止 ============
  //   run current batch
  //          |
  //   nextBatchIx == -1 ? --yes--> finish
  //          |
  //          no
  //          v
  //   barrier: 旧批线程到齐 -> load next Work -> barrier: 新 Work 可读 -> 下一轮
  while (ncclShmem.aborted == 0) { // 中止状态由 primitive 的检查逻辑更新。
    profiler(START);
    // 在当前 kernel 内分派 batch 执行函数；专用编号匹配时直接调用，否则通过 device 函数表调用。
    if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId) {
      SpecializedRunWorkBatch().run(); // 执行可内联的专用 batch 实现。
    } else {
      ncclDevFuncTable[ncclShmem.funcId](); // 按本 batch 的 funcId 调用 device 执行函数。
    }

    if (ncclShmem.nextBatchIx == -1) break; // 没有下一个 batch 了，退出循环
    int batchIx = ncclShmem.nextBatchIx; // helper 根据 nextJump 算出的同 channel 后继下标。
    __syncthreads(); // 等旧批的执行线程到齐，之后才可复用 workStorage。
    profiler(STOP);
    // 后续批次由整个 block 协作加载，comm/channel 沿用；覆盖当前 Work 描述和批次控制字段。
    loadWorkBatchToShmem(tid, tn, args, batchIx);
    __syncthreads(); // 新 Work 和 funcId/nextBatchIx 等已就绪，下一轮才能读取。
  }
  profiler(FINI); // 收尾当前批次的观测记录，并写回本 channel 的 workCounter。
}

__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K);
__device__ void ncclDevFunc_Nop();

#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args); \
  }

#define DEFINE_ncclDevKernel_nop(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {}

#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWorkBatch<coll, ty, redop<ty>, algo, proto>().run(); \
  }

#endif
