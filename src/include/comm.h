/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COMM_H_
#define NCCL_COMM_H_

// #include "transport.h"
#include "p2p.h"
#include "collectives.h"
#include "nccl_tuner.h"
#include "proxy.h"
#include "strongstream.h"
#include "nccl_net.h"
#include "register.h"
#include "graph.h"
#include "profiler.h"
#include "allocator.h"
#include "dev_runtime.h"
#include "sym_kernels.h"
#include "ce_coll.h"
#include "rma/rma.h"
#include "argcheck.h"
#include "mem_manager.h"
#include "tuning.h"
#include "enqueue/raw_task.h"
#include "enqueue/task_pretuning.h"
#include "enqueue/task_classify.h"
#include "enqueue/task_posttuning.h"
#include "enqueue/mgmt_task_enq.h"

#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#endif

#if CUDART_VERSION < 9000
struct cudaLaunchParams {
  void* func;
  dim3 gridDim;
  dim3 blockDim;
  void** args;
  size_t sharedMem;
  cudaStream_t stream;
};
#endif

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define CUDA_IPC_MIN 2097152UL

// Channels / LL tuning
#define NCCL_LL_THREAD_THRESHOLD 8
#define NCCL_LL128_THREAD_THRESHOLD 8
#define NCCL_SIMPLE_THREAD_THRESHOLD 64

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      void* ptrExchange;
      uint64_t redOpArgExchange[2];
      char pad2[CACHE_LINE_SIZE - sizeof(void*) - 2 * sizeof(uint64_t)];
      int offsFifo[NCCL_STEPS];
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      struct ncclConnFifo connFifo[NCCL_STEPS];
      int flush; // For GDRCopy-based flush
    };
    char pad4[MEM_ALIGN];
  };
};

enum helperThreadState {
  ThreadStart,
  ThreadStop
};

#define NCCL_IPC_POOL_SIZE (2 * NCCL_MAX_LOCAL_RANKS * NCCL_MAX_OPS)

struct ncclUserRedOp {
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  ncclDataType_t datatype;
  ncclDevRedOpFull opFull;
};

struct ncclNodeRanks {
  int localRanks;
  int* localRankToRank;
};

struct cliqueInfo {
  int id;
  int size;
  int* ranks;
};

struct ncclDestructor {
  struct ncclDestructor* next;
  void* obj;
  struct ncclComm* comm;
  ncclResult_t (*fn)(struct ncclDestructor* me);
};

struct ncclCommCallback {
  struct ncclCommCallback* next;
  ncclResult_t (*fn)(struct ncclComm* comm, struct ncclCommCallback* cb);
};
struct ncclCommEventCallback {
  struct ncclCommEventCallback* next;
  cudaEvent_t event;
  ncclResult_t (*fn)(struct ncclComm* comm, struct ncclCommEventCallback* cb);
};

struct ncclSharedResources {
  int refCount;
  struct ncclComm* owner; /* comm which creates this shared res. */
  struct ncclChannelPeer* peers[MAXCHANNELS];
  struct ncclDevChannelPeer* devPeers[MAXCHANNELS];
  /* P2P operation counter, one per channel */
  uint64_t p2pOpCount[MAXCHANNELS];
  /* Collective operation counter */
  uint64_t collOpCount;
  int tpNRanks;
  int tpNLocalRanks;
  int tpNChannels;
  int tpP2pNChannels;
  int tpP2pChunkSize;
  uint64_t magic;

  // top parent rank to localRank translation table
  int* tpRankToLocalRank;
  // Internal streams
  struct ncclStrongStream deviceStream, hostStream;
  int persistentRefs;
  cudaEvent_t launchEvent, scratchEvent;

  /* proxy related shared res */
  struct ncclProxyState* proxyState;

  // GIN state
  struct ncclGinState ginState;
};

struct ncclChannel {
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;
  struct ncclRing ring;
  int* devRingUserRanks;
  struct ncclTree tree;

  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;

  struct ncclNvls nvls;

  int id; // index of this channel
  uint32_t workFifoProduced; // +1 successor of last used work fifo byte

  /* comm split sharable resources */
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};

struct ncclWorkBatchList {
  struct ncclWorkBatchList* next;
  struct ncclDevWorkBatch batch;
};
struct alignas(16) ncclWorkList {
  struct ncclWorkList* next;
  enum ncclDevWorkType workType;
  int size; // Size of struct following this node
  // ncclDevWorkColl, ncclDevWorkColLReg, ncclDevWorkP2p[]...
};

struct ncclCollnetHandleList {
  struct ncclCollnetHandleList* next;
  void* collnetHandle;
  size_t size;
  const void* buffer;
  struct ncclProxyConnector* proxyconn;
};

struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  ncclDataType_t datatype;
  ncclRedOp_t opHost;
  struct ncclDevRedOpFull opDev;
  int chunkSteps, sliceSteps;
  // Computed later:
  size_t trafficBytes;
  int32_t nMaxChannels:8;
  int32_t nWarps:8;
  int32_t algorithm:8, protocol:8;
  uint32_t isCollnet:1, isNvls:1, isSymLast:1;
  uint32_t devFuncId:29;
  int regBufType;
  // number of elements in planner->ipcMemQueue associated with this collective
  int nCleanupQueueElts;

  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  ncclSymRegType_t winRegType;
  void* sendMhandle;
  void* recvMhandle;
  void** sendNetHandles;
  void** recvNetHandles;
  void** srecvNetHandles;
  // index for IPC record lookup
  uintptr_t sendbuffOffset;
  uintptr_t recvbuffOffset;
  uintptr_t* sendbuffRmtAddrs;
  uintptr_t* recvbuffRmtAddrs;

  // Profiler plugin
  int eActivationMask;
  void* groupApiEventHandle;
  void* collApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
  // Inclusive channel range this task ran on; mirrors devWork->channelLo/Hi.
  uint8_t channelLo;
  uint8_t channelHi;

  // Per-collective config related members
  bool aggIsolate; // Whether task needs to be isolated from aggregation because of config options.
                   // Config options related to resource and algorithm usually need the isolation.
  int minCTAs;
  int maxCTAs;
  int nvlsCTAs;
  int cgaClusterSize;
  // Resolved algorithm selection, captured at append time for the same reason.
  uint64_t algMask;    // set bit == algorithm allowed by the filter; 0 == automatic (no filter)
  int forceAlgSelection; // 1 (default) == error on unsatisfiable selection; 0 == fall back to automatic
  int CTAPolicy;  // resolved effective CTAPolicy for this task
  // Per-call profiler annotation (0 == untagged), resolved at task-append time and
  // delivered verbatim to profiler plugins.
  uint64_t profilerTag;
};

struct ncclTaskBcast {
  struct ncclTaskBcast* next;
  ncclFunc_t func;
  void* recvbuff;
  const void* sendbuff;
  size_t count;
  ncclDataType_t datatype;
  int root;
  int ringDepth;

  // what computes after...
  int32_t algorithm:8, protocol:8;

  // Profiler plugin
  int eActivationMask;
  void* groupApiEventHandle;
  void* collApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
  uint64_t profilerTag; // Per-call profiler annotation (0 == untagged)
};

struct ncclTaskP2p {
  struct ncclTaskP2p* next;
  ncclFunc_t func;
  ncclFunc_t collAPI;
  void* buff;
  size_t count;
  ncclDataType_t datatype;
  int root;
  size_t bytes;
  bool allowUB;

  // Profiler plugin
  int eActivationMask;
  void* groupApiEventHandle;
  void* p2pApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
  uint64_t profilerTag; // Per-call profiler annotation (0 == untagged)
  // Per-direction channels used by this task; read by the profiler to emit and
  // advertise KernelCh per direction (StartTaskEvents / PostPlanWork).
  uint64_t channelMask;
  // Shared by both tasks of an addP2pToPlan() pair; 0 = unassigned.
  uint16_t p2pPairId;
};

struct ncclTaskRma {
  struct ncclTaskRma* next;
  ncclFunc_t func;
  int ctx;
  size_t count;
  ncclDataType_t datatype;
  size_t bytes;

  void const* srcBuff;
  size_t srcWinOffset;
  struct ncclDevrWindow* srcWinHost;

  int peer;
  size_t peerWinOffset;
  struct ncclDevrWindow* peerWinHost;

  // Signal operations
  ncclSignalMode_t signalMode;
  int signalIdx;

  // WaitSignal operations
  int* peers;
  int* nsignals;
  int* signalIdxs;
  int npeers;

  // Profiler plugin
  int eActivationMask;
  void* groupApiEventHandle;
  void* rmaApiEventHandle;
  void* eventHandle;
  uint8_t nChannels;
};

struct ncclKernelPlan {
  // A kernel plan is also a callback that reclaims itself. Hence this must
  // be the first member.
  struct ncclCommCallback reclaimer;

  struct ncclComm* comm;
  struct ncclKernelPlan* next;

  bool persistent; // aka captured in a graph
  bool isHostCbEnq;
  bool isSymColl;
  bool isCeColl;
  bool isRma;
  enum ncclDevWorkStorageType workStorageType;
  bool kernelSpecialized;
  int kernelDynSmem; // only for symmetric kernels
  void* kernelFn;
  union {
    struct ncclDevKernelArgs* kernelArgs;
    void* kernelSymArgs;
    struct ncclCeCollArgs* ceCollArgs;
    struct ncclRmaArgs* rmaArgs;
  };
  size_t kernelArgsSize;
  uint64_t channelMask; // bitset of which channels are present
  bool hasProxyOps; // does any channel have a non-empty proxyOpQueue
  // Any task with ncclProfileKernelCh; gates the captured host callback.
  bool hasProfilerOps;
  // Source of ncclTaskP2p::p2pPairId; incremented per addP2pToPlan() call.
  uint16_t p2pPairCounter;
  int threadPerBlock;
  int cgaClusterSize;  // per-launch CGA cluster size; defaults to comm->config.cgaClusterSize

  int collOpCount; // Number of collectives in this plan.
  int nWorkBatches; // Number of work batches.
  int nTasksBcast; // Number of bcast tasks in this plan.
  size_t workBytes; // Sum size of all work (in the fifo) in bytes.
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> workQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> cleanupQueue;
  void* workBufPersistent;

  struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> p2pTaskQueue;
  struct ncclIntruQueue<struct ncclTaskBcast, &ncclTaskBcast::next> bcastTaskQueue;
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next> rmaTaskQueueProxy;
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next> rmaTaskQueueCe;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;

  // Profiler plugin
  void* groupApiEventHandle;
  void* kernelLaunchEventHandle;
  void* groupEventHandle;
};

////////////////////////////////////////////////////////////////////////////////
// Roughly sorts ncclTaskColl's by their size descending. This structure is
// self-referential, meaning that pointers it contains internally may point
// into the structure itself. This means that it is NOT memcpy-moveable:

struct ncclTaskCollSorter {
  static constexpr int UnitLog2 = 10; // 1K
  static constexpr size_t UnitSize = 1 << UnitLog2;
  static constexpr int MaxLog2 = 30; // 1GB
  static constexpr size_t MaxSize = 1ull << MaxLog2;
  // Number of bins between powers of 2. For 4 bins, the worst case out-of-order
  // relative magnitude is (5/4)-1 = 25%
  static constexpr int BitsPerPow2 = 2;
  static constexpr int BinsPerPow2 = 1 << BitsPerPow2;
  static constexpr int BinCount = 1 + (MaxLog2 - UnitLog2) * BinsPerPow2;

  struct ncclTaskColl* head;
  struct ncclTaskColl* tail;
  // Least bin such that it and all above are empty.
  int binEdge;
  // Pointer to the pointer to this bin's head node which is either the
  // previous node's `next` field or `head`.
  struct ncclTaskColl** bins[BinCount];
};

inline void ncclTaskCollSorterInsert(struct ncclTaskCollSorter* me, struct ncclTaskColl* x, size_t size) {
  constexpr int UnitLog2 = ncclTaskCollSorter::UnitLog2;
  constexpr size_t MaxSize = ncclTaskCollSorter::MaxSize;
  constexpr int BitsPerPow2 = ncclTaskCollSorter::BitsPerPow2;
  constexpr int BinCount = ncclTaskCollSorter::BinCount;
  // Value is bounded by MaxSize>>UnitLog2 which fits in uint32_t
  int bin = u32fpEncode(static_cast<uint32_t>(std::min(MaxSize, size) >> UnitLog2), BitsPerPow2);
  bin = BinCount - 1 - bin; // descending bin

  if (me->bins[bin] == nullptr) {
    if (me->binEdge <= bin) {
      me->binEdge = bin + 1;
      me->bins[bin] = me->tail ? &me->tail->next : &me->head;
      me->tail = x;
    } else {
      // Find successor non-empty bin after this one.
      int succ = bin + 1;
      while (me->bins[succ] == nullptr) succ++;
      // What was our successor's head's previous is now our head's previous.
      me->bins[bin] = me->bins[succ];
      // The first node we insert is our tail, so that becomes our successor's
      // head's new previous.
      me->bins[succ] = &x->next;
    }
  }
  // Push a new head for this bin.
  x->next = *me->bins[bin];
  *me->bins[bin] = x;
}

inline bool ncclTaskCollSorterEmpty(struct ncclTaskCollSorter* me) {
  return me->head == nullptr;
}

// Reset sorter and return sorted linked list of its coll tasks.
inline struct ncclTaskColl* ncclTaskCollSorterDequeueAll(struct ncclTaskCollSorter* me) {
  struct ncclTaskColl* head = me->head;
  if (head != nullptr) memset(me, 0, sizeof(*me));
  return head;
}

////////////////////////////////////////////////////////////////////////////////

struct ncclCudaStreamList {
  struct ncclCudaStreamList* next;
  cudaStream_t stream;
};

struct ncclKernelPlanner {
  //////////////////////////////////////////////////////////////////////////////
  // State for accumulating tasks between ncclGroupStart/End()
  //////////////////////////////////////////////////////////////////////////////

  struct Peer {
    bool sendSeen, recvSeen;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
    struct ncclIntruQueue<struct ncclTaskBcast, &ncclTaskBcast::next> bcastQueue;
  };
  struct ncclTaskCollSorter collSorter;
  struct Peer* peers /*[nRanks]*/;
  int nTasksColl, nTasksP2p, nTasksBcast, nTasksRma;
  int nTasksP2pSend, nTasksP2pRecv;

  struct {
    int minBcastPeer;  /* initialized to INT_MAX */
    int maxBcastPeer;  /* initialized to INT_MIN */
    int BcastPeers;  /* initialized to 0 */
  } bcast_info;

  bool persistent;
  // The list of user streams aggregated over all tasks present.
  struct ncclCudaStreamList* streams;
  // The most recent user stream. Ignored if streams==nullptr
  cudaStream_t streamRecent;
  // The graph capturing all user streams or invalid if none. Thus we restrict the
  // user that all streams must be captured in the same graph or not captured
  // at all. Technically we could probably relax this, but that would mean
  // collecting a different `ncclTasks` per graph and one for non-graph.
  struct ncclCudaGraph capturingGraph;

  //////////////////////////////////////////////////////////////////////////////
  // Lists of tasks to be assembled into plans.
  //////////////////////////////////////////////////////////////////////////////

  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collTaskQueue;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collCeTaskQueue;
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* rmaTaskQueues; // Per-context queue for RMA tasks
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> collSymTaskQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> collWorkQueue;
  struct ncclIntruQueue<struct ncclWorkList, &ncclWorkList::next> tmpCollWorkQueue;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> collCleanupQueue;

  //////////////////////////////////////////////////////////////////////////////
  // State for building current (Work-In-Progress) plan:
  //////////////////////////////////////////////////////////////////////////////

  struct WipPlan {
    struct Channel {
      struct {
        int workBytes; // Sum size of work metadata referenced by this batch.
        int nP2ps; // Number of p2p works in this batch
        int nBcasts; // Number of bcast works in this batch
        int p2pEpoch;
        int p2pRounds[NCCL_MAX_DEV_WORK_P2P_PER_BATCH]; // which rounds are present in this batch.
      } wipBatch; // work-in-progress batch which will be next tail of workBatchQueue
      int nWorkBatchesP2p; // number of p2p batches for this channel.
      int nWorkBatchesBcast; // number of bcast batches for this channel.
      struct ncclIntruQueue<struct ncclWorkBatchList, &ncclWorkBatchList::next> workBatchQueue;
      struct ncclIntruQueue<struct ncclProxyOp, &ncclProxyOp::enqNext> proxyOpQueue;
    } channels[MAXCHANNELS];
  } wipPlan;

  //////////////////////////////////////////////////////////////////////////////
  // State for launching built plans:
  //////////////////////////////////////////////////////////////////////////////

  // List of kernel plans built form tasks.
  struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
  // First of the unlaunched kernels in `planQueue`
  struct ncclKernelPlan* unlaunchedPlansHead;
};

#define NCCL_MAGIC 0x0280028002800280 // Nickel atomic number is 28.

typedef enum ncclGroupTaskType {
  ncclGroupTaskTypeCollective = 0,
  ncclGroupTaskTypeSymRegister = 1,
  ncclGroupTaskTypeRawTask = 2,
  ncclGroupTaskTypeMgmtTask = 3,
  ncclGroupTaskTypeNum = 4,
} ncclGroupTaskType_t;

struct ncclCommSymTeams;

// NCCL_CHECK_MODE=DEBUG_LOCAL/DEBUG_GLOBAL
// ncclCheckModeDebugLocal : check the input args/pointers locally, it replaces ncclParamCheckPointers()
// ncclCheckModeDebugGlobal : check the input args globally such as symmetric buffer check, etc.
typedef enum ncclCheckMode {
  ncclCheckModeDefault = 0,
  ncclCheckModeDebugLocal = 1,
  ncclCheckModeDebugGlobal = 2,
} ncclCheckMode_t;

// 通信域的中枢对象：一个 rank 的全部 host 侧状态。生命周期 = ncclCommInitRank .. ncclCommDestroy。
// 阅读提示：字段大致按"init 阶段填充顺序"排列，下面按功能域分块标注。
// 注意：布局对 devComm 拷贝、cache line padding（intraPad*）和首尾 magic 校验敏感，不要随意重排字段。
struct ncclComm {
  // ============ 对象自检与内存生命周期 ============
  uint64_t startMagic; // 与 endMagic 成对，CommCheck() 每次 API 入口校验，用于检测 comm 内存被踩/野指针
  // memPermanent: comm 全生命周期 arena（channel、topo、各 memPool 的 backing），commFree 时整体析构
  // memScoped:    ncclGroupStart/End 作用域内的临时 arena，GroupEnd 时 Pop 回滚 bump 指针
  struct ncclMemoryStack memPermanent, memScoped;
  // List of destructors to run when comm is destructed
  // 头插的 LIFO 延迟清理链（ncclCommPushFree / PushCudaFree 等），commFree 时逆序执行
  struct ncclDestructor* destructorHead;

  // ============ 资源共享与父子 comm（split / grow / shrink）============
  struct ncclCudaContext* context;      // per-CUDA-context 的 launchOrder 状态，按 commHash 引用计数
  struct ncclSharedResources* sharedRes; // 可跨 comm 复用：peers/devPeers、内部 stream、proxyState、ginState
  /* map to top parent ranks. */
  // 本 comm 的 rank -> 最初创建 sharedRes 的那个 comm 的 rank；非共享时是恒等映射。
  // 用它索引 sharedRes->peers[] 以及 proxy/net 侧的 tpRank
  int* topParentRanks;
  int* topParentLocalRanks;

  // ============ 拓扑、对端信息与本地网络能力 ============
  struct ncclChannel channels[MAXCHANNELS]; // 每个 channel 一套 ring/tree/collnet/nvls 的 peer 连接
  struct ncclPeerInfo* peerInfo;            // 长度 nRanks，AllGather1 换来的全体能力名片（见 transport.h）
  struct ncclTopoSystem* topo;              // 本节点探测出的 PCIe/NVLink/NIC 拓扑图
  // 长度 nRanks 的 P2P proxy connector 缓存（懒初始化），跨进程交换 cuMem handle / UDS fd
  struct ncclProxyConnector* gproxyConn;
  // legacy IPC 注册的延迟清理队列，comm 销毁时同步 drain
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next> legacyRegCleanupQueue;
  bool peerInfoValid;
  int minNetCount; // Minimum number of network devices local to a rank
  float minLocalNetBw; // Minimum total network bandwidth local to a GPU
  float minNetBw; // Minimum bw of any network device local to a rank
  // 上面三个都是 AllGather3 后的全局 min，用来限制 registered buffer 能铺开的 channel 数

  // ============ 网络插件（NET / CollNet / RMA 的可插拔实现）============
  ncclNet_t* ncclNet;    // 选中的网络插件 vtable：isend/irecv/regMr...（内置 IB 或外部 plugin）
  void* netContext;
  void* rmaContext;
  int netPluginIndex;
  int rmaPluginIndex;
  int ncclNetVer;
  ncclNetDeviceType netDeviceType; // 是否支持 device-initiated 网络（GIN/GDAKI）
  ncclCollNet_t* ncclCollNet;      // in-network reduction（SHARP）插件
  void* collNetContext;

  // ============ 建连控制面 ============
  void* bootstrap; // 带外 TCP 通道，负责 AllGather peerInfo 与交换 ncclConnect 信封
  bool isGrow; // true if this comm is created via ncclCommGrow
  // Bitmasks for ncclTransportP2pSetup
  // connectSend[peer] / connectRecv[peer] 的 bit i 表示"与该 peer 的第 i 个 channel 待建连"；
  // ncclTransportP2pSetup 消费后清零
  uint64_t* connectSend;
  uint64_t* connectRecv;
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS]; // 各算法的拓扑搜索结果（带宽/延迟/pattern）
  int maxTreePattern;                               // 全局 max，tree tuning 时据此折减带宽估计
  bool initAlgoChannels[NCCL_NUM_ALGORITHMS];       // runtimeConn 下各算法是否已完成懒建连
  bool runtimeConn; // if dynamic connection is supported
                    // 开启后 init 只做 setupChannel，ring/tree 建连推迟到首次 collective
  bool directMode; // if any process manages more than one local rank
  int cuMemSupport;

  // ============ 身份标识与本地硬件属性 ============
  uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect
                  // mismatches.
                  // bootstrap/proxy socket 的握手码，防止不同 comm 的连接串线；split/grow 由父 magic 派生

  uint64_t commHash; // comm 的全局标识（uniqueId 哈希），用于 NVTX/RAS/tuner，并混入 peerInfo 的 host/pidHash
  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int nvmlDev; // my nvml device index
  int compCap; // compute capability of the GPU
  int minCompCap, maxCompCap; // min/max compute capability in the communicator
  int64_t busId;   // my PCI bus ID in int format
  ncclAffinity cpuAffinity; // CPU affinity of the GPU
                            // init/group 的异步线程会临时绑到这组 CPU 上，保证内存分配落在近端 NUMA
  int cudaArch; // matches __CUDA_ARCH__ of device

  int cpuArch;   // architecture - As defined in src/include/graph.h, e.g. x86/arm/ppc/mixed
  int cpuVendor; // vendor - As defined in src/include/graph.h
                 // 两者经 AllGather3 归一：集群异构时统一置为 MIXED

  // ============ rank 空间映射：全局 rank <-> 节点内 rank ============
  int node;
  int nNodes;
  int localRank;
  int localRanks;
  int maxLocalRanks; // 全局最大每节点 rank 数；== 1 时即 isOneRPN
  int minLocalRanks;
  int* rankToNode;
  int* rankToLocalRank;
  int* localRankToRank;
  // localRanks and localRanktoRank for all nodes
  struct ncclNodeRanks* nodeRanks;
  // MNNVL: Multi-Node NVLink
  // 跨节点 NVLink：同一 clusterUuid+cliqueId 的 GPU 可像同机一样走 P2P
  int MNNVL; // true when MNNVL is available
  struct cliqueInfo clique; // Our MNNVL clique information
  int cliqueRank; // Our rank within the MNNVL clique
  int contiguousRanksPerHost; // Number contiguous ranks per host. INT_MAX if non-uniform.

  // NVL Domain info
  ncclNvlDomainInfo_v5_t nvlDomainInfo; // domain 数与 min/max ranks per domain，传给 tuner plugin

  // ============ 运行模式开关 ============
  ncclCheckMode_t checkMode; // NCCL_CHECK_MODE/CHECK_POINTERS：参数校验强度，DEBUG_GLOBAL 时启用跨 rank 校验
  bool dmaBufSupport;
  bool ccEnable; // Confidential Computing 模式，开启时禁用 work FIFO

  // ============ 操作计数器 ============
  // Counter for tracking CUDA launches (P2P and collectives included)
  uint64_t opCount;
  // Collective operation counter
  uint64_t collOpCount; // 与 sharedRes->collOpCount 合成 proxy op id，用于 host/proxy 对齐同一次操作

  // ============ Channel 数量与 buffer 尺寸配置 ============
  // Channels for collectives
  int nChannels; // connection nChannels
                 // 建连用的 channel 数（ring/tree 图取 min 后跨 rank 对齐），enqueue 也以它为上限
  int collChannels; // enqueue nChannels
                    // 现状：仅在 connect 阶段赋值（split-share 时被 parent 截断），暂无读取方
  int nvlsChannels; // enqueue nChannels
                    // NVLS 专用上限，由 ncclNvlsTuning 独立算出，与 nChannels 解耦
  int nvlsTreeMaxChunkSize;

  // all nvls heads stored to check if we can splitShare
  int nvlsHeads[MAXCHANNELS];
  // Channels (per peer) for p2p
  int p2pnChannels;        // P2P 专用 channel 池大小（>= nChannels）
  int p2pnChannelsPerPeer; // 单个 peer 最多铺开几条并行 channel
  int p2pSchedGroupSize;   // 多节点 P2P 的分组大小，决定每轮的 channel base
  int p2pMaxPeers;         // 参与调度的并发 peer 上限

  // Should this comm allocate LL buffers for network P2P connections?
  bool allocP2pNetLLBuffers;

  // Buffer sizes
  int buffSizes[NCCL_NUM_PROTOCOLS]; // LL/LL128/SIMPLE 各自的连接 buffer 大小；stepSize = buffSizes[p]/NCCL_STEPS
  int p2pChunkSize;
  int nvlsChunkSize;

  // Cross-clique P2P: when true, use global rank for IPC buffer indexing
  bool p2pCrossClique;
  // NVL Domain size: number of ranks in the same NVLink domain (same clusterUuid)
  int nvlDomainSize;

  // Tuner values
  struct ncclTuningContext_t tuningContext;

  // ============ 异步错误与 abort ============
  /* This attribute can indicate the states of communicators and return code of
   * asynchronous NCCL operations. */
  ncclResult_t asyncResult; // 非阻塞模式下由 ncclCommGetAsyncError 读出

  // Flag to ask NCCL kernels to abort
  // abortFlag 是 host 可见副本，abortFlagDev 拷进 devComm 供 kernel 轮询，两者由 setCommAbortFlags 同写
  uint32_t* abortFlag;
  uint32_t* abortFlagDev;
  int* abortFlagRefCount; // split 共享 abort flag 时的引用计数
  // 不共享 abort flag 时，父 comm 用这两个指针指向子 comm 的 flag，实现级联 abort
  uint32_t* childAbortFlag;
  uint32_t* childAbortFlagDev;
  uint32_t destroyFlag; // ncclCommDestroy/Abort 入口置 1，挡住重复销毁
  uint32_t revokedFlag; // ncclCommRevoke 置 1，之后不允许再 split 共享资源

  // ============ Host -> Device：devComm 镜像与 work FIFO ============
  // Device side of the communicator (for cudaFree's)
  struct ncclKernelComm* devComm; // actually = &ncclKernelCommAndChannels::comm

  uint32_t workArgsBytes; // max size of kernel args
                          // work 小到能塞进 kernel 参数时就不走 FIFO（ncclDevWorkStorageTypeArgs）
  uint32_t workFifoBytes; // size of workFifoBuf, power of 2
  // 环形 buffer：host enqueue 侧写入 ncclDevWork*（生产者），kernel 侧读取（消费者）；
  // 走 GDR 映射或 pinned host 内存，Dev 后缀是 device 侧地址
  void* workFifoBuf;
  void* workFifoBufDev;
  void* workFifoBufGdrHandle;

  // Monotonic number of bytes (mod 1<<32) sent to fifo.
  // Produced 由 host 推进；Consumed 靠 kernel 完成事件回调补齐，二者之差即在途量，用于背压
  uint32_t workFifoProduced;
  uint32_t workFifoProducedLastRecorded;
  uint32_t workFifoConsumed;

  // ============ 进程内多 comm 同步（sense-reversing barrier）============
  // 同一进程管多张卡时，这些 comm 串成链表并用下面的 barrier 对齐 kernel launch。
  // intraPad1/2 把 phase / counter / gate 隔到不同 cache line，避免 counter 的原子写和 gate 的自旋读互相打架
  // Intra-process sync
  struct ncclComm* intraComm0; // leader of intra-process comms (self possible)
  struct ncclComm* intraNext; // next of intra-process comms, intraComm0 is head
  int intraRank;
  int intraRanks;
  uint32_t intraBarrierPhase; // 本地 sense 位，每过一次 barrier 翻转
  char intraPad1[64 - sizeof(uint64_t)];
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  char intraPad2[64 - sizeof(uint64_t)];
  uint64_t intraBarrierGate; // only used if this is intraComm0

  // ============ Proxy ============
  struct ncclProxyState* proxyState; // proxy 线程及其连接/进度状态，split 时可与父共享
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
                        // 原子减之后的旧值，为 0 才真正停线程

  // ============ CollNet（in-network reduction）============
  // Whether this communicator uses collNet
  bool isOneRPN; // maxLocalRanks == 1，即每节点仅一张卡
  uint8_t collNetSupportMatrix[4 /*sum,prod,max,min*/][ncclNumTypes]; // 各 (redop,dtype) 是否所有 head 都支持
  int* collNetHeads;    // 各节点的 CollNet head rank 列表
  int collNetHeadsNum;
  int collNetChainSupport;
  // Node-major rank order with transport heads first on each node.
  // dense rank -> user rank 的逆映射，PAT/rail 连接与 device 侧寻址都依赖它
  int* denseToUserRank;
  /* sharable collNet proxy progress resource. */
  struct ncclCollNetSharedRes* collNetSharedRes;

  // NVLink SHARP (NVLS) support
  int nvlsSupport;    // 硬件 multicast + 拓扑检查的结果
  int nvlsRegSupport; // 能否注册用户 buffer 到 NVLS（同进程多 rank 或 MNNVL 会被关掉）
  /* sharable NVLS resource. */
  struct ncclNvlsSharedRes* nvlsResources;

  // ============ 任务对象池（backing 全部来自 memPermanent）============
  // 大部分池在 reclaimPlan() 里随 kernel plan 一起归还，Rma 池例外（RMA 执行完就地归还）
  // pools backed by comm->memPermanent
  struct ncclMemoryPool memPool_ncclTaskBcast;
  struct ncclMemoryPool memPool_ncclTaskColl;
  struct ncclMemoryPool memPool_ncclTaskP2p;
  struct ncclMemoryPool memPool_ncclTaskRma;
  struct ncclMemoryPool memPool_ncclRawTask;
  struct ncclMemoryPool memPool_ncclProxyOp;
  struct ncclMemoryPool memPool_ncclKernelPlan;

  // ============ Group 聚合与 enqueue 流水线 ============
  // 一次 groupEnd 的处理链：rawTaskQueue -> (pretuning/tuning/classify) -> classifiedTaskQueues
  //                        -> posttuning -> planner -> kernel launch + proxy 上传
  // Next comm in this thread's active ncclGroup[Start|End](). Holds "0x1" when
  // this comm is not yet in a group.
  struct ncclComm* groupNext[ncclGroupTaskTypeNum];
  // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
  struct ncclComm* preconnectNext;
  int localPersistentRefs; // number of persistent plan-lists capturing this comm
                           // 被 CUDA Graph 捕获的 plan 数，destroy 时要自旋等它归零
  // 长度 nRanks 的轮次表：第 r 轮和谁发、和谁收，决定 P2P 的 channel 映射顺序
  struct P2pSchedulePair {
    int sendRank;
    int recvRank;
  }* p2pSchedule;

  struct ncclKernelPlanner planner;                     // task 累积 -> plan 构建 -> 待发射 plan 队列
  struct ncclRawTaskQueue rawTaskQueue;                 // API 刚入队、尚未调优的原始 task
  struct ncclClassifiedTaskQueues classifiedTaskQueues; // 调优后按执行路径分流（sym/legacy/p2p/rma/ce...）
  // Queue of management tasks (comm init/destroy/finalize/etc.) enqueued for this
  // comm during a ncclGroup[Start|End]() scope.
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> mgmtTaskQueue;
  bool simulationMode; // 目前无任何读写点，预留字段
  void* ringTasks; // An array of nRanks pointers used in ring sorting rooted collectives (bcast)

  // ============ 异步回收与回调 ============
  cudaMemPool_t memPool; // 本 GPU 的 cudaMallocAsync 池，供 persistent/graph 模式分配 device work buffer
  // Queue of events and associated callbacks for cleaning up asynchronous work.
  // Using this is preferable to using CUDA host callbacks because host callbacks
  // won't allow the work following the callback to run until the callback completes,
  // which comes at expense to perf.
  // 主线程 poll 时用 cudaEventQuery 探测，就绪才跑回调
  struct ncclIntruQueue<struct ncclCommEventCallback, &ncclCommEventCallback::next> eventCallbackQueue;

  // user-created reduction ops
  // freelist 数组：freeHead 指向下一个空槽，handle 与 comm 地址做 mangle 防跨 comm 误用
  int userRedOpCapacity, userRedOpFreeHead;
  ncclUserRedOp* userRedOps;

  // Queue of things for the main thread to do
  int reclaimSteps; // groupEnd 每处理一个 comm +1，攒够阈值就顺手 poll 一次 callbackQueue
  // MPSC：其它线程（host stream 回调等）投递 plan 回收请求，主线程单消费
  struct ncclIntruQueueMpsc<struct ncclCommCallback, &ncclCommCallback::next> callbackQueue;

  // ============ 生命周期状态机 ============
  ncclConfig_t config;
  // initState is to more conveniently reclaim resources when errors happen.
  ncclResult_t initState; // ncclInProgress -> ncclSuccess / 错误码
  // flag to indicate if ncclCommFinalize() is called
  bool finalizeCalled;
  // shared structures for finalization
  int finalizeRankCnt; // 仅 intraComm0 使用：同进程多 rank 销毁时的到达计数
  // group job to support multi-thread FT
  struct ncclGroupJob* groupJob;

  // Flag indicating if this communicator shares resources with parent or children
  bool shareResources;

  // ============ 可插拔插件 ============
  // Tuning plugin
  int tunerPluginLoaded;
  ncclTuner_t* tuner;
  void* tunerContext;

  // Profiler plugin
  void* profilerContext;
  uint64_t seqNumber[NCCL_NUM_FUNCTIONS]; // 按 collective 类型单调递增，profiler/RAS 用它对齐事件
  struct ncclProfilerCommState profiler;

  // ============ RMA / Debug / CE collective ============
  // RMA state
  struct ncclRmaState rmaState; // RMA proxy 线程态 + CE 侧 RMA context
  // 首个 symmetric window 注册时入队，groupEnd 时才真正 ncclRmaCeInit（延迟初始化）
  struct ncclIntruQueue<struct ncclRmaCeInitTask, &ncclRmaCeInitTask::next> rmaCeInitTaskQueue;

  // Debug check
  // checkMode == DEBUG_GLOBAL 时存参数快照，groupEnd 起异步 job 做跨 rank 一致性校验
  struct ncclIntruQueue<struct ncclArgsInfo, &ncclArgsInfo::next> argsInfoQueue;

  // CE Collective
  struct ncclCeColl ceColl; // 用 Copy Engine 而非 SM 执行的 collective 状态
  struct ncclIntruQueue<struct ncclCeInitTask, &ncclCeInitTask::next> ceInitTaskQueue;

  // ============ 注册缓存与全局能力汇总（下面这批多为 AllGather 后的归约结果）============
  // buffer registration cache
  struct ncclRegCache regCache; // 用户 buffer 注册缓存，按页对齐区间索引，避免重复注册 MR/IPC
  int isAllNvlink;        // 全体 GPU 路径 <= PATH_PIX（允许经 NVSwitch 间接）
  bool isAllDirectP2p; // Subject to NCCL_P2P_LEVEL (for local ranks only).
  bool isAllCudaP2p; // Raw CUDA capability (for local ranks only).
  bool isAllDirectNvlink; // All GPUs are directly connected to each other through NVLink.
                          // 比 isAllNvlink 更严：必须 <= PATH_NVL 直连
  int symmetricSupport; // 对称内存总开关，由 isAllCudaP2p + cuMem + GIN 能力共同决定
  int gpuCftSupport;    // 全体取 min，见 peerInfo 同名字段
  bool useNetPXN;       // 是否用了跨 GPU 的 net proxy（全局 OR）
  bool useGdr;          // 全体 net 连接是否都启用了 GDR（全局 AND）
  bool hasMloPart; // if mlopart is used
  bool hasMultiRankNvml; // if multiple ranks are using the NVML device
  ncclGinConnectionType_t globalGinSupport; // NONE / RAIL / FULL
  bool globalRmaProxySupport;
  bool hostRmaSupport;
  int childCount; // 已派生的子 comm 数，用来给子 comm 派生唯一 magic/commHash

  // ============ 对称内存运行时 ============
  struct ncclDevrState devrState; // The symmetric runtime state
  struct ncclSymkState symkState; // The symmetric kernels state (built on previous)

  // ============ 显存管理器 ============
  // 所有 ncclCudaMalloc 类分配都经它记账（persist/scratch/offload/peer import），split-share 时可复用父实例
  struct ncclMemManager* memManager;  // Memory manager
  // ncclCommSuspend/Resume 的延迟执行队列，groupEnd 时 drain
  struct ncclIntruQueue<struct ncclMemManagerTask, &ncclMemManagerTask::next> suspendTaskQueue;
  struct ncclIntruQueue<struct ncclMemManagerTask, &ncclMemManagerTask::next> resumeTaskQueue;

  uint64_t endMagic; // 与 startMagic 配对的尾哨兵
};

static_assert(offsetof(struct ncclComm, startMagic) == 0, "startMagic must be the first field of ncclComm");
static_assert(offsetof(struct ncclComm, endMagic) == sizeof(struct ncclComm) - sizeof(uint64_t),
              "endMagic must be the last field of ncclComm");

enum ncclLaunchMode {
  ncclLaunchModeInvalid = 0,
  ncclLaunchModeParallel,
  ncclLaunchModeGroup
};
extern enum ncclLaunchMode ncclParamLaunchMode;

void ncclCommPushFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* buf);
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle);

inline ncclResult_t ncclCommPollCallbacks(struct ncclComm* comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  struct ncclCommCallback* cb = ncclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  while (cb != nullptr) {
    struct ncclCommCallback* next = cb->next;
    ncclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != ncclSuccess) result = res1;
    cb = next;
  }
  NCCLCHECK(result);
  return ncclSuccess;
}

inline ncclResult_t ncclCommPollEventCallbacks(struct ncclComm* comm, bool waitSome) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  while (true) {
    struct ncclCommEventCallback* cb = ncclIntruQueueHead(&comm->eventCallbackQueue);
    if (cb == nullptr) break;
    cudaError_t ok;
    if (waitSome) {
      ok = cudaEventSynchronize(cb->event);
      waitSome = false;
    } else {
      ok = cudaEventQuery(cb->event);
      if (ok == cudaErrorNotReady) break;
    }
    ncclIntruQueueDequeue(&comm->eventCallbackQueue);
    if (ok == cudaSuccess) {
      NCCLCHECKGOTO(cb->fn(comm, cb), result, finish);
    } else {
      CUDACHECKGOTO(ok, result, finish);
    }
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return ncclSuccess;
}

inline void ncclCommIntraBarrierIn(struct ncclComm* comm, uint32_t x) {
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // Release everyone (just me).
    comm->intraBarrierGate = (uint64_t(x) << 32) | (phase ^ 1);
  } else {
    struct ncclComm* comm0 = comm->intraComm0;
    uint64_t count =
      COMPILER_ATOMIC_ADD_FETCH(&comm0->intraBarrierCounter, (uint64_t(x) << 32) + 1, std::memory_order_release);
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // Reset.
      COMPILER_ATOMIC_STORE(&comm0->intraBarrierCounter, 0ULL, std::memory_order_relaxed);
      // Release everyone.
      COMPILER_ATOMIC_STORE(&comm0->intraBarrierGate, (count >> 32 << 32) | (phase ^ 1), std::memory_order_release);
    }
  }
}

// returns sum of x values contributed to ncclCommIntraBarrierIn(comm, x)
inline uint32_t ncclCommIntraBarrierOut(struct ncclComm* comm) {
  struct ncclComm* comm0 = comm->intraComm0;
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  uint64_t gate = COMPILER_ATOMIC_LOAD(&comm0->intraBarrierGate, std::memory_order_relaxed);
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // Spin vigorously for first 5us.
      if (clockNano() - t0 >= 5 * 1000) std::this_thread::yield();
      gate = COMPILER_ATOMIC_LOAD(&comm0->intraBarrierGate, std::memory_order_relaxed);
    } while ((gate & 1) != phase);
  }
  if (comm->intraRanks != 1) std::atomic_thread_fence(std::memory_order_acquire);
  return gate >> 32;
}

// Scrambles the bits of non-builtin values of ncclRedOp_t according to the
// communicator memory address. Used to catch bugs so that integer handles
// associated with this communicator won't collide with handles of other
// communicatrs. This function is its own inverse.
static inline ncclRedOp_t ncclUserRedOpMangle(ncclComm* comm, ncclRedOp_t op) {
  // Preserve the built-in values.
  if (int(op) < int(ncclNumOps)) return op;
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
  h &= int(ncclMaxRedOp); // ncclMaxRedOp is a power of 2 minus 1
  int op1 = int(h) ^ int(op);
  // Since builtin values are preserved, we also have to preserve their preimage.
  return op1 < int(ncclNumOps) ? op : ncclRedOp_t(op1);
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm);
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);

// Process-wide NCCL_CTA_POLICY env override, or NCCL_CONFIG_UNDEF_INT when unset/invalid.
int ncclGetEnvCtaPolicy();

#endif
