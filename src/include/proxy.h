/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PROXY_H_
#define NCCL_PROXY_H_

#include "device.h"
#include "info.h"
#include "socket.h"
#include "ipcsocket.h"
#include "nccl_net.h"
#include "shmutils.h"
#include "p2p.h"
#include "collectives.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin/gin_host.h"
#endif
#include "os.h"

#include <atomic>
#include <mutex>
#include <condition_variable>

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollnetChain,
  ncclPatternCollnetDirect,
  ncclPatternNvls,
  ncclPatternNvlsTree,
  ncclPatternPatUp,
  ncclPatternPatDown,
  ncclPatternSend,
  ncclPatternRecv,
} ncclPattern_t;

enum ncclProxyOpState {
  ncclProxyOpNone,
  ncclProxyOpReady,
  ncclProxyOpProgress
};

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyState*, struct ncclProxyArgs*);

#define NCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH <= MAXCHANNELS, "Not enough sub space for max work elements");

union ncclProxyOpSpecifics {
  struct {
    size_t sizePerRank;
    int nNodes, node;
  } collnetDirect;
  struct {
    int sendSlices;
    int recvSlices;
    int stepSize;
  } bcast;
};

// 「快递单」三层级之一：主线程 -> proxy 线程的跨线程「订单」。
// 按 channel、连接方向及任务生成，经共享 ncclProxyOpsPool 交给负责该连接的 Proxy；
// 纯参数、无进度状态。proxy 线程取出后合并成 ncclProxyArgs（批）+ ncclProxySubArgs（状态机）。
struct ncclProxyOp {
  struct ncclProxyConnection* connection; // 目标连接（proxy 侧对某条 ncclConnector 的映射）
  ssize_t nbytes;                         // 传输字节参数；含义需结合算法及 progress 的分块/注册路径
  uint64_t opCount;                       // 操作排序/合批标识，编码区分 collective 与 P2P
  int root;
  int next;                               // ops pool 内的链表下标
  int nsteps;                             // 本次任务的 step 工作量；按 sliceSteps 等粒度推进
  size_t chunkSize;
  size_t sliceSize;
  size_t loopSize;
  size_t loopOffset;
  size_t channelSize;
  uint8_t sliceSteps;
  uint8_t chunkSteps;
  uint8_t channelId;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclFunc_t*/ collAPI;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t protocol;
  uint8_t algorithm;
  uint8_t reg;
  // collnet/p2p/coll buffer reg handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  int isOneRPN;
  RingAlgorithm* ringAlgo;
  union ncclProxyOpSpecifics specifics;
  int nChannels;
  int nPeers;

  // Profiler plugin
  union {
    struct ncclTaskColl* coll;
    struct ncclTaskP2p* p2p;
  } task;
  int eActivationMask;
  void* taskEventHandle;
  int rank;
  int peer;
  ncclPid_t pid;
  void* profilerContext;
  uint64_t workCounter;

  struct ncclProxyOp* enqNext;
};

struct ncclProxySubArgs;

struct ncclProxyEventHandle {
  void* stepEventHandle;
  struct ncclProxySubArgs* subArgPtr;
};

// 「快递单」最小单元：一条连接上的传输任务及其进度。
// NET 中 base 是连接的绝对 step 起点，posted/received/transmitted/done 从 0 开始计相对进度；
// 访问连接计数器或 buffer 环槽时再加 base，不能写成 base <= done。
//
//   NET send：posted（提供 GPU 生产窗口）-> transmitted（数据就绪且 isend 提交）-> done（test 完成）
//   NET recv：posted（irecv 提交）-> received（接收完成）-> transmitted（完成所需 flush 后发布 tail）
//             -> done（GPU 消费完毕，回收窗口）
// NET recv 不单独推进 flushed；CollNet 等实现使用该字段，不能跨 transport 套用同一计数链。
// requests 保存网络插件的异步 request，不等同于 Verbs WQE；槽位索引和有效并发深度由 progress 决定。
struct ncclProxySubArgs {
  // ---- (a) 目标 ----
  struct ncclProxyConnection* connection; // 该 sub 服务的 Proxy 连接，关联 transportResources 与推进实现
  int reg;                                // 用户 buffer 已注册（零拷贝路径）
  // collnet handles
  void* sendMhandle;
  void* recvMhandle;
  uint8_t* sendbuff;
  uint8_t* recvbuff;
  size_t offset;
  ssize_t loopSize;
  ssize_t loopOffset;
  int channelId;
  // ---- (b) 任务量 ----
  int nsteps;                             // 总步数
  ssize_t nbytes;
  ssize_t chunkSize;
  int peer;
  int isOneRPN;
  RingAlgorithm* ringAlgo;
  int groupSize; // Number of consecutive sub operations sharing the same recvComm
  // ---- (c) 进度字段：使用哪些计数器由 transport 决定 ----
  uint64_t base;                          // NET 中由 resources->step 按 chunkSteps 向上对齐得到的绝对起点
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;                      // CollNet 等路径使用；NET recv 不单独推进
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;                           // 不作为 NET progress 的完成判据；NET 使用 nsteps 与相对进度
  int regBufferReady;
  void* requests[NCCL_STEPS];             // 插件异步请求槽；send/recv 分组与阶段可采用不同的索引方式

  // Profiler plugin
  int eActivationMask;
  int rank;
  ncclPid_t pid;
  void* profilerContext;
  void* taskEventHandle;
  void* opEventHandle;
  void* kernelEventHandle;
  struct ncclProxyEventHandle pHandles[NCCL_STEPS];
  size_t transSize;
  uint64_t workCounter;

  void* recvRequestsCache[NCCL_STEPS];
  int recvRequestsSubCount;
};

// 「快递单」中间层：Proxy 合批后的 sub 集合，共用 progress 与操作参数；也可承载 P2P。
// progressOps 遍历 active 链，调用 args->progress(proxyState, args) 推进该批 subs。
struct ncclProxyArgs {
  // ---- (a) 批 ----
  struct ncclProxySubArgs subs[NCCL_PROXY_MAX_SUBS];
  proxyProgressFunc_t progress;           // 该批用哪个 transport 的推进函数（如 net.cc 的 sendProxyProgress）
  int nsubs;
  int done;                               // 已完成的 sub 数；== nsubs 时整批结束
  int onePPN;
  // ---- (b) 本批操作的公共参数（来自 ncclProxyOp）----
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  size_t chunkSize;
  size_t totalSendSize;
  size_t totalRecvSize;
  size_t sendSizePerRound;
  size_t recvSizePerRound;
  uint8_t /*ncclDataType_t*/ dtype;
  uint8_t /*ncclDevRedOp_t*/ redOp;
  uint8_t /*ncclPattern_t*/ pattern;
  uint8_t /*ncclFunc_t*/ coll;
  uint8_t /*ncclFunc_t*/ collAPI;
  uint8_t protocol;
  uint8_t algorithm;
  // ---- (c) 状态 ----
  int state;                              // ncclProxyOpReady -> ncclProxyOpProgress -> ncclProxyOpNone
  char* sharedBuff[NCCL_STEPS];           // 共享 buffer 模式的每步暂存区
  int sharedSize[NCCL_STEPS];
  int nChannels;
  int nPeers;

  int idle;                               // 本轮是否毫无推进；连续 idle 时 proxy 线程 backoff

  // ---- (d) 链表 ----
  // Element linking
  struct ncclProxyArgs* next;             // active 链（progressOps 遍历）
  struct ncclProxyArgs* nextPeer;         // 同一 proxyAppendPtr 队列上的下一批，保证该队列 FIFO
  struct ncclProxyArgs** proxyAppendPtr;  // 队列尾指针；可为连接私有，也可指向 transport 的共享队列

  union ncclProxyOpSpecifics specifics;
};
#define NCCL_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// Make sure we have enough to store two full rounds of operations on all channels.
// Otherwise we'd be unable to post half of them to free new elements. Each
// p2p work contains a send and recv proxy op hence the 2x before it.
#define MAX_OPS_PER_PEER (2 * MAXCHANNELS * 2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH)

struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER * NCCL_MAX_LOCAL_RANKS];
  volatile int nextOps;
  volatile int nextOpsEnd;
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];
  int syncObjectsInitialized;
  std::mutex mutex;
  std::condition_variable cond;
};

struct ncclProxyOps {
  ncclProxyOpsPool* pool;
  ncclShmHandle_t handle;
  int count;
  int freeOp;
  int nextOps;
  int nextOpsEnd;
};

struct ncclProxySharedP2p {
  int refcount;
  ssize_t size;
  char* cudaBuff;
  char* hostBuff;
  // CUDA IPC
  ncclIpcDesc ipcDesc;
  struct ncclProxyArgs* proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct ncclProxyPeer {
  struct ncclProxySharedP2p send;
  struct ncclProxySharedP2p recv;
};

struct ncclSharedNetComms {
  int activeConnect[MAXCHANNELS];
  int activeAccept[MAXCHANNELS];
  void* sendComm[MAXCHANNELS];
  void* recvComm[MAXCHANNELS];
  int sendRefCount[MAXCHANNELS];
  int recvRefCount[MAXCHANNELS];
};

struct ncclProxyPool;
struct ncclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct ncclProxyOpsPool* opsPool;
  ncclShmHandle_t handle;
  char opsPoolShmSuffix[16];

  std::thread thread;
  std::atomic<int> stop{0};
  struct ncclProxyPeer** localPeers;
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS];
  struct ncclProxyArgs* active;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
  int nextOps;
};

// Expected proxy response fifo
struct ncclExpectedProxyResponse {
  void* opId;
  int respSize;
  bool done;
  void* respBuff;
  ncclResult_t res;
  struct ncclExpectedProxyResponse* next;
};

struct ncclProxyAsyncOp {
  int type;
  struct ncclProxyConnection* connection;
  int reqSize, respSize;
  char *reqBuff, *respBuff;
  void* opId;
  ncclProxyAsyncOp* next;
};

struct ncclProxyLocalPeer {
  struct ncclSocket sock;
  int tpRank;
  int tpLocalRank;
  ncclProxyAsyncOp* asyncOps;
  int asyncOpCounter;
};

// Common response header for all proxyOps
// We pack this into a struct to reduce the number of blocking send and recv calls
struct ncclProxyRpcResponseHeader {
  void* opId;
  ncclResult_t res;
  int respSize;
};

// UDS support
struct ncclIpcHdr {
  int type;
  int rank;
  int reqSize;
  int respSize;
  void* opId;
  uint64_t data[16]; // 128-bytes
};

struct ncclProxyState {
  int refCount;
  struct ncclComm* comm;
  int tpRank;
  int tpnRanks;
  int tpLocalnRanks;
  int cudaDev;
  int p2pnChannels;
  int p2pChunkSize;
  int nChannels;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;
  bool dmaBufSupport;
  ncclNet_t* ncclNet;
  ncclCollNet_t* ncclCollNet;
  struct ncclGinState* ginState;
  uint32_t* abortFlag;
  bool directMode;
  struct ncclMemManager* memManager;  // Shared memory manager for proxy allocations
  // Service threads
  std::thread thread;
  std::thread threadUDS;
  struct ncclSocket* listenSock;
  struct ncclIpcSocket ipcSock;
  int stop;
  ncclResult_t asyncResult;

  // Used by main thread
  union ncclSocketAddress* peerAddresses;
  struct ncclSocket* peerSocks;
  struct ncclProxyOps* proxyOps;
  void** sharedDevMems;
  int peerArraySize;  // Size of peerSocks/proxyOps/sharedDevMems arrays (tpNRanks)
  struct ncclIpcSocket peerIpcSock; // cuMEM API support (UDS)
  uint64_t* peerAddressesUDS; // cuMem API support (UDS)

  // Progress thread
  struct ncclProxyProgressState progressState;

  // Network plugin
  void* netContext;
  ncclNetAttr_t netAttr;
  void* collNetContext;

  // Profiler plugin
  void* profilerContext;

  // Queue of expected responses from the proxy
  struct ncclExpectedProxyResponse* expectedResponses;
};

enum proxyConnectState {
  connUninitialized = 0,
  connInitialized = 1,
  connSharedInitialized = 2,
  connSetupDone = 3,
  connConnected = 4,
  numConnStates = 5
};

struct proxyMemHandle {
  void* handle;
  struct proxyMemHandle* next;
};

struct ncclProxyConnection {
  int send, transport, shared;
  int tpLocalRank, sameProcess;
  struct ncclSocket* sock;
  struct ncclTransportComm* tcomm;
  struct ncclProxyArgs* proxyAppend;
  struct ncclProxyArgs** proxyAppendPtr;
  void* transportResources;
  ncclNetDeviceHandle_t* netDeviceHandle;
  void* mhandles[NCCL_NUM_PROTOCOLS];
  proxyConnectState state;
  struct ncclCollNetSharedRes* collNet;
  int needsProxyProgress;
  struct ncclIntruQueue<struct proxyMemHandle, &proxyMemHandle::next> proxyMemHandleQueue;
};

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* proxyOp, bool* justInquire);
ncclResult_t ncclProxyStart(struct ncclComm* comm);
ncclResult_t ncclProxyInit(struct ncclComm* comm, struct ncclSocket* sock, union ncclSocketAddress* peerAddresses,
                           uint64_t* peerAddressesUDS);
ncclResult_t ncclProxyCreate(struct ncclComm* comm);
ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int proxyRank,
                              struct ncclProxyConnector* proxyConn);

// NB: ncclProxyMsgTypeStr[] in proxy.cc needs to match
enum ncclProxyMsgType {
  ncclProxyMsgInit = 1,
  ncclProxyMsgSharedInit = 2,
  ncclProxyMsgSetup = 3,
  ncclProxyMsgConnect = 4,
  ncclProxyMsgStart = 5,
  ncclProxyMsgClose = 6,
  ncclProxyMsgAbort = 7,
  ncclProxyMsgStop = 8,
  ncclProxyMsgGetFd = 9, // cuMem API support (UDS)
  ncclProxyMsgQueryFd = 10,
  ncclProxyMsgRegister = 11,
  ncclProxyMsgDeregister = 12
};

// This function is called by a client of the proxy that needs to invoke any of the non-progress proxyOp types
// Call this function on the client, supplying a locally unique opId. Then, poll on the return value of
// ncclPollProxyResponse(), supplying the same opId to confirm the operation has completed
ncclResult_t ncclProxyCallAsync(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff,
                                int reqSize, int respSize, void* opId);

// This function will internally call ncclProxyCallAsync() and spin until ncclPollProxyResponse() confirms the result
// is received
ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int type, void* reqBuff,
                                   int reqSize, void* respBuff, int respSize);
ncclResult_t ncclPollProxyResponse(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* respBuff,
                                   void* opId);

// UDS support
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int rank, void* handle, int* convertedFd);
ncclResult_t ncclProxyClientQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, int localFd,
                                            int* rmtFd);
ncclResult_t ncclProxyClientBatchQueryFdBlocking(struct ncclComm* comm, struct ncclProxyConnector* proxyConn,
                                                 int* localFds, int* rmtFds, int numSegments);

ncclResult_t ncclProxyStop(struct ncclComm* comm);
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm);
ncclResult_t ncclProxyDestroy(struct ncclComm* comm);
#endif
