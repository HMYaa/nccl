/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "device.h"
#include "graph.h"
#include "nvmlwrap.h"
#include "core.h"

#define NTRANSPORTS 4
#define TRANSPORT_UNDEFINED -1
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3

#include "proxy.h"
#include "comm.h"
#include "bootstrap.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;
extern struct ncclTransport collNetTransport;
// ncclTransports[NTRANSPORTS] = { P2P, SHM, NET, COLLNET }   ← 数组下标即优先级
// 每个: { name, canConnect, send:ncclTransportComm, recv:ncclTransportComm }
extern struct ncclTransport* ncclTransports[];
// Forward declarations
struct ncclRing;
struct ncclConnector;
struct ncclComm;

int64_t ncclParamMultiSegmentRegister();
extern int64_t ncclParamNvlsEnable();

// 每个 rank 的"能力名片"。init 时由 fillInfo() 填好，经 bootstrapAllGather 一次性全量交换
// （见 init.cc），之后作为 transport->canConnect() 的输入向量，用于判定两个 rank 之间能走哪条链路。
// 注意：整体按 sizeof(struct ncclPeerInfo) 交换，字段顺序即线上布局，不要随意重排。
struct ncclPeerInfo {
  // === 身份标识 ===
  int rank;
  int cudaDev; // CUDA_VISIBLE_DEVICES 过滤后的序号
  int nvmlDev; // NVML(NVIDIA Management Library,一套 GPU 管理/查询库) 序号，不受 CUDA_VISIBLE_DEVICES 影响；用于日志，以及判定多 rank 复用同一物理卡

  // === 内存/DMA 能力（另见下方 cuMemGdrSupport）===
  int gdrSupport; // GPUDirect RDMA：NIC 能否直接读写显存，决定 NET 传输是否绕开 host bounce buffer

  // === 局部性判定：决定 P2P/SHM 是否可用 ===
  uint64_t hostHash; // 主机指纹（含 commHash）。相等 => 同主机 => 可考虑 P2P/SHM
  uint64_t pidHash;  // 进程指纹（含 commHash）。与 hostHash 同时相等 => 同进程（P2P_SAME_PID）
                     // => 可直接用裸指针，无需 CUDA IPC
  dev_t shmDev;      // /dev/shm 的 st_dev。容器间若不是同一个 shm 挂载点则 SHM 传输不可用

  // === 拓扑定位 ===
  int64_t busId;           // PCI busId，作为 topo 图中 GPU 节点的 key
  cudaUUID_t gpuUuid;      // 用于检测多个 rank 是否绑定到同一 GPU/分区
  struct ncclComm* comm;   // 同进程时可直接解引用对端 comm，用来串起 intraNext 进程内 comm 链表

  // === GPU 属性 ===
  int cudaCompCap;         // SM 架构版本
  int gpuCftSupport;       // CFT 所需的 CUDA 版本，0 表示不支持；comm 级取所有 rank 的最小值
  size_t totalGlobalMem;   // 显存容量（上取整到 4GB）；对称内存取全体最大值作为 VA 窗口 stride

  // === MNNVL：跨节点 NVLink domain ===
  nvmlGpuFabricInfoV_t fabricInfo; // clusterUuid/cliqueId/state；同 cluster 同 clique 才能走 MNNVL
  int fabricHandleSupport;         // 能否导出 fabric handle（CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED）
  int cuMemSupport;                // VMM/cuMem API 是否可用，决定 P2P 走 cuMem 还是 legacy CUDA IPC

  // === 版本校验 ===
  int version; // NCCL_VERSION_CODE，所有 rank 必须一致，否则 init 直接报错

  // === 网络/对称内存能力（这几项 comm 级都取全体交集）===
  uint64_t supportedGinTypeBitMask; // 本 rank 可用的 GIN backend 位掩码（GDAKI/PROXY...）
  bool crossNicSupport;             // 是否允许跨 NIC 通信，影响 GIN 是 FULL 还是仅 RAIL 连接
  bool rmaPluginAvailable;          // RMA plugin 是否已加载
  bool cuMemGdrSupport;             // GDR 与 CUDA VMM 能否共存（GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED）

  // === 分区 GPU ===
  int mloPart; // MLOPart partition index, or -1 if not an MLOPart GPU
};

#define CONNECT_SIZE 256
#define NCCL_MAX_PAGE_SIZE (512L * 1024L * 1024L)
#define NCCL_REC_PAGE_SIZE (2L * 1024L * 1024L)
struct ncclConnect {
  char data[CONNECT_SIZE];
};

#if CUDART_VERSION >= 12010

#define NVLS_HANDLE_SIZE 64
struct ncclNvlsSharedRes {
  int refCount;
  bool inited;
  CUmulticastObjectProp bufProp;
  CUmulticastObjectProp signalProp;
  CUmemAccessDesc accessDesc;
  int dev;
  size_t creditUCSize;
  size_t creditMCSize;
  size_t buffUCSize;
  size_t buffMCSize;
  CUmemGenericAllocationHandle mcBuffHandle; // Multicast handle for NVLS buffer
  CUmemGenericAllocationHandle mcCreditHandle; // Multicast handle for NVLS credit buffer
  char* mcBuff; // Multicast NVLS buffer address
  char* mcCredit; // Multicast NVLS credit address
  CUmemGenericAllocationHandle ucBuffHandle; // Unicast Handle for NVLS buffer
  CUmemGenericAllocationHandle ucCreditHandle; // Unicast Handle for NVLS credit buffer
  char* ucBuff; // Unicast NVLS buffer address
  char* ucCredit; // Unicast NVLS credit address
  int nChannels;
  int nHeads;
  int chunkSize;
  int treeMaxChunkSize;
  struct ncclShmemCollBuff nvlsShmem;
  void* nvlsShmemHandle;
};

#endif /* CUDART_VERSION >= 12010 */

struct ncclCollNetSharedRes {
  int refCount;
  int size;
  char* cudaBuff;
  char* hostBuff;
  struct ncclProxyArgs* proxyAppend[2 * NCCL_MAX_NETDEVS];
  void* resources;
  int nChannels;
  size_t buffSize;
};
/*

GPU 侧的数据面完全不碰 vtable，碰 vtable 的只有 CPU proxy
  那一半。

  精确版本

     1 │              ┌── host 控制面：建链 ──────────────────────┐
     2 │ vtable 覆盖  │   setup / connect / free                  │  函数指针调用
     3 │              │   proxySetup / proxyConnect / proxyRegister│
     4 │              ├── CPU 数据面：proxy progress 循环 ─────────┤
     5 │              │   proxyProgress  ← 每轮循环通过指针调用    │
     6 │              └───────────────────────────────────────────┘
     7 │
     8 │              ┌── GPU 数据面：kernel ─────────────────────┐
     9 │ vtable 不覆盖│   只读 ncclConnInfo 里的裸指针            │  零间接跳转
    10 │              │   buffs[] / head / tail / connFifo / flags │
    11 │              └───────────────────────────────────────────┘

*/
//10 个指针：控制面被劈成两半
struct ncclTransportComm {
  // --- 调用者线程（ncclCommInitRank 路径）

  // 本地分配 + 往 ncclConnect 里填自己的信息
  ncclResult_t (*setup)(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*,
                        struct ncclConnect*, struct ncclConnector*, int channelId, int connIndex);
  // 收到对端信封后完成 attach
  ncclResult_t (*connect)(struct ncclComm* comm, struct ncclConnect*, int nranks, int rank, struct ncclConnector*);
  ncclResult_t (*free)(struct ncclComm* comm, struct ncclConnector*);

  // --- ncclProxyCallAsync（跨线程/跨进程 RPC）

  ncclResult_t (*proxySharedInit)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                  int nChannels);
  ncclResult_t (*proxySetup)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff,
                             int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyConnect)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff,
                               int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyFree)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState);
  // 数据面！轮询 CQ、推进 FIFO
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*);
  // MR 注册
  ncclResult_t (*proxyRegister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyDeregister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState,
                                  void* reqBuff, int reqSize, int* done);
};

struct ncclTransport {
  const char name[8];
  ncclResult_t (*canConnect)(int*, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*,
                             struct ncclPeerInfo*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend,
                                     int* peerSend, int connIndex);
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex);
ncclResult_t ncclTransportCheckP2pType(struct ncclComm* comm, bool* isAllDirectP2p, bool* directMode,
                                       bool* isAllCudaP2p);
bool ncclP2pUsesMemcpy();

ncclResult_t ncclNvlsInit(struct ncclComm* comm);
ncclResult_t ncclNvlsTuning(struct ncclComm* comm);
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm);
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm);
ncclResult_t ncclNvlsGraphRegisterBuffer(
  struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize, size_t recvbuffSize,
  int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);
ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv);
ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle* mcHandler, CUdeviceptr ptr,
                                 int dev, size_t ucsize, size_t mcsize);
ncclResult_t ncclNvlsFree(struct ncclComm* comm);

enum {
  collNetRecv = 0,
  collNetSend = 1
};
bool ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel,
                               int masterRank, int masterPeer, int collNetGraphChannelId, int type,
                               ncclConnect* connect);
ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail);
ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm);
ncclResult_t ncclCollnetLocalRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type,
                                            int* outRegBufUsed, void** outHandle);
ncclResult_t ncclCollnetGraphRegisterBuffer(
  struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufFlag, void** outHandle,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);
ncclResult_t ncclCollnetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyconn, void* handle);

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm);
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm);
ncclResult_t ncclTransportInitRankMap(struct ncclComm* comm, int nHeads, const int* heads);
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm);

ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]);
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm);
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm);

ncclResult_t ncclNetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* handle);
ncclResult_t ncclNetLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize,
                                        struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag,
                                        void** outHandle);
ncclResult_t ncclNetGraphRegisterBuffer(
  ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers,
  int* outRegBufFlag, void** outHandle,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);

ncclResult_t ncclRegisterP2pIpcBuffer(
  struct ncclComm* comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
ncclResult_t ncclRegisterP2pNetBuffer(
  struct ncclComm* comm, void* userbuff, size_t size, struct ncclConnector* conn, int* regFlag, void** handle,
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
ncclResult_t ncclRegisterCollBuffers(
  struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
  void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS],
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect);
ncclResult_t ncclRegisterCollNvlsBuffers(
  struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS],
  void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS],
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect);
ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, ncclFunc_t func, int* recChannels);

#if CUDART_VERSION >= 12010
ncclResult_t ncclNvlsGroupCreate(struct ncclComm* comm, CUmulticastObjectProp* prop, int rank, unsigned int nranks,
                                 CUmemGenericAllocationHandle* mcHandle, char* shareableHandle);
ncclResult_t ncclNvlsGroupConnect(struct ncclComm* comm, char* shareableHandle, int rank,
                                  CUmemGenericAllocationHandle* mcHandle);
#endif

ncclResult_t ncclIpcSymmetricInit(struct ncclComm* comm);
ncclResult_t ncclIpcMapSymmetric(struct ncclComm* comm, size_t offset, size_t size,
                                 CUmemGenericAllocationHandle memHandle, void** symPtr);
ncclResult_t ncclIpcFreeSymmetric(struct ncclComm* comm, size_t size, void* symPtr);
ncclResult_t ncclIpcSymmetricFinalize(struct ncclComm* comm);
ncclResult_t ncclNvlsSymmetricInit(struct ncclComm* comm);
ncclResult_t ncclNvlsMapSymmetric(struct ncclComm* comm, size_t offset, size_t ucsize, void* ucaddr);
ncclResult_t ncclNvlsFreeSymmetric(struct ncclComm* comm, size_t ucsize, void* ucaddr);
ncclResult_t ncclNvlsSymmetricFinalize(struct ncclComm* comm);

#endif
