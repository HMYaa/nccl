/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_COMMON_H_
#define NET_IB_COMMON_H_

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "profiler/net_ib.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"
#include "wqe_lat_mon.h"

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)
extern char ncclIbIfName[MAX_IF_NAME_SIZE + 1];
extern union ncclSocketAddress ncclIbIfAddr;

enum ncclIbRequestMatchingScheme {
  BY_INDEX = 0,
  BY_ID = 1,
};

struct ncclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr* mr;
};

struct ncclIbMrCache {
  struct ncclIbMr* slots;
  int capacity, population;
};

extern int ncclNMergedIbDevs;
#define NCCL_IB_MAX_DEVS_PER_NIC NCCL_NET_MAX_DEVS_PER_NIC
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE * NCCL_IB_MAX_DEVS_PER_NIC) + NCCL_IB_MAX_DEVS_PER_NIC
// 「虚拟 NIC」：ncclNet 对外暴露的 dev 号指的是它，不是物理口。默认每个物理口一个；
// 拓扑层（topo.cc · ncclTopoMakeVNics → makeVDevice）可把多个物理口合并为一个，之后一条连接的 QP 条带到所有口上。
struct alignas(64) ncclIbMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  int16_t railId;
  int16_t planeId;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

struct ncclIbStats {
  int fatalErrorCount;
};

enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

// 一个物理 HCA 端口（ncclIbInitDevices 枚举，进程内全局 ncclIbDevs[]）。PD 在此按设备共享（pdRefs 计数），
// MR 缓存也挂在这里（按页对齐 + 引用计数，跨 comm 复用）；每个设备一条 ncclIbAsyncThread 收 async event
// 并把致命错误累计到 stats.fatalErrorCount，数据面每次 isend/irecv/test 前检查。
extern int ncclNIbDevs;
struct alignas(64) ncclIbDev {
  std::mutex mutex;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  uint64_t currSpeed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char fullPciPath[PATH_MAX];
  char* pciPath;
  int realPort;
  int maxQp;
  float latency;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  uint32_t oooRqSize;  // valid only when ar=1
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
  int16_t railId;
  int16_t planeId;
  int16_t planeIdx;
  enum ncclIbProvider ibProvider;
  union {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
  struct ncclIbGidInfo gidInfo;
};

#define MAX_IB_DEVS 32
#define MAX_IB_VDEVS MAX_IB_DEVS * 8
extern struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
extern struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
extern int ncclIbRelaxedOrderingEnabled;
extern uint64_t ncclIbSpeedChangeCounter;
extern int64_t ncclParamIbEventBasedLb();
extern int64_t ncclParamIbEventBasedLbRemote();

#define NCCL_IB_LLSTR(ll) \
  (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

struct alignas(32) ncclIbRemoteSpeedBuf {
  volatile uint64_t counter;
  uint16_t speedGbps[NCCL_IB_MAX_DEVS_PER_NIC];
};
static_assert(sizeof(ncclIbRemoteSpeedBuf) == 32);

// Per-Dev connection metadata
struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE and IB Rounter
  union ibv_gid gid;

  // The key used for remote access to the addr exchanged by the peers
  // in ncclIbConnectionMetadata::addr
  // This member is populated differently on the sender and on the receiver
  // side.
  // The sender side populates this member with the RKey obtained after it
  // registered the CTS FIFO (on the specific device).
  // The receiver side populates this member with the RKey obtained after it
  // registered the completion records structure (on the specific device).
  uint32_t rkey;

  // remote dev info
  union ibv_gid remoteGid;

  uint64_t currSpeed;
  uint32_t remSpeedBufRkey;
};

#define MAX_QPS_PER_REQ 8
struct ncclProfilerInfo {
  void* qpEventHandles[MAX_QPS_PER_REQ];
  int qpIndex[MAX_QPS_PER_REQ];
  int nEventHandles;
  ncclProfilerNetIbDescr_v1_t data;
  void* pHandle;
};

#define NCCL_NET_IB_MAX_RECVS 8

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
#define NCCL_NET_IB_REQ_GIN_IPUT 4
#define NCCL_NET_IB_REQ_GIN_IGET 5
extern const char* ncclIbReqTypeStr[];

// Maximal number of QPs a communicator can have for data transfers
#define NCCL_IB_MAX_QPS 128

// ============ 数据面协议总览（读结构体前先看这段）============
// 一条 NET 连接 = 1 条 TCP（只建链）+ nqps 个 RC QP（数据）+ 两张「对端可 RDMA WRITE 直写」的 host 表：
//   sender  侧 ncclIbSendComm::ctsFifo          ← receiver 在 irecv 时写 CTS（addr/rkeys/tag/idx）
//   receiver 侧 ncclIbRecvComm::cmplsRecords     ← sender 在多收（nreqs>1）时写 sizes[]
// 每张表在对端都有一份同布局的「影子」（remCtsFifo.elems / remCmplsRecords.elems），本地注册 MR 取 lkey
// 作 gather 源，对端表的 addr+rkey 在建链 metadata 里交换。数据本身由 sender RDMA WRITE 直写 CTS 给的地址，
// 末尾一条 RDMA WRITE WITH IMM 触发 receiver 预投的空 recv WR 产生 CQE。没有 SEND/RECV 语义的数据报文。
// 槽位：两侧各自的 base.fifoHead 单调递增，slot = fifoHead % NET_IB_MAX_REQUESTS(256)，请求 id = fifoHead。

// receiver 侧每槽一份的「完成记录」，被 sender RDMA WRITE 直写（sizes）或 receiver 本地填（completions）。
// Tracks data transfers between sender and receiver. A multi-recv/send uses a
// single record.
struct ncclIbRequestCompletionRecord {
  // This array communicates data transfer sizes from the sender to the
  // receiver. The sender writes the size of each completed data transfer to
  // this array. The receiver reads these sizes before reporting completion of
  // the corresponding receive request to the user.
  int sizes[NCCL_NET_IB_MAX_RECVS];
  // The receiver fills this array to signal the completion of a data transfer.
  // The sender can then read this array to see the receiver's status. If the
  // sender detects an error or device failure, it reads this array to
  // determine if the receiver considered the transfer complete. This prevents
  // the sender from retransmitting data if the failure was only visible on the
  // sender's side. Based on the array's contents, the sender decides if, how,
  // and on which QPs/devices to replay the transfer.
  bool completions[NCCL_IB_MAX_QPS];
};

// 一次 isend / irecv / iflush 的 host 侧句柄（即 ncclNet 返回给 Proxy 的 request，存在 sub->requests[]）。
// 生命周期：ncclIbGetRequest 从 base.reqs[] 找 type==UNUSED 的槽 → 提交时按「每 QP 一个 signaled CQE」预置
// events[dev] → ncclIbTest 逐设备 poll_cq，按 wr_id 反查到本请求后 events[dev]-- → 全零即完成，FreeRequest 置回 UNUSED。
// 反查键：sender 侧 wr_id 低 8 bit = slot（multi-send 每个请求占 8 bit）；receiver 侧 BY_INDEX 用 wr_id = slot，
// BY_ID 用 imm_data = id % 2^32；flush 用 wr_id = reqIndex + NCCL_IB_FLUSH_REQ_WR_ID_OFFSET。
struct ncclIbRequest {
  // ---- (a) 身份 ----
  struct ncclIbNetCommBase* base; // 所属 comm（send 或 recv），从中取 vProps / qps / reqs
  int type;                       // NCCL_NET_IB_REQ_*：UNUSED 即空闲槽
  struct ncclSocket* sock;
  // ---- (b) 完成计数：按物理设备维度，不是按 QP ----
  // Array of counters. Each element in the array is populated with the expected
  // number of completion events that the request is expecting to be generated
  // on the device corresponding to the index of the element. After the request
  // is initialized and posted, for every completion event generated by a
  // device, the corresponding counter is decremented. When the counter reaches
  // zero it means that the request was fully completed on that device.
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to the per-device base structures to make it easier to
  // poll the device's CQ when the request is tested for progress.
  // The pointers are initialized only for the devices that the request expects
  // to receive completions from.
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct ncclProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];
#endif
  // ---- (c) 槽位与多收 ----
  uint64_t id;   // = 提交时的 base.fifoHead；slot = id % NET_IB_MAX_REQUESTS
  int nreqs;     // 同一槽的 multi-recv 路数（irecv 的 n，isend 从 CTS 的 nreqs 读回），1..NCCL_NET_IB_MAX_RECVS
  // ---- (d) 方向私有 ----
  union {
    struct {
      int size;      // min(用户 size, CTS.size)
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC]; // 每物理设备一个 lkey（regMr 每口注册一次），按 QP 所在设备选
      // Tracks whether data was transmitted on a QP for this request.
      bool sentData[NCCL_IB_MAX_QPS];
      // Per-device LB weights used for chunk computation.
      uint8_t weights[NCCL_IB_MAX_DEVS_PER_NIC];
    } send;
    struct {
      struct ncclIbRequestCompletionRecord* cmplsRecords;
      // Aggregates the size of a send request when sender does not write to the
      // completion records array.
      int aggSize;
    } recv;
    struct {
      int rank;
    } iput;
    struct {
      int rank;
    } iget;
  };
  void* rmaProxyCtx;
};

// comm 在「一个物理设备」上的 Verbs 资源：PD 按设备全局共享（ncclIbDev::pd + pdRefs），CQ 按 comm×设备私有，
// 该设备上的所有 QP（数据 / CTS / flush）共用这一个 CQ，ncclIbTest 只 poll 这里。
struct ncclIbNetCommDevBase {
  int ibDevN;        // ncclIbDevs[] 下标
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
  // Resolved once at device init and reused by every QP (like the GID index above).
  int pkeyIndex;
};

// Snapshot the device-wide GID info into a comm's per-device base under a mutex.
static inline void ncclIbGidInfoSnapshot(struct ncclIbNetCommDevBase* base, struct ncclIbDev* ibDev) {
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  base->gidInfo = ibDev->gidInfo;
}

// 一条 CTS（Clear-To-Send）：receiver 在 irecv 时填好，RDMA WRITE 到 sender 的 ctsFifo[slot][r]。
// 64 B、32 B 对齐（见下方 static_assert），idx 放最后当 valid flag：sender 先读 idx == fifoHead+1，
// fence，再读 addr/rkeys/tag。「一条不被拆写」是对 Relaxed Ordering 下 PCIe 写事务的假设。
struct ncclIbSendFifo {
  uint64_t addr;                            // receiver 用户 buffer 地址（数据 WRITE 的 remote_addr）
  uint64_t size;                            // receiver 期望的最大长度，sender 取 min
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC]; // receiver 每物理设备一个 rkey，sender 按 QP 的 remDevIdx 选
  uint32_t nreqs;                           // 本槽 multi-recv 路数
  uint32_t tag;                             // 对号用；net.cc 传 tpRank / tpRemoteRank
  uint64_t idx;                             // = receiver fifoHead+1，有效标志兼序号
};

struct ncclIbQpInitAttr {
  ibv_qp_state state;
  int pkeyIndex;
  uint8_t portNum;
  int qpAccessFlags;
};

struct ncclIbQpRtrAttr {
  enum ibv_mtu mtu;
  uint8_t linkLayer;
  uint8_t tc;
  int sl;

  uint32_t remoteQpNum;
  uint32_t remoteLid;
  union ibv_gid remoteGid;

  uint8_t localIbPort;
  uint8_t localPortFlags;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

struct ncclIbQpRtsAttr {
  int timeout;
  int retryCnt;
};

// 一个 RC QP 及其「可重放」的建链参数：Init/Rtr/Rts 三组 attr 建链时填好、故障恢复时原样重灌。
// 数据 QP 按 qpIndex % ndevs 条带到 merged 设备上；sender 侧 QP maxRecvWr=0（不收报文），
// receiver 侧 QP 的 SQ 只发 CTS、RQ 预投空 WR 接 IMM。
struct ncclIbQp {
  struct ibv_qp* qp;
  // The index of the device on which this QP was created on.
  int devIndex;  // 本地 merged 设备下标（vProps.devs[]）

  // The ECE (enhanced connection establishment) used on this QP.
  // Note: This is the reduced ECE exchanged between the sender and receiver.
  struct ibv_ece ece;
  int eceSupported;

  // Stores the attributes used to configure the QP to allow QP restore after
  // failure.
  struct ncclIbQpInitAttr initAttr;
  struct ncclIbQpRtrAttr rtrAttr;
  struct ncclIbQpRtsAttr rtsAttr;

  // The index of the device on the remote side to which this QP is connected
  // to.
  int remDevIdx; // 对端 merged 设备下标；选 CTS.rkeys[remDevIdx] / remDevs[remDevIdx].rkey 用
  struct ncclIbWqeLatMon latMon;
};

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256,
              "request id are encoded in wr_id and we need up to 8 requests ids per completion");

// Structure to describe the completion records on the sender side.
struct ncclIbRemCompletionsRecords {
  // A "shadow" structure of the receiver's completion records in which the
  // sender tracks the completion records locally on its side. Sender uses this
  // memory to place the records it writes/reads to/from the receiver's
  // completion records.
  int elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  // Address in memory of the completion records structure on the receiver side.
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the sender chooses the
  // RKey (depending on the device being used) when it accesses the receiver's
  // completion records structure.
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
};

// A per-dev struct for netIbSendComm
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* ctsFifoMr;
  struct ibv_mr* putSignalScratchpadMr;
  struct ibv_mr* cmplsRecordsMr;
  struct ibv_sge sge;
};

// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Forward declaration
struct ncclIbResiliency;

// send / recv comm 的公共头（两者首字段，可互相 cast）。ncclIbGetNetCommDevBase 按 isSend 取各自 devs[i].base。
struct alignas(32) ncclIbNetCommBase {
  // ---- (a) 本地设备视图 ----
  ncclNetVDeviceProps_t vProps; // 本 merged 设备包含哪些物理口（devs[ndevs]）
  bool isSend;
  // ---- (b) 请求池与 QP 池 ----
  struct ncclIbRequest reqs[NET_IB_MAX_REQUESTS]; // 256 = 32 在途请求 × 8 路 multi-recv
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];           // qps[qpIndex]，qpIndex % ndevs 即所在设备
  // Array of pointers to the "actual" QPs that are used for data transfers.
  // The pointers point to QPs in the ncclIbNetCommBase::qps[] array.
  struct ncclIbQp* activeQps[NCCL_IB_MAX_QPS];    // 默认恒等映射；resiliency 故障切换时改指向
  uint64_t fifoHead;   // 本侧单调递增的槽游标：irecv / isend 成功各 ++；slot = fifoHead % 256，req->id = fifoHead
  int nqps;            // = max(local ndevs, remote ndevs) × NCCL_IB_QPS_PER_CONNECTION，两侧一致
  int splitDataOnQps;  // NCCL_IB_SPLIT_DATA_ON_QPS：1 则每请求跨所有 nqps 切片，0 则只用 nDataQps 个
  // ---- (c) 建链控制面 ----
  struct ncclSocket sock; // 建链 TCP；连上后只在 close 时用
  int ready;              // connect 侧 RTS 后置 1 并发给 accept 侧
  // Track necessary remDevInfo here
  int nRemDevs;           // 对端 merged 设备的物理口数，可与本地 ndevs 不同
  bool remOooRq;
  bool localOooRq;
  int recvMatchingScheme; // BY_INDEX（默认，wr_id=slot）/ BY_ID（imm=id；OOO RQ 或端口 failover 时强制）
  int nDataQps;           // = max(local ndevs, remote ndevs)：splitDataOnQps=0 时每请求用的 QP 数
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC]; // 对端每物理口的 lid/gid/mtu/rkey，metadata 里来
  // statistics about the comm
  struct ncclIbStats stats;
  struct ncclIbResiliency* resiliency;
  uint64_t speedChangeCounter;
  uint64_t totalSpeed;
  uint8_t weights[NCCL_IB_MAX_DEVS_PER_NIC];
  uint64_t devSpeeds[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Compute per-device LB weights (1-100); weight is never 0 since it is not considered a speed update but rather a port down.
static inline void ncclIbComputeLbWeights(struct ncclIbNetCommBase* base) {
  int ndevs = base->vProps.ndevs;
  // totalSpeed can not be 0: devices with inactive ports (speed 0) are
  // skipped at init, and speed-to-zero events are skipped in
  // ncclIbUpdateDeviceSpeed (port-failover handles those).
  uint8_t totalWeight = 0;
  for (int d = 0; d < ndevs; d++) {
    base->weights[d] = ncclParamIbEventBasedLb() ? (base->devSpeeds[d] * 100 / base->totalSpeed) : (100 / ndevs);
    totalWeight += base->weights[d];
  }
  if (totalWeight < 100) {
    base->weights[ndevs - 1] += (100 - totalWeight);
  }
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex);

static inline void ncclIbComputeDevSpeeds(struct ncclIbNetCommBase* base) {
  uint64_t totalSpeed = 0;
  for (int d = 0; d < base->vProps.ndevs; d++) {
    int ibDevN = ncclIbGetNetCommDevBase(base, d)->ibDevN;
    int remDevIdx = base->qps[d].remDevIdx;
    uint64_t localSpeed = COMPILER_ATOMIC_LOAD(&ncclIbDevs[ibDevN].currSpeed, std::memory_order_relaxed);
    uint64_t remoteSpeed = (base->remDevs[remDevIdx].currSpeed > 0) ? base->remDevs[remDevIdx].currSpeed : localSpeed;
    base->devSpeeds[d] = std::min(localSpeed, remoteSpeed);
    totalSpeed += base->devSpeeds[d];
  }
  base->totalSpeed = totalSpeed;
}

// qpIndex is the index relative to a device.
// For example, if a device has 2 QPs, qpIndex can be 0 or 1.
static inline ncclResult_t ncclIbCommBaseGetQpByIndex(struct ncclIbNetCommBase* commBase, int devIndex, int qpIndex,
                                                      ncclIbQp** qp) {
  if (devIndex < 0 || devIndex >= commBase->vProps.ndevs) {
    WARN("NET/IB: Invalid device index %d, expected [0, %d)", devIndex, commBase->vProps.ndevs);
    return ncclInternalError;
  }
  *qp = commBase->activeQps[commBase->vProps.ndevs * qpIndex + devIndex];
  return ncclSuccess;
}

// Each request is transferred over all devices, and depending on the
// "splitDataOnQps" configuration parameter, a request may be transferred over
// a single QP per device or on all QPs of each device.
static inline ncclResult_t ncclIbCommBaseGetNqpsPerRequest(struct ncclIbNetCommBase* baseComm, int* nQps) {
  if (nQps == NULL) {
    WARN("NET/IB: nQps output parameter is NULL");
    return ncclInternalError;
  }
  if (baseComm->nDataQps == -1) {
    WARN("NET/IB: nDataQps is not initialized");
    return ncclInternalError;
  }
  if (baseComm->nqps == -1) {
    WARN("NET/IB: nqps is not initialized");
    return ncclInternalError;
  }
  *nQps = (baseComm->splitDataOnQps == 1) ? baseComm->nqps : baseComm->nDataQps;
  return ncclSuccess;
}

// The function selects the QP to be used for the request. The QP selected
// based on the request ID and also based on the provided QP index. A request
// can be posted on multiple QPs. For example, if a request is posted on 4
// QPs, this function should be called 4 times, each time with a different
// qpIndex, ranging from 0 to 3.
// The function outputs the selected QP in the outQp argument and populates the
// outQpIndex argument with the index of the selected QP. Note that the
// outQpIndex is the index of the QP in the base::qps[] array.
static inline ncclResult_t ncclIbCommBaseGetQpForRequest(struct ncclIbNetCommBase* baseComm, const uint64_t id,
                                                         const uint8_t qpIndex, ncclIbQp** outQp, int* outQpIndex) {
  int nQps = 0;
  NCCLCHECK(ncclIbCommBaseGetNqpsPerRequest(baseComm, &nQps));
  *outQpIndex = (id * nQps + qpIndex) % baseComm->nqps;
  *outQp = baseComm->activeQps[*outQpIndex];
  if (*outQp == NULL) {
    WARN("NET/IB: QP is NULL for request id %lu, QP index %d", id, *outQpIndex);
    return ncclInternalError;
  }
  return ncclSuccess;
}

// Get a QP object from a QP number. If not NULL, qpIndex will also return the
// index of the QP in the ncclIbNetCommBase::qps[] array.
static inline ncclResult_t ncclIbCommBaseGetQpByQpNum(struct ncclIbNetCommBase* commBase, int devIndex, uint32_t qpNum,
                                                      ncclIbQp** qp, int* qpIndex) {
  if (devIndex < 0 || devIndex >= commBase->vProps.ndevs) {
    WARN("NET/IB: Invalid device index %d, expected [0, %d)", devIndex, commBase->vProps.ndevs);
    return ncclInternalError;
  }
  if (qp == NULL) {
    WARN("NET/IB: QP output pointer is NULL");
    return ncclInternalError;
  }
  TRACE(NCCL_NET, "NET/IB: %s: Looking for QP num %u on devIndex %d among %d QPs", __func__, qpNum, devIndex,
        commBase->nqps / commBase->vProps.ndevs);
  for (int qpIndexInDev = 0; qpIndexInDev < (commBase->nqps / commBase->vProps.ndevs); qpIndexInDev++) {
    *qp = &(commBase->qps[commBase->vProps.ndevs * qpIndexInDev + devIndex]);
    if ((*qp)->qp->qp_num == qpNum) {
      if (qpIndex != NULL) {
        *qpIndex = *qp - commBase->qps;
      }
      return ncclSuccess;
    }
  }
  *qp = NULL;
  return ncclInternalError;
}

static inline ncclResult_t ncclIbPostRecvWorkRequest(struct ibv_qp* qp, struct ibv_recv_wr* wr) {
  struct ibv_recv_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_recv(qp, wr, &bad_wr));
  return ncclSuccess;
}

// sender 侧 comm（ncclNet 的 sendComm）。数据面职责：等 CTS → 按权重切片 → 每 QP 一条 RDMA WRITE +
// 末尾一条 RDMA WRITE WITH IMM（唯一 signaled，所以每 QP 每请求恰好 1 个 CQE）。
struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // ---- (a) 被对端直写的表 + 预分配的 WR/SGE ----
  // Start with CTS FIFO and ibv structs as they have alignment restrictions

  // CTS FIFO from which the sender reads the Clear-to-Send (CTS) messages that
  // are written by the receiver (The receiver side writes into it upon
  // issuing a (multi-)receive request). Each row in the 2D array corresponds
  // to a single CTS message but can describe multiple recv-requests issued
  // on the receiver side.
  struct ncclIbSendFifo ctsFifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS]; // [slot][r]，receiver RDMA WRITE 直写
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];        // 每路一条 SGE，MultiSend 逐 QP 改 addr/length
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1]; // nreqs 条 WRITE + 1 条 WRITE_WITH_IMM，链表一次 post_send
  // ---- (b) 每物理口的 MR / CQ ----
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // ---- (c) 槽位 → 请求 的反查表（ncclIbTest 用 wr_id 低 8 bit 查这里）----
  // Array of pointers to store the send requests for faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array based on the "slot" they
  // are associated with.
  struct ncclIbRequest* sendReqs[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];

  // Counter per "slot" on how many send request were called for a multi-recv
  int sendReqsCnt[NET_IB_MAX_REQUESTS]; // isend 到达 ++，等于 nreqs 才真正 post；完成 --，归零才清槽复用
  struct ncclIbRemCompletionsRecords remCmplsRecords; // receiver cmplsRecords 的影子 + 远端 addr/rkeys
  int ar; // Use adaptive routing when all merged devices have it enabled；AR 且 size > NCCL_IB_AR_THRESHOLD 时数据与 IMM 拆两条 WR
  uint64_t putSignalScratchpad;

  struct ncclIbRemoteSpeedBuf remoteSpeedBuf;
  struct ibv_mr* remoteSpeedMr;
  uint64_t remoteSpeedCounter;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0,
              "ncclIbNetCommBase size must be 32-byte multiple to ensure ctsFifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, ctsFifo) % 32) == 0, "ncclIbSendComm ctsFifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

// GDR flush 资源：一个「连到自己」的 loopback RC QP，对刚收到的 GPU buffer 发 1 字节 RDMA READ 到 host，
// READ 的 CQE 到达即保证 NIC 先前对同一 PCIe 目标的写已落地、对 GPU 可见。首次 iflush 才懒创建。
struct ncclIbGpuFlush {
  struct ibv_mr* hostMr; // 指向 ncclIbRecvComm::gpuFlushHostMem 的 4 B host MR（READ 的落点）
  struct ibv_sge sge;    // length = 1
  struct ncclIbQp qp;
};

// This structure describes the FIFO which the receiver uses when it sends CTS
// messages to the sender.
struct ncclIbRemCtsFifo {
  // A "shadow" structure of the sender's CTS FIFO in which the receiver tracks
  // the CTS FIFO locally on its side. Receiver uses this memory to place the
  // CTS messages and populates the RDMA message "gather address" with the
  // memory of the CTS message that is sent.
  struct ncclIbSendFifo elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the receiver chooses the
  // RKey (depending on the device being used) when it posts a CTS to the
  // sender
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
};

struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
  // MR that is obtained after registering the "shadow" CTS FIFO on the
  // receiver's side. The LKey of this MR allows RDMA operations on the receiver
  // side to gather CTS messages (formatted by the receiver) and write them to
  // the sender's CTS FIFO.
  struct ibv_mr* ctsFifoMr;
  // MR that is obtained after registering the completion records on the
  // receiver side. The RKey of this MR is provided to the sender side, to allow
  // the sender side to access receiver's completion records using RDMA
  // operations.
  struct ibv_mr* cmplsRecordsMr;
  // SGE to avoid allocation of SGE structures on the stack when receiver
  // posts RDMA operations. The SGE is populated by the address of the memory
  // in which the CTS message formatted on the receiver is placed.
  struct ibv_sge sge;
  struct ibv_mr* speedUpdateMr;
};

#define NCCL_IB_RECV_WR_ID_DUMMY UINT64_MAX
#define NCCL_IB_SPEED_UPDATE_WR_ID (UINT64_MAX - 1)

// receiver 侧 comm（ncclNet 的 recvComm）。数据面职责：irecv 时每 QP 预投一条空 recv WR（接 IMM）+
// 用 RDMA WRITE 把 CTS 写进 sender 的 ctsFifo；数据到达由 IMM 的 CQE 得知；GDR 时再 iflush。
struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  // ---- (a) 每物理口的 MR / CQ / flush QP ----
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // ---- (b) 槽位 → 请求 的反查表（BY_INDEX 用 wr_id=slot，BY_ID 用 imm % 256 查这里）----
  // Array of pointers to store the recv requests to allow faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array using a hash (modulo) on
  // their ID.
  struct ncclIbRequest* recvReqs[NET_IB_MAX_REQUESTS];
  // ---- (c) 与 sender 互写的两张表 ----
  // Structure to hold all the related structures regarding the CTS FIFO
  // structure.
  struct ncclIbRemCtsFifo remCtsFifo; // sender ctsFifo 的影子（本地填好后作 RDMA WRITE 的 gather 源）+ 远端 addr/rkeys
  // Structure to hold all the completion records of all the outstanding
  // receive requests on the receiver side.
  struct ncclIbRequestCompletionRecord cmplsRecords[NET_IB_MAX_REQUESTS]; // [slot]，sender 多收时直写 sizes[]
  // ---- (d) GDR flush ----
  int gpuFlushHostMem;   // flush READ 的 host 落点
  int flushEnabled;      // nv_peermem 或 DMA-BUF 可用且未 NCCL_GDR_FLUSH_DISABLE
  int flushQpSl;         // loopback QP 的 sl/tc 沿用 connect 侧 metadata
  int flushQpTc;
  bool flushQpsCreated;  // 懒创建标志
  // ---- (e) recv WR 投递策略 ----
  bool prepostReceiveWorkRequests; // 默认 false：irecv 时逐 QP 投一条；OOO RQ / resiliency 时强制预投满并在 CQE 后补投
  // To avoid allocation and memset on the data-path a single structure is used
  // and only the wr_id is updated before posting a receive work request.
  struct ibv_recv_wr ibRecvWorkRequest; // num_sge = 0 的空 WR：只为接 WRITE_WITH_IMM 的 CQE

  uint64_t remSpeedBufAddr;
  struct ncclIbRemoteSpeedBuf speedUpdateBuf;
  uint16_t lastSentSpeeds[NCCL_IB_MAX_DEVS_PER_NIC];
  uint64_t lastSentCounter;
  bool postedSpeedUpdate;
};
static_assert((offsetof(struct ncclIbRecvComm, remCtsFifo) % 32) == 0,
              "ncclIbRecvComm ctsFifo must be 32-byte aligned");

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend);
ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm);
ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm);

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage* stage;
};

static inline void ncclIbCheckSpeedChanges(struct ncclIbSendComm* sendComm, struct ncclIbNetCommBase* base) {
  bool speedChanged = false;
  // Local speed change detection
  if (base->speedChangeCounter != COMPILER_ATOMIC_LOAD(&ncclIbSpeedChangeCounter, std::memory_order_acquire)) {
    base->speedChangeCounter = COMPILER_ATOMIC_LOAD(&ncclIbSpeedChangeCounter, std::memory_order_acquire);
    speedChanged = true;
  }
  // Remote speed change detection
  if (sendComm->remoteSpeedCounter !=
      COMPILER_ATOMIC_LOAD(&sendComm->remoteSpeedBuf.counter, std::memory_order_acquire)) {
    sendComm->remoteSpeedCounter = COMPILER_ATOMIC_LOAD(&sendComm->remoteSpeedBuf.counter, std::memory_order_acquire);
    for (int i = 0; i < base->nRemDevs; i++) {
      base->remDevs[i].currSpeed = (uint64_t)sendComm->remoteSpeedBuf.speedGbps[i] * 1000;
    }
    speedChanged = true;
  }
  if (speedChanged) {
    ncclIbComputeDevSpeeds(base);
    ncclIbComputeLbWeights(base);
  }
}

static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  COMPILER_ATOMIC_STORE(&stat->fatalErrorCount, 0, std::memory_order_relaxed);
  return ncclSuccess;
}
static void ncclIbStatsFatalError(struct ncclIbStats* stat) {
  COMPILER_ATOMIC_FETCH_ADD(&stat->fatalErrorCount, 1, std::memory_order_relaxed);
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}
ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName);

extern ncclProfilerCallback_t ncclProfilerFunction;

extern std::thread ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args);

ncclResult_t ncclIbGdrSupport();
ncclResult_t ncclIbPeerMemSupport();
ncclResult_t ncclIbDmaBufSupport(int dev);

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex);
ncclResult_t ncclIbGetGidIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                               int* gidIndex);
ncclResult_t ncclIbGetPkeyIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                int* pkeyIndex);
ncclResult_t ncclIbGidInfoQuery(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                struct ncclIbGidInfo* gidInfo);
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req);
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r);

ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mrFlags, void** mhandle);

int ncclIbGetTrafficClass(void* ctx);
void ncclIbSetTrafficClass(void* ctx, int trafficClass);

// Net IB plugin entry functions.
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction,
                        ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbDevices(int* ndev);
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);
ncclResult_t ncclIbConnectImpl(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                               ncclNetDeviceHandle_t** sendDevComm, int nQpsPerDev, int envTrafficClass);
ncclResult_t ncclIbConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                           ncclNetDeviceHandle_t** sendDevComm);
ncclResult_t ncclIbAcceptImpl(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm, int nQpsPerDev);
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm);
ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle);
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                         void** request);
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                         void** request);
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t ncclIbCreateFlushQp(struct ncclIbRecvComm* comm);
ncclResult_t ncclIbTest(void* request, int* done, int* sizes);
ncclResult_t ncclIbCloseSend(void* sendComm);
ncclResult_t ncclIbCloseRecv(void* recvComm);
ncclResult_t ncclIbCloseListen(void* listenComm);
ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props);
ncclResult_t ncclIbFinalizeDevices(void);
ncclResult_t ncclIbFinalize(void* ctx);
ncclResult_t ncclIbSetNetAttr(void* ctx, ncclNetAttr_t* netAttr);

static inline void printIbWcStatusHint(int status) {
  switch (status) {
  case IBV_WC_LOC_PROT_ERR:
    INFO(NCCL_NET,
         "HINT: In many cases this error indicates that ACS is enabled, which breaks the GPU Direct RDMA protocol.");
    INFO(NCCL_NET, "HINT: To confirm, set NCCL_NET_GDR_LEVEL=0; if that resolves it, "
                   "disable ACS following your vendor documentation.");
    return;
  case IBV_WC_WR_FLUSH_ERR:
    INFO(NCCL_NET, "HINT: In many cases this error indicates that NICs on the same node cannot talk to each other.");
    INFO(NCCL_NET, "HINT: To confirm, use a lower level tool like ib_write_bw to communicate across NICs on the same "
                   "node.");
    return;
  case IBV_WC_RETRY_EXC_ERR:
    INFO(NCCL_NET, "HINT: In many cases this error indicates that NCCL_IB_TIMEOUT is set too short (the default value "
                   "is 20, which is ~30 seconds before timing out).");
    INFO(NCCL_NET, "HINT: To confirm, increase NCCL_IB_TIMEOUT (see "
                   "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-timeout).");
    return;
  default:
    break;
  }
}

// GID Format
// global:  |              64b  - subnet-prefix                |                 64b - EUI                          |
// raw   :  | 10b fixed | 22b 0 | 16b FLID | 16b subnet-prefix |                 64b - EUI                          |
static uint16_t ncclIbExtractLocalSubnetPrefix(uint64_t subnet_prefix) {
  return (be64toh(subnet_prefix) & 0xffff);
}

static int ncclIbExtractFlid(union ibv_gid* gid) {
  return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
}

#endif
