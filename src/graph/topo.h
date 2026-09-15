/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"
#include "xml.h"
#include "net.h"
#include "os.h"

#define LOC_BW 5000.0
#define MLOPART_LOC_BW 2618.0
#define SM60_NVLINK_BW 18.0
#define SM70_NVLINK_BW 20.0
#define SM80_NVLINK_BW 20.0
#define SM90_NVLINK_BW 20.6
#define SM86_NVLINK_BW 12.0
#define SM100_NVLINK_BW 40.1
#define PCI_BW 12.0           // PCI Gen3 x16
#define AMD_ZEN12_BW 16.0
#define AMD_ZEN34_BW 24.0
#define AMD_ZEN5_BW 32.0
#define BDW_QPI_BW 6.0
#define SKL_QPI_BW 10.0
#define SRP_QPI_BW 22.0
#define ERP_QPI_BW 40.0
#define ZPI_BW 6.0
#define YONGFENG_ZPI_BW 9.0
#define P9_BW 32.0
#define ARM_BW 6.0
#define NET_BW 12.0           // 100Gbit

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth.
#define INTEL_P2P_OVERHEAD(bw) (bw * 6 / 5)

#define NCCL_TOPO_NODE_TYPES 10
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
#define GIN 6
#define RMA 7
#define DEV 8
#define CXB 9 // C2C Cross-Bridge: shared C2C bus node for GPUs split with mlopart
extern const char* topoNodeTypeStr[];
/*
  LINK_LOC   GPU 与自己的 DEV
  LINK_NVL   NVLink
  LINK_C2C   GPU/DEV 到 CPU 的 C2C
  LINK_PCI   PCIe
  LINK_SYS   NUMA/CPU 间互联
  LINK_NET   NIC 到 Network Plugin endpoint
*/
// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_NVL 1
// Skipping 2 for PATH_NVB
#define LINK_C2C 3
#define LINK_PCI 4
// Skipping 5 for PATH_PXB
// Skipping 6 for PATH_PXN
// Skipping 7 for PATH_P2C
// Skipping 8 for PATH_PHB
#define LINK_SYS 9
#define LINK_NET 10
extern const char* topoLinkTypeStr[];

extern int64_t ncclParamPxnC2c();
// 节点
struct ncclTopoNode;
// 边：一条「有向」物理出边，存在源节点的 links[] 里。
// - 有向：ncclTopoConnectNodes(a, b) 只加 a→b，反向需再调一次（建图代码总是成对调用）。
// - 可聚合：同一对节点间同 type 的边只存一条，bw 累加（XML <nvlink count=N> → 一条 bw = N × 单链带宽）。
// - bw 来自 topo.h 顶部常量表（查表值，不是测量值），是后续路径/搜索的统一度量单位。
struct ncclTopoLink {
  int type;                     // LINK_*（LOC / NVL / C2C / PCI / SYS / NET），与 PATH_* 数值尽量对齐
  float bw;                     // GB/s，聚合后的总带宽
  struct ncclTopoNode* remNode; // 对端节点
};
// Allows for up to 32 NICs per node on GB200-NVL72
#define NCCL_TOPO_MAX_LINKS 576
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES * NCCL_TOPO_NODE_TYPES)

// 路径：从某节点出发到某目标的一条完整路线（ncclTopoNode::paths[t][i] 的元素）。
// 由 paths.cc · ncclTopoSetPaths 以目标为源做 BFS 反向填出：list[0] 是本节点自己的出边，
// 依次到目标。更新优先级：type 更小 > 同 type 下 bw 更大 > 同 type 同 bw 下 count 更少。
// 之后 ncclTopoComputePaths 还会按策略覆盖（P2P 禁用 / 无 GDR 绕 CPU、PXN、不可达置 PATH_NET），
// 所以这里的 type 已经是「NCCL 允许怎么走」，Transport 选型直接查它。
struct ncclTopoLinkList {
  struct ncclTopoLink** list; // 指向沿途各节点 links[] 里的真实边，不拷贝
  int count;     // Number of links stored in list. 跳数
  int capacity;  // Number of entries allocated for list.
  float bw; // 全程瓶颈带宽 = 沿途各边 bw 取 min
  int type; // 路线等级 PATH_*（沿途各跳取 max，PCI→PCI 记 PXB、经 CPU 记 PHB、经 DEV 中转记 NVB）
};

#define NCCL_TOPO_UNDEF (-1)

#define NCCL_TOPO_ID_LOCAL_ID_MASK 0x00ffffffffffffff
#define NCCL_TOPO_ID_SYSTEM_ID(id) (id >> 56) // → 高 8 bit，多机时区分节点
#define NCCL_TOPO_ID_LOCAL_ID(id) (id & NCCL_TOPO_ID_LOCAL_ID_MASK) // → 低 56 bit，本机 busId / dev 号
#define NCCL_TOPO_LOCAL_NIC_ID(numaid, busid) (((int64_t)numaid << 56) + busid)
/*
    63                    56 55                              0
    ├──── systemId (8b) ────┼──────── localId (56b) ──────────┤
*/
#define NCCL_TOPO_ID(systemid, localid) (((int64_t)systemid << 56) + (localid & NCCL_TOPO_ID_LOCAL_ID_MASK))
#define NCCL_TOPO_GPU_LOCAL_RANK_SHIFT 40
#define NCCL_TOPO_GPU_LOCAL_ID(busId, localRankOnDev) \
  ((((uint64_t)(localRankOnDev)) << 40) | ((busId) & ((((uint64_t)1) << 40) - 1)))
#define NCCL_TOPO_MLOPART_MASK (0x3) // lower 2 bits: bit[0]=enabled, bit[1]=partition index
#define NCCL_TOPO_MLOPART_DEV_MAX (2) // max DEV nodes per physical GPU (one per uGPU partition)
/*
    MLOPart busId 低位编码：
      bit[1:0] = {partition_index, enabled_flag}
*/
#define NCCL_TOPO_MLOPART(mloPart) ((((int64_t)(mloPart) << 1) | 0x1) & NCCL_TOPO_MLOPART_MASK)
#define NCCL_TOPO_MLOPART_BUSID(busId, mloPart) \
  ((mloPart) != NCCL_TOPO_UNDEF ? ((busId) | NCCL_TOPO_MLOPART(mloPart)) : (busId))

// 拓扑图节点。字段分三组：
//   [身份]   type / id / union{...}      建图时由 XML 填入，之后只读
//   [Layer-0] links[] / nlinks           物理出边邻接表，建图时填
//   [Layer-1] paths[t][]                 到「类型 t 的每个节点」的预计算路线，ncclTopoComputePaths 填，
//                                        Trim 删节点后必须 RemovePaths 再重算（i 是桶内下标，会挪动）
//   [搜索]   used                        ncclTopoCompute 回溯时的访问标记
// 注意 GPU 与 DEV 是两个节点：GPU 是 rank 视角的逻辑节点（带 gpu.rank），DEV 是物理芯片，
// 二者用 LINK_LOC 互连，NVLink / C2C 挂在 DEV 上（v2.31 为 MLOPart 分区引入）。
struct ncclTopoNode {
  int type;   // GPU / PCI / NVS / CPU / NIC / NET / GIN / RMA / DEV / CXB
  int64_t id; // NCCL_TOPO_ID(systemId, localId)：高 8 bit 区分主机，低 56 bit 为 busId / dev 号
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;                    // 该 GPU 对应的 comm rank；paths/搜索结果里的 rank 号都由此而来
      int cudaCompCap;
      int gdrSupport;
      int mloPart; // MLOPart partition index, or NCCL_TOPO_UNDEF if not MLOPart
      struct ncclTopoNode* parent; // parent DEV node
    } gpu;
    struct {
      uint64_t device;  // Same as pci.device, a combination of vendor, device, subsystem_vendor and subsystem_device
      int dev; // NVML dev number
      int cudaCompCap;
      int nGpus; // number of GPU partitions attached to this DEV node
    } dev;
    struct {
      int dev; // Plugin dev number
      uint64_t vendor; // PCI vendor ID
      uint64_t device; // PCI device ID
      uint64_t pciId;
      uint64_t asic;
      int port;
      float bw;
      float latency;
      int gdrSupport;
      int collSupport;
      int maxChannels;
      int localGpu;
      int16_t railId;
      int16_t planeId;
    } net;
    struct {
      int arch;
      int vendor;
      int model;
      ncclAffinity affinity;
    } cpu;
    struct {
      uint64_t device;
    } pci;
  };
  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];  // 邻接表，表示边
  // Pre-computed paths to GPUs and NICs   从当前这个 node 出发，到“类型为 t 的第 j 个节点”的路径。
  // 按目标类型分组的寻址表」——paths[t][j] = 从本节点到 nodes[t].nodes[j] 的预计算最优路径（hop 链 + 瓶颈带宽 + PATH 等级）。不是物理边，是算法用的路由缓存。
  // paths[NET][2] │ 从我家到「第 3 个 NET 站点」的路线
  /*  稀疏分配
      paths[NET][n].list  = [link₀*, link₁*, link₂*]
                                │         │         │
                                ▼         ▼         ▼
                              真实存在于某个 node->links[] 里的 ncclTopoLink
                              link->remNode 指向下一个节点
  */

  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

// 某一类型节点的「桶」。paths[t][i] 与 graph->intra 里的 GPU 下标都指这里的 nodes[i]，
// 因此 ncclTopoRemoveNode 会 memmove 挪动后续元素——这正是 Trim 后必须重算 paths 的原因。
struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

/*
  一张总图

     1 │ncclTopoSystem
     2 │└─ nodes[type]           ← 桶
     3 │   └─ ncclTopoNode
     4 │      ├─ links[]         ← Layer-0：物理出边邻接表
     5 │      └─ paths[type][]   ← Layer-1：到某类每个节点的预计算最优路由
     6 │           └─ { type, bw, count, list[] }
*/
struct ncclTopoSystem {
  int systemId;
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES];
  int nHosts;
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES]; // 按类型桶划分
  float maxBw;
  float totalBw;
  int inter;
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
// Removing a node invalidates computed paths. Callers must remove any paths before calling this
// function and recompute them before using the topology for path-dependent operations.
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);
void ncclTopoRemovePaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank);
ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min);
ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max);
ncclResult_t ncclTopoSplitNvLink(struct ncclTopoSystem* system, int* splitNvLink);

enum {
  NCCL_NET_MERGE_POLICY_ALL = 0,
  NCCL_NET_MERGE_POLICY_RAIL = 1
};

struct ncclTopoNetInfo {
  bool coll;
  bool gin;
  bool rma;
  bool net;
  // communicator-specific information
  int netPluginIndex;
  int maxDevsPerNic;
  bool dmaBufSupport;
  // NIC fusion
  int mergeLevel;
  int mergePolicy;
  const char* forceMerge;
  // dev count tracking functions (not part of ncclNet)
  ncclResult_t (*getDevCount)(int, int*, int*);
  ncclResult_t (*setVirtDevCount)(int, int);
  // ncclNet API functions
  const char* name;
  ncclResult_t (*getProperties)(int, ncclNetProperties_t*);
  ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*);
  ncclResult_t (*devices)(int*);
};

ncclResult_t ncclTopoProcessNet(ncclXml* xml, const char* dumpXmlFile, struct ncclTopoNetInfo* net);
ncclResult_t ncclTopoGetFusionEnv(int* mergeLevel, const char** forceMerge);

#define NCCL_TOPO_XML_MAX_NODES 256
#define NCCL_GRAPH_XML_MAX_NODES 65536
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, uint64_t localHostHash);
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode* xmlGraphs, struct ncclTopoSystem* system,
                                     struct ncclTopoGraph* graph, int* nChannels);
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system,
                                      struct ncclXml* xml);

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax);
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw);

static ncclResult_t ncclTopoIdToIndex(struct ncclTopoSystem* system, int type, int64_t id, int* index) {
  *index = -1;
  for (int i = 0; i < system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

static ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index, bool showWarn) {
  *index = -1;
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].gpu.rank == rank) {
      *index = i;
      return ncclSuccess;
    }
  }
  if (showWarn) WARN("ncclTopoRankToIndex could not find rank %d", rank);
  return ncclInternalError;
}

static ncclResult_t ncclTopoDevToRank(struct ncclTopoSystem* system, int systemId, int dev, bool warn, int* rank) {
  *rank = -1;
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    // Only consider GPUs on the given node
    if (NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id) != systemId) continue;
    if (system->nodes[GPU].nodes[i].gpu.dev == dev) {
      *rank = system->nodes[GPU].nodes[i].gpu.rank;
      return ncclSuccess;
    }
  }
  if (warn) WARN("ncclTopoDevToRank could not find rank for nvml dev %d in systemId %d", dev, systemId);
  return ncclInternalError;
}

extern struct kvDict nicPathKvList[];

static ncclResult_t ncclTopoIdToNetDev(struct ncclTopoSystem* system, int64_t id, int* netDev) {
  *netDev = -1;
  for (int i = 0; i < system->nodes[NET].count; i++) {
    if (system->nodes[NET].nodes[i].id == id) {
      *netDev = system->nodes[NET].nodes[i].net.dev;
      return ncclSuccess;
    }
  }
  WARN("Could not find NET with id %lx", id);
  return ncclInternalError;
}

// Returns NVLink bw in GB/s
static float ncclTopoNVLinkBw(int cudaCompCap) {
  return cudaCompCap >= 100 ? SM100_NVLINK_BW :
         cudaCompCap >= 90  ? SM90_NVLINK_BW :
         cudaCompCap == 86  ? SM86_NVLINK_BW :
         cudaCompCap >= 80  ? SM80_NVLINK_BW :
         cudaCompCap >= 70  ? SM70_NVLINK_BW :
         cudaCompCap >= 60  ? SM60_NVLINK_BW :
                              SM80_NVLINK_BW;
}

// Mirror bits
static bool isPow2(int val) {
  return (val & (val - 1)) == 0;
}
static int mirrorBits(int val, int pow2) {
  int mirror = 0;
  for (int b = 1, mb = (pow2 >> 1); b < pow2; b <<= 1, mb >>= 1) {
    if (val & b) mirror |= mb;
  }
  return mirror;
}
#endif
