/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "device.h"
#include "os.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include "gdrwrap.h"

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile = NULL);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm);
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks);
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink);
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected);
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm);

// Query topology
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId,
                               int peerRank, int64_t* id, int* dev, int* proxyRank);
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p,
                              int* read, int* intermediateRank, int* cudaP2p, int* isCrossClique = nullptr);
ncclResult_t ncclTopoCheckMNNVL(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2,
                                int* ret);
enum ncclTopoGdrMode {
  ncclTopoGdrModeDisable = 0,
  ncclTopoGdrModeDefault = 1,
  ncclTopoGdrModePci = 2,
  ncclTopoGdrModeNum = 3
};
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int rank, int64_t netId, int read,
                              enum ncclTopoGdrMode* gdrMode);

enum ncclTopoFlushType {
  ncclTopoFlushNone = 0,   // no flush needed
  ncclTopoFlushAlways = 1, // flush always needed
  ncclTopoFlushC2c = 2     // PCIe NIC and C2C sync path are unordered, flush is needed.
};
static inline uint32_t ncclGdcPinFlag(enum ncclTopoFlushType flush) {
  if (flush == ncclTopoFlushC2c && ncclGdrPinV2Available()) return GDR_PIN_FLAG_FORCE_PCIE;
  return GDR_PIN_FLAG_DEFAULT;
}
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank,
                               enum ncclTopoFlushType* flush);
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw);
ncclResult_t ncclTopoIsGdrAvail(struct ncclTopoSystem* system, int rank, bool* avail);
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net);
int ncclPxnDisable(struct ncclComm* comm);
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks);
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu);

ncclResult_t ncclGetUserP2pLevel(int* level);

// Find CPU affinity
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity);

#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_ARCH_ARM 3
#define NCCL_TOPO_CPU_ARCH_MIXED 4
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define NCCL_TOPO_CPU_VENDOR_MIXED 4
#define NCCL_TOPO_CPU_MODEL_INTEL_BDW 1
#define NCCL_TOPO_CPU_MODEL_INTEL_SKL 2
#define NCCL_TOPO_CPU_MODEL_INTEL_SRP 3
#define NCCL_TOPO_CPU_MODEL_INTEL_ERP 4
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN12 1
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN34 2
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN5 3
#define NCCL_TOPO_CPU_MODEL_YONGFENG 1
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* comm, int* localGinDevs, int* localGinCount);
ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm* comm, int* localRmaDevs, int* localRmaCount);
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex);
ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw);

enum netDevsPolicy {
  NETDEVS_POLICY_AUTO = 0x0,
  NETDEVS_POLICY_ALL = 0x1,
  NETDEVS_POLICY_MAX = 0x2,
  NETDEVS_POLICY_UNDEF = 0xffffffff
};
ncclResult_t ncclTopoGetNetDevsPolicy(enum netDevsPolicy* policy, int* policyNum);

// Allows for up to 576 GPUs (e.g., NVLD144) with headroom for internal operations
#define NCCL_TOPO_MAX_NODES 640
ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                              int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType);
ncclResult_t ncclTopoGetDevNodes(struct ncclTopoSystem* system, int64_t baseId, struct ncclTopoNode** nodes,
                                 int* nNodes);
/*
  PATH_LOC  本地自身
  PATH_NVL  直接 NVLink/NVSwitch
  PATH_NVB  经过中间 GPU 的 NVLink
  PATH_C2C  C2C
  PATH_PIX  至多经过一个 PCI switch
  PATH_PXB  经过多个 PCI switch
  PATH_P2C  GPU 经 C2C 到 CPU，再经 PCIe 到 NIC
  PATH_PXN  经中间 GPU 到 NIC
  PATH_PHB  经过 PCI Host Bridge/CPU
  PATH_SYS  跨 NUMA
  PATH_NET  跨网络
  PATH_DIS  不可达
*/
// Local (myself)
#define PATH_LOC 0

// Connection traversing NVLink
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2

// Connection through C2C
#define PATH_C2C 3

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 4

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#define PATH_PXB 5

// Connection between a GPU and a NIC using the C2C connection to the CPU and the PCIe connection to the NIC
#define PATH_P2C 6

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network
// send/recv operations.
#define PATH_PXN 7

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 8

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#define PATH_SYS 9

// Connection through the network
#define PATH_NET 10

// New type of path which should precede PATH_PIX
#define PATH_PORT PATH_NVL

// Disconnected
#define PATH_DIS 11
extern const char* topoPathTypeStr[];

// Init search. Needs to be done before calling ncclTopoCompute
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

#define NCCL_TOPO_PATTERN_BALANCED_TREE \
  1   // Spread NIC traffic between two GPUs (Tree parent + one child on first
                                            // GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE \
  2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree
                                            // children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct

/*
一句话定义

  ncclTopoGraph 是一次拓扑搜索的请求单 + 结果单：ncclTopoCompute 拿着 Input 字段当约束，在本节点的 ncclTopoSystem 上搜出一组"GPU 怎么串成链 + 从哪张网卡进出"的方案，写回 Output 字段。

  大白话类比

  像给旅行社下的行程委托单：

  • 上半部分是你的要求：走环线还是树形（pattern）、最少几条线路（minChannels）、能不能从不同机场进出（crossNic）
  • 下半部分是旅行社填回来的方案：实际排了几条线（nChannels）、每条线的途经城市顺序（intra）、进出机场（inter）、这条线能跑多快（bwIntra/bwInter）

  关键在于同一张单子既是输入又是输出，而且旅行社在报不出价时会自己放宽你的要求（降带宽、放松路径类型），然后把放宽后的实际值填回单子——所以你交出去的约束和拿回来的值可能不一样。

  物理逻辑

  四个层次：

     1 │ Input 约束                          搜索                        Output 解
     2 │ ┌──────────────┐         ┌────────────────────────┐      ┌──────────────────┐
     3 │ │ pattern      │────────▶│ ncclTopoSearchRec      │─────▶│ nChannels        │
     4 │ │ min/maxChan  │         │  (DFS + 回溯 + 超时)    │      │ intra[] inter[]  │
     5 │ │ crossNic     │         │                        │      │ bw/type/nHops    │
     6 │ │ collNet      │         │ 沿 path 扣带宽，不够就   │      └──────────────────┘
     7 │ └──────────────┘         │ rewind（followPath）    │
     8 │                          └────────────────────────┘
     9 │                                    │ 搜不到就放宽，逐级降级：
    10 │                                    ▼
    11 │    sameChannels 1→0 ▶ pattern 简化 ▶ typeIntra++ ▶ typeInter++ ▶ crossNic ▶ bw 降档

  intra / inter 的内存布局是这个结构最容易看错的地方：

     1 │        channel 0                channel 1               ...
     2 │intra:  [r0 r1 r2 r3 ... r_ngpus-1][r0 r1 r2 ...        ]   stride = ngpus(本节点GPU数)
     3 │         └── 本节点内的 GPU 串行顺序，存 rank 号 ──┘
     4 │
     5 │inter:  [ netIn  netOut ][ netIn  netOut ]              ...   stride = 2
     6 │          进            出
     7 │        NET node id = (systemId << ...) | localId  ← 所以是 int64_t 而非 int

  三个必须抓住的物理约束：

  这是节点局部视角，不是全局视图。 intra 的 stride 是本节点 GPU 数（connect.cc 用 c * localRanks，search.cc 用 c * ngpus，两者必须相等），数组开到 NCCL_TOPO_MAX_NODES(640) 只是静态上界。全局的环要靠
  ncclTopoPostset 把各节点的段首尾相接才拼出来。

  搜索是"扣带宽 + 回溯"，不是打分排序。 followPath 沿路径把 bwIntra/bwInter 从每条 link 的余量里真实扣掉，扣不动就 rewind 退回去。所以 bwIntra 既是约束又是结果——它是"我要求每 channel
  有这么多带宽"，搜索成功后就成了"每 channel 实际有这么多带宽"。

  降级顺序编码了性能偏好。 先放宽 sameChannels（结构对称性最便宜），再简化 pattern，再放宽路径类型（typeIntra 先于 typeInter，因为节点内绕远比跨节点绕远代价低），最后才降带宽档位。而 pass 2
  会反向尝试提升带宽——先找到可行解，再在可行解附近爬坡。

  跨节点必须归一化。 每个节点独立搜索，结果可能不同；init.cc 在 allgather 后对所有 rank 取保守值：带宽和 nChannels 取 min，路径类型取 max（越大越松），保证全 communicator 的 tuning 模型一致。
*/

// 某一种算法在「本节点」拓扑上的搜索请求 + 搜索结果。
// ncclTopoCompute 以 Input 字段为约束在 ncclTopoSystem 上做带回溯的搜索，把最优解写回
// Output 字段；一个 communicator 为 ring/tree/collnet/nvls 各算一份，存在 comm->graphs[]。
// 注意这里描述的是「本节点内 GPU 怎么串 + 从哪张网卡进出」，跨节点的 rank 拼接由
// ncclTopoPreset/ncclTopoPostset 完成。
struct ncclTopoGraph {
  // Input / output
  // 搜索标签，同时用于匹配 NCCL_GRAPH_FILE 里的 <graph id=...>。
  // 注意它不等于 comm->graphs[] 的下标（NCCL_ALGO_*），别拿来当索引用。
  int id; // ring : 0, tree : 1, collnet : 2, nvls : 3, collnetDirect : 4
  int pattern;     // NCCL_TOPO_PATTERN_*，决定 NIC 流量怎么摊到 GPU 上；搜索会降级改写它
  int crossNic;    // 入口/出口是否允许用不同网卡（0/1/2，由 NCCL_CROSS_NIC 与 pattern 共同决定）
  int collNet;     // 纯输入：是否为 collnet 图，影响搜索时对 NIC 的处理
  int minChannels; // 可接受的 channel 数下界，某些 pattern 会被改写（如 NVLS 强制拉满）
  int maxChannels; // 上界，同上；ring 用 MAXCHANNELS/2 留一半给 tree
  // Output
  int nChannels;      // 实际搜到的 channel 数，0 表示该算法在本拓扑上不可用
  float bwIntra;      // 单 channel 节点内可用带宽 (GB/s)，搜索时按候选速度档逐级下调
  float bwInter;      // 单 channel 跨节点可用带宽 (GB/s)
  float latencyInter; // 出口网卡的固有延迟，供 tuning 模型算 interLat
  int typeIntra;      // 节点内允许的最差路径类型 (PATH_*)，越大越松
  int typeInter;      // GPU 到 NIC 允许的最差路径类型 (PATH_*)
  int sameChannels;   // 1 = 所有 channel 用同一条 GPU 序列；放开为 0 可换更高带宽
  int nHops;          // 解的总跳数，带宽相同时作为次级择优指标
  // channel c 的节点内 GPU 序列：intra[c * ngpus + i] 存 rank 号，ngpus 为本节点 GPU 数
  // （即 localRanks）。数组按 NCCL_TOPO_MAX_NODES 开满，实际只用前 nChannels * ngpus 项。
  int intra[MAXCHANNELS * NCCL_TOPO_MAX_NODES];
  // channel c 的入口/出口网卡：inter[c*2+0] 进、inter[c*2+1] 出，存的是 NET node id
  // （高位 systemId + 低位 localId，故为 int64_t），-1 表示无网卡可用
  int64_t inter[MAXCHANNELS * 2];
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);

struct ncclTopoRanks {
  int crossNicRing;
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
  int nvlsHeads[MAXCHANNELS];
  int nvlsHeadNum;
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
                             struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs,
                             struct ncclComm* parent);

ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm);
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs);
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes,
                                 int numPipeOps, float* time);

#endif
