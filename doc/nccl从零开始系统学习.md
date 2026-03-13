---

## type: 技术框架
title: NCCL 从零开始系统学习
tags: [NCCL, 系统学习, 通信框架, AI训练, RDMA, GPU通信, 网络架构]
领域: [系统底层开发, NCCL, 网络知识, 代码阅读]
状态: 进行中
创建日期: 2026-02-25

> **定位**：一份面向 RDMA 驱动/网络架构开发者的 NCCL 深度学习笔记。
> 全文以「**NCCL 如何使用和管理网络**」为主线，每个概念从已有的 RDMA 知识出发。
>
> **核心方法论**：把 NCCL 看成一个**拓扑感知的、运行在 GPU 上的高性能消息路由器**。
> 算法是路由策略，Channel 是虚电路，ncclNet 是数据链路层，硬件是物理层。
>
> **学习路线**：认知校准 → 网络建模 → 协议栈拆解 → 传输核心 → 关键网络技术 → 性能工程 → 实战

---

# 第一章：NCCL 是什么 — 网络架构师的视角

## 1.1 一句话定义

**NCCL（NVIDIA Collective Communications Library）** 是 NVIDIA 开发的多 GPU 集合通信库。

**网络架构师视角**：NCCL 是一个**拓扑感知的异构网络消息路由器**——它探测底层网络拓扑（NVLink / PCIe / RDMA / TCP），在异构链路上自动选择最优路由策略（Ring / Tree / CollNet），通过持久化 GPU 线程驱动数据搬运，最终在多 GPU 间完成集合通信。

## 1.2 NCCL 为什么值得网络人深入


| 维度                    | 解释                                          |
| --------------------- | ------------------------------------------- |
| **它是网络子系统的重度用户**      | 大模型训练中 60%+ 的集群网络带宽被 NCCL 消耗                |
| **它重新发明了很多网络概念**      | 流控、多路径、拥塞控制、QoS——只不过搬到了 GPU 显存上             |
| **它是 RDMA 知识的最佳落地场景** | ibv_post_send / ibv_reg_mr / CQ 轮询、GDR，全部用到 |
| **网络插件是开放接口**         | ncclNet 插件接口允许你写自定义传输后端——这是 RDMA 开发者的主战场    |


## 1.3 NCCL 的「网络世界观」

NCCL 的核心设计哲学是**异构网络感知**，它把所有可能的 GPU 间通信路径看成一张带权图：

```
节点内:
  GPU ──NVLink (600GB/s)──→ GPU        权重 = 最高
  GPU ──PCIe Switch (32GB/s)──→ GPU     权重 = 中
  GPU ──PCIe → CPU → PCIe (16GB/s)──→ GPU  权重 = 低

节点间:
  GPU ──[GDR]→ NIC ──RDMA (25-50GB/s)──→ NIC ──[GDR]→ GPU  权重 = 中
  GPU → Host → NIC ──TCP──→ NIC → Host → GPU             权重 = 最低
```

> **架构师洞察**：MPI 把网络看成均质的，所以 MPI 的拓扑优化是"锦上添花"。
> NCCL 把网络看成异构的，拓扑优化是"生死攸关"——NVLink 和 TCP 差 1000 倍，选错路径就是灾难。

## 1.4 在 AI 训练栈中的位置

```
┌─────────────────────────────────────┐
│  PyTorch / DeepSpeed / Megatron-LM  │  ← 框架层（你不需要关心）
├─────────────────────────────────────┤
│  torch.distributed / c10d           │  ← 分布式封装（薄 wrapper）
├─────────────────────────────────────┤
│  NCCL  (ncclAllReduce / ...)        │  ← 集合通信引擎 ★ 本文主角
│    ├─ Algorithm: Ring / Tree        │
│    ├─ Protocol: Simple / LL / LL128 │
│    └─ Transport: P2P / SHM / NET   │
├─────────────────────────────────────┤
│  ncclNet Plugin Interface           │  ← 网络抽象层 ★ 你的主战场
│    ├─ 内置: IB (net_ib.cc)          │
│    ├─ 插件: aws-ofi-nccl (EFA)     │
│    └─ 插件: 你自己写的              │
├─────────────────────────────────────┤
│  Hardware:                          │
│    NVSwitch | PCIe Switch | IB NIC  │  ← 物理层
└─────────────────────────────────────┘
```

---

# 第二章：核心概念

## 2.1 对象模型


| 概念               | 定义                        | 网络类比                          | 深层理解                                             |
| ---------------- | ------------------------- | ----------------------------- | ------------------------------------------------ |
| **ncclComm**     | 通信域，管理一组 GPU 的通信上下文       | 类似 RDMA PD（Protection Domain） | 持有所有 Channel、Connection、Proxy 的生命周期              |
| **Rank**         | GPU 在通信域中的编号 (0 ~ N-1)    | 类似节点 IP                       | 一个进程可持有多个 Rank（多 GPU）                            |
| **Channel**      | 通信并行度的基本单位                | **虚电路**                       | 每个 Channel 有独立的 Ring/Tree 路径、独立的 Buffer、独立的 QP 对 |
| **Connection**   | 单个 Channel 内，与某个 Peer 的连接 | 一对 RDMA QP                    | 持有 Send/Recv Buffer、MR、QP 等资源                    |
| **Proxy Thread** | CPU 代理线程，替 GPU 发起网络 I/O   | 用户态 RDMA 轮询线程                 | 必须存在：GPU 线程无法调用 ibv_post_send                    |


> **架构师洞察**：Channel 是理解 NCCL 性能的关键。更多的 Channel = 更多的并行网络连接 = 更高的聚合带宽。
> 但 Channel 也消耗 GPU SM 和显存。这是一个典型的**并行度 vs 资源**的 trade-off。
> 默认情况下 NCCL 根据拓扑自动决定 Channel 数（通常 2~16）。

## 2.2 集合通信原语


| 原语                | 网络行为                                 | 典型用途                   |
| ----------------- | ------------------------------------ | ---------------------- |
| **AllReduce**     | 所有 Rank 的数据全局归约 + 广播，每个 Rank 都拿到完整结果 | 梯度同步（最高频）              |
| **ReduceScatter** | 归约 + 分散，每个 Rank 只拿 1/N 结果            | ZeRO 优化器               |
| **AllGather**     | 每个 Rank 广播自己的片段，所有 Rank 拼出完整数据       | ZeRO 参数重组              |
| **Broadcast**     | Root → 所有 Rank                       | 参数初始化                  |
| **Reduce**        | 所有 Rank → Root                       | 汇总统计量                  |
| **Send/Recv**     | 点对点传输                                | Pipeline 并行的跨 Stage 通信 |


> **关键认知**：现代大模型训练（如 Megatron-LM）中，ReduceScatter + AllGather 的使用频率正在超过 AllReduce，
> 因为 ZeRO-3 / FSDP 将梯度和参数分片到各 Rank。理解这一点有助于你判断网络流量模型。

## 2.3 传输后端


| 传输方式             | 条件               | 单链路带宽               | 延迟     | 网络开发者关注度    |
| ---------------- | ---------------- | ------------------- | ------ | ----------- |
| **P2P (NVLink)** | 同机，NVLink 互联     | 600GB/s (NVL4)      | ~1μs   | 低（GPU 内部总线） |
| **P2P (PCIe)**   | 同机，同 PCIe Switch | ~32GB/s (PCIe4 x16) | ~2μs   | 低           |
| **SHM**          | 同机，跨 NUMA        | ~20GB/s（走 Host 内存）  | ~5μs   | 低           |
| **NET (RDMA)**   | 跨机               | 25-50GB/s (HDR/NDR) | ~2-5μs | ★★★ 核心      |
| **NET (Socket)** | 跨机，无 RDMA        | ~10GB/s             | ~50μs  | ★ 兜底        |


---

# 第三章：集合通信算法 — 网络流量模型

> **本章目标**：不仅理解算法"做什么"，更要理解算法在网络上"长什么样"——
> 每条链路上跑多少流量、流量什么时候到达、瓶颈在哪里。

## 3.1 Ring AllReduce

### 核心思想

N 个 GPU 组成逻辑环，数据切 N 份，通过 Scatter-Reduce + AllGather 两轮完成全局归约。

### 两阶段过程

**阶段一：Scatter-Reduce（N-1 步）**

- 每步每个 GPU 向下游发一个 Chunk，同时接收上游 Chunk 并做 Reduce
- 结束后，每个 GPU 恰好持有一个 Chunk 的完整归约结果

**阶段二：AllGather（N-1 步）**

- 每步每个 GPU 将完整 Chunk 传给下游
- 结束后，所有 GPU 拥有全部归约结果

### 3-GPU 示例

```
初始:  GPU0:[A0,B0,C0]  GPU1:[A1,B1,C1]  GPU2:[A2,B2,C2]

Scatter-Reduce Step 1:
  GPU0──C0──→GPU1    GPU1──A1──→GPU2    GPU2──B2──→GPU0
  GPU0: B0+B2        GPU1: C0+C1        GPU2: A1+A2

Scatter-Reduce Step 2:
  GPU0──B02──→GPU1   GPU1──C01──→GPU2   GPU2──A12──→GPU0
  GPU0: A全          GPU1: B全          GPU2: C全

AllGather Step 1:
  GPU0──A全──→GPU1   GPU1──B全──→GPU2   GPU2──C全──→GPU0

AllGather Step 2:
  循环传递，所有 GPU 持有 [A全, B全, C全]
```

### 网络流量分析


| 指标          | 值                        | 网络含义         |
| ----------- | ------------------------ | ------------ |
| 总步数         | 2(N-1)                   | 延迟正比于 Rank 数 |
| 每步每 GPU 发送量 | Data/N                   | 每条链路的瞬时负载    |
| 每 GPU 总发送量  | 2(N-1)/N × Data ≈ 2×Data | 每个端口的总流量     |
| 链路利用率       | **接近 100%**              | 每一步每条链路都在传数据 |


> **架构师洞察**：Ring 的精妙之处在于**所有链路同时工作、负载完全均衡**。
> 这和网络中的「满二分带宽拓扑」追求的目标一致。
> 但延迟是 O(N)——对 1024-GPU 集群意味着 ~1000 步串行延迟，这在小消息场景下不可接受。

### 跨机 Ring 的物理链路映射

```
8-GPU 2-Node Ring (4 GPU/Node, 1 NIC/Node):

  Node 0                    Node 1
  GPU0 ──NVLink──→ GPU1     GPU4 ──NVLink──→ GPU5
   ↑                ↓        ↑                ↓
  GPU3              GPU2     GPU7              GPU6
   ↑    ←NVLink←    ↓        ↑    ←NVLink←    ↓
   │                         │
   └──────── RDMA ───────────┘
        (跨机段: 瓶颈!)
```

跨机链路是整个 Ring 的瓶颈——节点内 NVLink 600GB/s，跨机 RDMA 只有 25GB/s。
NCCL 通过**多 Channel（每个走不同 NIC）** 来缓解这个瓶颈。

## 3.2 Tree AllReduce

### 核心思想

构建树形拓扑：Reduce（叶→根汇聚）+ Broadcast（根→叶分发）。

### 单树的问题

Root 节点同时收所有子节点的数据并发送结果——负载是 Leaf 的 2 倍，且 Root 所在的链路成为瓶颈。

### Double Binary Tree

构建**两棵互补的二叉树**并行传输：

- 树 1 的 Root 在树 2 中是 Leaf
- 每个节点在两棵树上的角色互补
- 两棵树各传一半数据，所有节点负载均衡

### 复杂度对比


| 指标    | Ring   | Tree         |
| ----- | ------ | ------------ |
| 步数    | 2(N-1) | 2×log₂(N)    |
| 延迟    | O(N)   | O(log N)     |
| 带宽利用率 | 接近最优   | 较低（树根瓶颈）     |
| 适合场景  | 大数据量   | 小数据量 / 大规模集群 |


> **架构师洞察**：Ring vs Tree 的选择本质上是**带宽优先 vs 延迟优先**的经典网络 trade-off。
> 和网络中「store-and-forward vs cut-through」的选择逻辑一致。

## 3.3 CollNet（SHARP 网内计算）

### 核心思想

让**网络交换机**参与 Reduce 计算。数据不需要到达所有 GPU 再归约——在交换机内部就完成了。

```
传统 Tree:
  GPU0 ──data──→ GPU_root ──reduce──→ result ──broadcast──→ GPUx

CollNet (SHARP):
  GPU0 ──data──→ IB Switch (内置 Reduce 引擎) ──result──→ GPU0
  GPU1 ──data──→ IB Switch                     ──result──→ GPU1
```

### 网络意义

- 跨机通信量减半（不需要 AllGather 阶段）
- 延迟 O(log N) 但常数更小
- **前提**：需要 Mellanox/NVIDIA 支持 SHARP 的交换机

> **架构师洞察**：SHARP 代表了网络架构的一个趋势——**计算下沉到网络设备**。
> 类似 SmartNIC offload，但做得更激进——直接在交换芯片上做归约运算。

## 3.4 算法选择策略

NCCL 根据数据量、GPU 数量、是否有 SHARP 自动选择：


| 条件         | 选择      | 原因           |
| ---------- | ------- | ------------ |
| 数据量大、GPU 少 | Ring    | 带宽利用率最优      |
| 数据量小、GPU 多 | Tree    | O(log N) 延迟  |
| 有 SHARP 硬件 | CollNet | 网内归约，延迟和带宽双赢 |


环境变量 `NCCL_ALGO=RING/TREE/COLLNET` 可强制指定。

---

# 第四章：软件架构 — 对标网络协议栈

## 4.1 整体架构

```
┌────────────────────────────────────────────────────────────┐
│  User API: ncclAllReduce(sendbuf, recvbuf, count, ...)     │  ← 应用层
├────────────────────────────────────────────────────────────┤
│  Enqueue Layer: 用户调用 → ncclWork 结构体 → Host FIFO     │  ← 会话层
├────────────────────────────────────────────────────────────┤
│  Algorithm Layer: 选择 Ring/Tree, 计算 Pattern/Chunk       │  ← 路由层
├────────────────────────────────────────────────────────────┤
│  Protocol Layer: Simple / LL / LL128                       │  ← 传输层
│    决定数据如何分帧、如何标记、如何确认                       │
├────────────────────────────────────────────────────────────┤
│  Transport Layer:                                          │  ← 数据链路层
│    ┌──────────┐  ┌──────────┐  ┌────────────────────────┐  │
│    │ P2P      │  │ SHM      │  │ NET (ncclNet Plugin)   │  │
│    │ NVLink   │  │ Host Mem │  │ ibv_post_send / socket │  │
│    └──────────┘  └──────────┘  └────────────────────────┘  │
├────────────────────────────────────────────────────────────┤
│  Hardware: NVSwitch | PCIe Switch | IB NIC | Ethernet NIC  │  ← 物理层
└────────────────────────────────────────────────────────────┘
```

## 4.2 对标网络协议栈


| NCCL 层    | 网络协议栈等价物            | 核心职责                       |
| --------- | ------------------- | -------------------------- |
| User API  | Application         | 提供 AllReduce/Broadcast 等语义 |
| Enqueue   | Session             | 管理异步调用、任务排队                |
| Algorithm | Network/Routing     | 决定数据走什么路径（Ring 环路/Tree 树路） |
| Protocol  | Transport (TCP/UDP) | 决定数据如何分帧、可靠性保证、延迟优化        |
| Transport | Data Link           | 驱动具体硬件发送/接收，管理 Buffer 和连接  |
| Hardware  | Physical            | NVLink/PCIe/RDMA 物理传输      |


## 4.3 Persistent Kernel — 为什么不能每次启动新线程

**问题**：`cudaLaunchKernel` 延迟约 10μs。AllReduce 在训练中每 10-50ms 调一次，如果每次都 launch Kernel，仅启动开销就占 0.02-0.1%。听起来不多？但 Ring 的 2(N-1) 步中，每步都会有同步等待，launch 开销会被放大。

**方案**：初始化时启动一批永不退出的 GPU 线程（占几个 SM），通过轮询 Host 内存中的任务队列获取新任务。

```
初始化阶段 (仅一次):
  Host → cudaLaunchKernel → GPU 上常驻一批线程, 进入 while(true) 轮询循环

运行阶段 (每次 ncclAllReduce):
  Host → 构造 ncclWork → 写入 FIFO → 更新 Head 指针
  GPU  → volatile 读 Head → 发现新任务 → 执行通信 → 回到轮询
```

> **RDMA 类比**：等价于 RDMA 用户态驱动中的**持久轮询线程**——启动后永不退出，
> 通过轮询 CQ 发现完成事件。避免了每次操作都 create/destroy 线程的开销。

## 4.4 NCCL 中的两对生产者-消费者

NCCL 内部有**两对方向相反的生产者-消费者关系**，功能完全不同，容易混淆：

```
对 ①: 任务下发 (CPU → GPU)                对 ②: 网络 I/O 触发 (GPU → CPU)
┌──────────┐   FIFO   ┌──────────┐       ┌──────────┐  Mailbox  ┌──────────┐
│ Host CPU │ ───────→ │ GPU 线程  │       │ GPU 线程  │ ────────→ │  Proxy   │
│ 写 ncclWork│        │ 轮询执行   │       │ 写 Head   │          │ ibv_post │
└──────────┘          └──────────┘       └──────────┘          └──────────┘
"我有新的 AllReduce 任务给你"              "数据算好了, 你可以发到网络了"
```


| 对比         | 对 ①：任务下发              | 对 ②：网络 I/O 触发           |
| ---------- | --------------------- | ----------------------- |
| **方向**     | CPU → GPU             | GPU → CPU               |
| **介质**     | FIFO (Pinned Memory)  | Mailbox (Pinned Memory) |
| **生产者写什么** | ncclWork 结构体 (整个任务描述) | Head 计数器 (一个整数)         |
| **消费者做什么** | GPU 线程解析任务、启动通信       | Proxy 调用 ibv_post_send  |
| **详见**     | 本节 (4.3/4.4)          | 第七章 (7.4/7.5)           |


### 对 ①：任务 FIFO 的 Ring Buffer 机制

```
FIFO 布局 (Pinned Memory, CPU 和 GPU 共享访问):
  ┌────────┬────────┬────────┬────────┬─── ───┐
  │ Slot 0 │ Slot 1 │ Slot 2 │ Slot 3 │ ...   │  每个 Slot 存一个 ncclWork
  └────────┴────────┴────────┴────────┴───────┘
       ↑ Tail (GPU 读到这)        ↑ Head (CPU 写到这)

CPU (生产者):
  fifo[head % DEPTH] = ncclWork{...};    // 写任务
  __sync_synchronize();                   // 内存屏障: 确保数据落地
  head++;                                 // 更新写指针

GPU (消费者):
  while (*head_ptr == tail) ;             // volatile 轮询: 等待新任务
  __threadfence_block();                  // GPU 侧内存屏障
  work = fifo[tail % DEPTH];              // 读取任务
  tail++;                                 // 更新读指针
```

ncclWork 中 GPU 关心的核心字段：


| 字段             | GPU 线程用它做什么                             |
| -------------- | --------------------------------------- |
| 操作类型           | 跳转到 AllReduce / Broadcast / ... 对应的代码路径 |
| 用户 Buffer 地址   | 从这里读源数据、把结果写回这里                         |
| 数据量 + Chunk 方案 | 计算自己负责哪一段数据、每次搬多大的 Slice                |
| Channel 编号     | 确定用哪组 Scratch Buffer、连接哪个 Peer          |


> **RDMA 类比**：这个 FIFO 就是一个**软件实现的 Work Queue**。
> CPU 像 `ibv_post_send` 一样往里写 Work Request，GPU 像网卡一样从里面取 WR 执行。
> Head/Tail 指针的语义和 RDMA SQ 的 Producer Index / Consumer Index 完全一致。
> 区别是 RDMA 的 WQ 由硬件（网卡）消费，这里的 FIFO 由 GPU 线程消费。

---

# 第五章：ncclNet 网络插件接口 ★ 网络开发者的主战场

> **为什么这章最重要**：ncclNet 是 NCCL 对外暴露的网络抽象接口。
> 掌握了这个接口，你就能：(1) 理解 NCCL 如何使用网络 (2) 为自己的硬件写 NCCL 传输后端 (3) 排查网络性能问题。

## 5.1 插件加载机制

```
NCCL 启动时的插件发现顺序:
  1. 检查 NCCL_NET_PLUGIN 环境变量（如 "ofi"）
  2. dlopen("libnccl-net-ofi.so") 或 dlopen("libnccl-net.so")
  3. 查找导出符号 ncclNetPlugin_v8
  4. 调用 init() 初始化插件
  5. 如果失败或未找到，fallback 到内置 IB transport (net_ib.cc)
  6. 最后 fallback 到 Socket transport (net_socket.cc)
```

## 5.2 ncclNet 接口全貌（v8）

```c
typedef struct {
  const char* name;

  // === 初始化 / 发现 ===
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);          // 有多少个网络设备
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v8* props);

  // === 连接建立（三次握手） ===
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm, ...);
  ncclResult_t (*accept)(void* listenComm, void** recvComm, ...);

  // === 内存注册 ===
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);

  // === 数据传输（异步） ===
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag,
                        void* mhandle, void** request);
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes,
                        int* tags, void** mhandles, void** request);
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, size_t* sizes,
                         void** mhandles, void** request);
  ncclResult_t (*test)(void* request, int* done, int* sizes);

  // === 清理 ===
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v8_t;
```

## 5.3 接口语义详解 — RDMA 开发者视角


| 接口              | RDMA 等价操作                         | NCCL 调用时机         | 关键注意                     |
| --------------- | --------------------------------- | ----------------- | ------------------------ |
| `init`          | 扫描 IB 设备                          | 进程启动              | 初始化 ibv_context          |
| `devices`       | 枚举 HCA                            | 进程启动              | 返回可用网卡数                  |
| `getProperties` | ibv_query_device / ibv_query_port | Topo Discovery    | 返回速率、延迟、GDR 支持等          |
| `listen`        | bind + listen (OOB 连接)            | 连接建立              | 返回 handle 供对端 connect 使用 |
| `connect`       | ibv_create_qp + RTR/RTS           | 连接建立              | 创建 Send 端连接              |
| `accept`        | 接受连接 + 创建 QP                      | 连接建立              | 创建 Recv 端连接              |
| `regMr`         | ibv_reg_mr                        | 每个 Buffer 首次使用    | **性能关键**：频繁调用会拖垮性能       |
| `isend`         | ibv_post_send (RDMA WRITE)        | 每个 Slice 发送       | 异步，返回 request            |
| `irecv`         | ibv_post_recv 或预分配 Recv Buffer    | 每个 Slice 接收       | 告诉网络层"准备接收"              |
| `iflush`        | 内存屏障 / DMA 完成确认                   | GDR 模式下 irecv 完成后 | 确保 GPU 能看到 NIC 写入的数据     |
| `test`          | ibv_poll_cq                       | Proxy 轮询循环        | 检查异步操作是否完成               |


> **架构师洞察**：注意 `isend`/`irecv`/`test` 这组异步接口的设计——
> 它和 Linux aio 或 io_uring 的 submit/poll 模型完全一致。
> NCCL 把网络操作建模为「提交 → 轮询完成」，这让 Proxy Thread 可以高效地用单线程驱动多个并发传输。

## 5.4 MR 注册缓存 — 隐藏的性能杀手

`ibv_reg_mr` 涉及：内核态切换 → 锁页表 → IOMMU 映射。单次耗时 10-100μs。
如果每次 AllReduce 都对用户 Buffer 做 regMr / deregMr，性能会崩。

**NCCL 的解决方案：MR Registration Cache**

```
regMr(addr, size) 被调用时:
  1. 查 Cache: 这个 (addr, size) 注册过吗？
  2. 命中 → 直接返回 cached mhandle（引用计数+1）
  3. 未命中 → 调用 ibv_reg_mr → 缓存结果 → 返回

deregMr(mhandle) 被调用时:
  1. 引用计数-1
  2. 引用计数 > 0 → 保留在 Cache
  3. 引用计数 = 0 → 可以延迟释放或 LRU 淘汰
```

> **RDMA 类比**：你写 RDMA 应用时，一定不会在每次 `ibv_post_send` 前都调 `ibv_reg_mr`，
> 发完再 `ibv_dereg_mr`——那性能会烂到不可用。常规做法是连接建立时注册好 MR，整个生命周期复用。
>
> 但 NCCL 面对的问题更难：**用户 Buffer 地址不固定**。PyTorch 每次传给 `ncclAllReduce` 的 tensor
> 地址可能不同（显存分配器动态管理）。所以 NCCL 不能像你写 RDMA 应用那样"注册一次用到底"，
> 它需要一个 Cache——地址见过就复用旧 MR，没见过才真正调 `ibv_reg_mr`。
> 本质上和 CPU 的 **TLB**（Translation Lookaside Buffer）思路一致：
> 完整的地址翻译（页表遍历 / MR 注册）太贵，用 Cache 加速热路径。

## 5.5 学习参考：aws-ofi-nccl

[aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) 是 AWS 基于 libfabric (OFI) 实现的 NCCL 网络插件。

**为什么推荐它**：

- 代码量比 NCCL 内置的 net_ib.cc 小得多，结构更清晰
- 完整实现了 ncclNet_v8 的所有接口
- 注释详尽，是理解插件接口的最佳教材
- 展示了如何把一个非 IB Verbs 的网络 API（libfabric）适配到 ncclNet

---

# 第六章：数据面 — 字节如何在网络上流动

## 6.1 数据切分层次

```
User Buffer (例: 1GB AllReduce)
  │
  ├─→ Chunk: 按算法切分 (Ring: 1GB/8=128MB per Chunk)
  │     用途: 保证算法正确性
  │
  └─→ Slice: 按 Buffer 大小切分 (128MB → 64×2MB Slice)
        用途: 流水线粒度，每个 Slice 是一次 isend/irecv 的单位
```


| 层次        | 大小           | 决定因素              | 网络含义           |
| --------- | ------------ | ----------------- | -------------- |
| **Chunk** | Data/N_ranks | 算法                | 每个 Rank 负责的数据段 |
| **Slice** | 512KB ~ 2MB  | Scratch Buffer 容量 | 单次 RDMA 传输的粒度  |


> **架构师洞察**：Slice 大小是一个关键调优参数。太小 → RDMA 小消息多、overhead 大；太大 → 流水线深度不够、计算通信无法重叠。
> 这和网络中 MTU 的选择逻辑一致。

## 6.2 Scratch Buffer — NCCL 的 DMA Ring Buffer

### 是什么

一块初始化时 `cudaMalloc` 预分配的显存（每 Channel 约 4MB），已永久注册为 RDMA MR。

### 三重设计动机


| 动机             | 解释                                                         | RDMA 等价                |
| -------------- | ---------------------------------------------------------- | ---------------------- |
| **避免频繁 MR 注册** | 用户 Buffer 地址每次可能不同，Scratch Buffer 一次注册永久使用                 | 预注册的 DMA Buffer        |
| **原地操作安全**     | AllReduce 常是 in-place（结果覆盖输入）。上游数据直接写用户 Buffer 会踩掉未计算的本地数据 | Staging Buffer 解耦读写    |
| **标准化传输粒度**    | 用户 Buffer 大小随意，Scratch Buffer 是固定尺寸的标准容器                   | 固定大小的 Recv Ring Buffer |


### RDMA 精确类比

```
Scratch Buffer ≈ RDMA 的 Recv Ring Buffer (固定地址、固定大小、预注册 MR)
  作用: 解耦「用户数据的随意性」和「DMA 引擎的严谨性」
  生命周期: 和 Connection 一样长，不随单次操作创建/销毁
```

## 6.3 双缓冲（Double Buffering）

### 问题

单缓冲：NIC DMA 写入时 GPU 必须等待，GPU 计算时 NIC 空闲。吞吐减半。

### 方案

Scratch Buffer 分为 Slot 0 / Slot 1，交替使用：

```
        ┃  T1        ┃  T2        ┃  T3        ┃  T4        ┃  T5
  ──────╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━
  NIC   ┃ →Slot0     ┃ →Slot1     ┃ →Slot0     ┃ →Slot1     ┃ →Slot0
  DMA写 ┃ ██ Slice0  ┃ ██ Slice1  ┃ ██ Slice2  ┃ ██ Slice3  ┃ ██ ...
  ──────╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━
  GPU   ┃            ┃  ←Slot0    ┃  ←Slot1    ┃  ←Slot0    ┃  ←Slot1
  计算  ┃  (idle)    ┃ ▓▓ Slice0  ┃ ▓▓ Slice1  ┃ ▓▓ Slice2  ┃ ▓▓ ...
  ──────╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━
                     ↑ overlap    ↑ 满流水线: NIC 和 GPU 再无空闲

  ██ = NIC 正在 DMA 写入     ▓▓ = GPU 正在读取+计算
  Slot0/Slot1 交替角色：一个正被写，另一个正被读——永远不冲突
```

**关键**：T1 是冷启动代价（GPU 必须等第一块数据到达），从 T2 起 NIC 与 GPU 完全重叠，
吞吐接近 `min(NIC带宽, GPU计算速率)`，而非单缓冲下的 `NIC + GPU 串行求和`。

> **RDMA 类比**：和 RDMA Verbs 中的 **Double-Posting** 技术一致——
> 连续 post 两个 Recv WR，DMA 到第一个 Buffer 的同时处理前一个 Buffer 的数据。

## 6.4 信号量与流控

### GPU 如何知道数据到了

```
上游写完数据后:
  上游 GPU/Proxy → 更新 Tail 计数器 (Pinned Memory 或 P2P 直写)

下游 GPU 轮询:
  while (*tail < expected_step) ;     // 自旋
  __threadfence_block();               // 内存屏障: 确保读到最新数据
  // 安全读取 Scratch Buffer
```

### Credit 流控（防止覆写）

双缓冲只有 2 个 Slot，上游不能无限发送：

```
初始 Credit = 2 (Slot 0 + Slot 1)

上游发 Slice 0 → Credit = 1
上游发 Slice 1 → Credit = 0 → 阻塞
下游处理完 Slot 0 → 更新 Head → 上游感知到 → Credit = 1 → 继续发
```

> **RDMA 精确类比**：等价于 RDMA SRQ 的 Credit-Based Flow Control。
> 发端根据对端可用 Recv Buffer 数量决定是否可以继续 post_send。
> NCCL 的 Credit = Scratch Buffer Slot 数。

## 6.5 通信路径对比

### 路径 A：本机 P2P (NVLink/PCIe)

```
GPU0 计算完毕
  → GPU0 线程执行 ST.GPU 指令, 直接写入 GPU1 的 Scratch Buffer
  → GPU0 更新 GPU1 的 Tail 信号量
  → CPU 完全不参与, 延迟 ~1μs
```

### 路径 B：跨机 RDMA（最复杂路径）

```
1. GPU0 计算完 Slice → 数据在 GPU0 显存
2. GPU0 线程: __threadfence_system() → 更新 Mailbox Head (Pinned Memory)
3. CPU Proxy Thread: 轮询 Mailbox → 发现 Head 变化
4. Proxy: 调用 ncclNet->isend(gpu_addr, size, mr_handle)
   └→ 内部: ibv_post_send(QP, WR{RDMA_WRITE, gpu_addr, rkey})
5. NIC: 通过 PCIe DMA 读取 GPU0 显存 (GDR: GPU→PCIe Switch→NIC)
6. NIC: 组 IB 包 → 发到网络
7. 对端 NIC: 收到数据 → RDMA WRITE 到对端 GPU1 Scratch Buffer
8. 对端 Proxy: 轮询 ncclNet->test() → CQE 到达
   └→ 内部: ibv_poll_cq → got CQE
9. 对端 Proxy: 更新对端 GPU1 的 Tail 信号量
10. GPU1 线程: 发现 Tail 更新 → 解除自旋 → 开始计算
```

> **架构师洞察**：注意步骤 2→3 和 8→9 的 GPU↔CPU 交互。
> 这两个环节引入的延迟（各约 1-5μs）是跨机通信中除网络 RTT 外最大的开销来源。
> 这也是为什么 NVIDIA 一直在探索让 GPU 线程直接驱动网络操作（跳过 Proxy）的方案。

---

# 第七章：控制面 — 连接建立与生命周期

## 7.1 初始化全景

```
ncclCommInitRank(comm, nranks, id, rank)
  │
  ├─ Phase 1: Bootstrap（带外控制面）
  │   ├─ Rank 0 创建 TCP Listen Socket
  │   ├─ 所有 Rank 通过 TCP 连成控制环
  │   └─ 交换: hostname, GPU 编号, PID 等元信息
  │
  ├─ Phase 2: Topology Discovery（拓扑发现）
  │   ├─ 扫描 /sys/class/ 下的 PCIe/NVLink/NIC 设备
  │   ├─ 构建 GPU-NIC-Switch 拓扑图
  │   ├─ 为每条边赋带宽权值
  │   └─ 搜索最优 Ring/Tree 路径
  │
  ├─ Phase 3: Channel Setup（逻辑连接）
  │   ├─ 确定 Channel 数量（基于拓扑和 GPU 数）
  │   └─ 为每个 Channel 分配 Ring/Tree 路径上的 Peer 关系
  │
  ├─ Phase 4: Transport Setup（物理连接）★ 网络核心
  │   ├─ 对每个 Channel 的每对 Peer:
  │   │   ├─ P2P 连接: cudaIpcGetMemHandle 交换显存句柄
  │   │   └─ NET 连接:
  │   │       ├─ ncclNet->listen()    → 获得 handle
  │   │       ├─ 通过 Bootstrap 交换 handle
  │   │       ├─ ncclNet->connect()   → 创建 SendComm (QP)
  │   │       ├─ ncclNet->accept()    → 创建 RecvComm (QP)
  │   │       └─ ncclNet->regMr()     → 注册 Scratch Buffer
  │   └─ 分配 Scratch Buffer (cudaMalloc + regMr)
  │
  ├─ Phase 5: Proxy Start
  │   └─ 启动 CPU Proxy 线程, 开始轮询 Mailbox
  │
  └─ Phase 6: Kernel Launch
      └─ 启动 GPU Persistent Kernel (仅一次)
```

## 7.2 Bootstrap 协议详解

Bootstrap 是 NCCL 的**带外控制面**，类似 RDMA CM（Connection Manager）的角色。

```
Step 1: Rank 0 创建 TCP Server Socket, bind 端口
Step 2: 所有 Rank 从 ncclUniqueId 中解析出 Rank 0 的地址
Step 3: 所有 Rank 连接 Rank 0, 发送自己的 (rank, host, GPU info)
Step 4: Rank 0 收齐所有 Rank 信息后, 构建 Ring 拓扑:
        Rank0 → Rank1 → Rank2 → ... → RankN-1 → Rank0
Step 5: Rank 0 将 Ring 信息下发给所有 Rank
Step 6: 所有 Rank 通过 Ring 连接建立全连接的控制通道
```

> **架构师洞察**：Bootstrap 用 TCP 而非 RDMA，因为：
> (1) 控制面流量极小，不需要高性能 (2) TCP 的可移植性更好
> (3) RDMA 连接本身需要带外通道交换 QP 信息——鸡生蛋的问题。

## 7.3 Transport Setup 中的 QP 拓扑

假设 8 GPU、2 节点、4 Channel 的 Ring：

```
每个 Channel 对应独立的 Ring:
  Channel 0: GPU0→GPU1→GPU2→GPU3→GPU4→GPU5→GPU6→GPU7→GPU0
  Channel 1: GPU0→GPU2→GPU4→GPU6→GPU1→GPU3→GPU5→GPU7→GPU0
  ...

跨机连接 (GPU3→GPU4 和 GPU7→GPU0):
  每个 Channel 的每对跨机 Peer 创建一组 QP
  4 Channel × 2 跨机 Peer = 8 对 QP

  总 QP 数 = N_channels × N_cross_node_peers × 2 (send + recv)
```

> **架构师洞察**：QP 数量随 Channel 数线性增长。在大集群中，单个节点可能有 100+ QP。
> 这对 NIC 的 QP 缓存（Cache）和调度能力是一个挑战。
> 这也是为什么一些高性能网络方案倾向于用 Shared Receive Queue (SRQ) 来减少 QP 数量。

## 7.4 Proxy Thread 架构

### 线程模型

```
每个 ncclComm 启动一个 Proxy Thread:
  while (running) {
      for (each channel) {
          for (each connection in channel) {
              // 检查 Mailbox: GPU 有新数据要发?
              if (conn->head > conn->last_posted) {
                  ncclNet->isend(conn->sendComm, data, size, ...);
                  conn->last_posted++;
              }
              // 检查完成: 之前发的数据到了吗?
              if (conn->pending_requests > 0) {
                  ncclNet->test(request, &done, ...);
                  if (done) {
                      conn->tail++;  // 告诉 GPU: Buffer 可以复用
                      conn->pending_requests--;
                  }
              }
          }
      }
  }
```

### Progress Function 模式

Proxy Thread 的核心设计是 **progress function**——每个连接注册自己的 progress 函数，Proxy 在主循环中依次调用。这让单个线程能高效驱动多个并发传输。

> **RDMA 类比**：等价于 `libibverbs` 应用中常见的「单线程事件循环」模式——
> 一个线程轮询多个 CQ，根据 CQE 类型分发到不同的 handler。
> NCCL 的 Proxy Thread 就是这个模式的 GPU 通信版本。

## 7.5 Mailbox — GPU ↔ CPU 协议

Mailbox 是一块 Pinned Memory（CPU 和 GPU 通过 UVA 共享访问）：


| 字段   | 写方           | 读方           | 语义                             |
| ---- | ------------ | ------------ | ------------------------------ |
| Head | GPU 线程       | Proxy Thread | "第 N 个 Slice 数据准备好了，可以发送"      |
| Tail | Proxy Thread | GPU 线程       | "第 N 个 Slot 已传输完毕，Buffer 可以复用" |


```
本质上是一个单生产者单消费者的无锁 Ring Buffer:
  GPU 是 Producer (写 Head)
  Proxy 是 Consumer (读 Head, 写 Tail)
  不需要锁, 靠 volatile + memory fence 保证可见性
```

---

# 第八章：通信协议 — Simple / LL / LL128

## 8.1 协议总览


| 协议         | 数据传输方式                               | 完成通知机制         | 有效带宽   | 延迟           | 适用   |
| ---------- | ------------------------------------ | -------------- | ------ | ------------ | ---- |
| **Simple** | GPU→Mailbox→Proxy→ibv_post_send      | Proxy 轮询 CQ    | 100%   | 高 (经过 Proxy) | 大消息  |
| **LL**     | GPU 直接写 (data+flag) 到对端              | 对端 GPU 轮询 flag | 50%    | 极低           | <8KB |
| **LL128**  | GPU 用 128B 粒度写 (120B data + 8B flag) | 对端 GPU 轮询 flag | 93.75% | 低            | 中等消息 |


## 8.2 Simple 协议详解

```
发送端:
  GPU Thread → 计算完成 → __threadfence_system()
            → 写 Mailbox Head (Pinned Mem)
  Proxy Thread → 轮询 Head → 调用 ncclNet->isend()
              → (内部: ibv_post_send with RDMA_WRITE)
              → 轮询 ncclNet->test()
              → (内部: ibv_poll_cq)
              → 更新 Mailbox Tail

接收端:
  Proxy Thread → ncclNet->irecv() 提交接收
              → 轮询 ncclNet->test() → CQE 到达
              → ncclNet->iflush() (GDR 下确保数据可见)
              → 更新 GPU 可见的 Tail 信号量
  GPU Thread → 发现 Tail 更新 → 读取 Scratch Buffer → 开始计算
```

**网络行为**：每个 Slice 对应一次完整的 RDMA WRITE 操作。

## 8.3 LL (Low Latency) 协议

```
数据格式 (每 8 字节):
  [4B data] [4B flag]

发送端 GPU 线程:
  直接通过 P2P 或 RDMA WRITE 把 (data+flag) 写到对端 Scratch Buffer
  不经过 Proxy Thread!

接收端 GPU 线程:
  while (scratch[offset].flag != expected_flag) ;
  data = scratch[offset].data;
```

**关键特点**：

- 跳过 Proxy Thread，GPU 线程直接驱动传输
- 有效带宽只有 50%（一半被 flag 占用）
- 但延迟极低——省去了 GPU↔CPU 的 Mailbox 交互

> **架构师洞察**：LL 协议的本质是用**带内信令（in-band signaling）** 替代 **带外信令（out-of-band signaling）**。
> flag 和 data 在同一个 RDMA WRITE 中原子写入，接收端通过检查 flag 即可判断数据是否到达。
> 这和 TCP 中 PSH flag 的设计思路异曲同工——数据和控制信息复用同一个通道。

## 8.4 LL128 协议

```
数据格式 (每 128 字节):
  [120B data] [8B flag]

有效负载率: 120/128 = 93.75%
```

LL128 利用 NVLink 的 128B 原子传输保证：120B 数据和 8B flag 在一次 128B 写操作中原子到达。

> **注意**：LL128 主要用于 NVLink 场景。在 RDMA 跨机场景下，NCCL 通常在 Simple 和 LL 之间选择。

## 8.5 协议选择


| 数据量        | 选择             | 原因               |
| ---------- | -------------- | ---------------- |
| 极小 (<几 KB) | LL             | 延迟主导，50% 带宽损失可接受 |
| 中等         | LL128 (NVLink) | 延迟和带宽的折中         |
| 大 (>几 MB)  | Simple         | 带宽主导，100% 有效带宽   |


环境变量 `NCCL_PROTO=SIMPLE/LL/LL128` 可强制指定。

---

# 第九章：GPUDirect RDMA — 驱动级深度剖析

> **本章对 RDMA 驱动开发者特别重要**——你日常工作中的 MR 注册、DMA 映射、peer memory 全在这里汇聚。

## 9.1 三种数据路径

### 传统路径（无 GDR）

```
发送: GPU → cudaMemcpy → Host Bounce Buffer → ibv_post_send → NIC → 网络
接收: 网络 → NIC → Host Bounce Buffer → cudaMemcpy → GPU

PCIe 穿越次数: 4 (GPU→Host, Host→NIC, NIC→Host, Host→GPU)
```

### GPUDirect RDMA (GDR) — 同 PCIe Switch

```
发送: GPU → PCIe Switch → NIC → 网络
接收: 网络 → NIC → PCIe Switch → GPU

PCIe 穿越次数: 2
条件: GPU 和 NIC 在同一个 PCIe Switch 下
```

### GPUDirect RDMA — 跨 PCIe Switch

```
发送: GPU → PCIe Switch → CPU Root Complex → PCIe Switch → NIC → 网络

PCIe 穿越次数: 2, 但经过 Root Complex, 带宽受限
```

## 9.2 nvidia-peermem 内核模块 — 驱动实现

### 工作原理

```
1. NCCL 调用 ibv_reg_mr(pd, gpu_virt_addr, size, access_flags)

2. ib_core 内核模块调用 ib_umem_get() 尝试 pin 内存
   → 发现 gpu_virt_addr 不是普通用户内存

3. ib_core 查询已注册的 peer_memory_client
   → 找到 nvidia-peermem

4. nvidia-peermem 回调: acquire()
   → 调用 nvidia_p2p_get_pages(gpu_virt_addr)
   → 获取 GPU BAR 空间的物理地址 (struct nvidia_p2p_page_table)
   → 返回 sg_table (scatter-gather list of GPU BAR pages)

5. ib_core 拿到 sg_table → 创建 MR
   → MR 的 DMA 地址指向 GPU BAR 空间

6. 网卡 DMA 引擎直接通过 PCIe 读写 GPU BAR 地址
```

### 关键内核接口


| 接口                               | 来源           | 作用                      |
| -------------------------------- | ------------ | ----------------------- |
| `ib_register_peer_memory_client` | ib_core      | 注册 peer memory provider |
| `nvidia_p2p_get_pages`           | nvidia.ko    | 获取 GPU 显存的 BAR 物理地址     |
| `nvidia_p2p_put_pages`           | nvidia.ko    | 释放映射                    |
| `nvidia_p2p_free_page_table`     | nvidia.ko    | 释放页表                    |
| `mmu_notifier`                   | Linux kernel | GPU 显存释放时通知 IB 驱动撤销 MR  |


### GPU BAR 空间

GPU 显存通过 **BAR1/BAR3** 映射到 PCIe 地址空间。这让 PCIe 上的其他设备（如 NIC）可以像访问普通 MMIO 一样访问 GPU 显存。

```
CPU 视角的物理地址空间:
  0x00000000 ─── System RAM
  ...
  0xC0000000 ─── GPU 0 BAR1 (映射到 GPU 0 显存)
  0xE0000000 ─── GPU 1 BAR1 (映射到 GPU 1 显存)
  ...

NIC DMA 读取 0xC0000000 → PCIe 路由到 GPU 0 → GPU 0 返回显存数据
```

> **架构师洞察**：BAR 空间大小有限（通常 256MB ~ 64GB，取决于 GPU 型号和配置）。
> 当需要注册的 GPU 内存超过 BAR 空间大小时，nvidia-peermem 需要做**BAR 空间的动态映射/重映射**，
> 这会引入额外延迟。大模型训练中显存使用量巨大，这是一个实际的性能陷阱。

## 9.3 有无 GDR 对 NCCL Proxy 的影响


| 方面        | 无 GDR                                                          | 有 GDR                                     |
| --------- | -------------------------------------------------------------- | ----------------------------------------- |
| Proxy 发送  | cudaMemcpyAsync(host, gpu) → 等待 → ibv_post_send(host_addr)     | ibv_post_send(gpu_addr)                   |
| Proxy 接收  | ibv_post_recv(host_addr) → 等待 CQE → cudaMemcpyAsync(gpu, host) | ibv_post_recv(gpu_addr) → 等待 CQE → iflush |
| 延迟        | 高 (+10-30μs per memcpy)                                        | 低                                         |
| CPU 占用    | 高 (memcpy 消耗 CPU 带宽)                                           | 低                                         |
| iflush 需要 | 否                                                              | **是** (确保 NIC DMA 写入对 GPU 可见)             |


---

# 第十章：多轨网络与 NIC 亲和性

> **为什么这章重要**：生产环境的 GPU 服务器通常有 4~8 张网卡，NCCL 如何分配流量直接决定网络利用率。

## 10.1 多轨拓扑

典型的 8-GPU DGX 节点拓扑：

```
         ┌─── NIC 0 (mlx5_0) ───┐
  GPU 0 ─┤                       ├─ PCIe Switch 0
  GPU 1 ─┤                       │
         └───────────────────────┘

         ┌─── NIC 1 (mlx5_1) ───┐
  GPU 2 ─┤                       ├─ PCIe Switch 1
  GPU 3 ─┤                       │
         └───────────────────────┘

         ┌─── NIC 2 (mlx5_2) ───┐
  GPU 4 ─┤                       ├─ PCIe Switch 2
  GPU 5 ─┤                       │
         └───────────────────────┘

         ┌─── NIC 3 (mlx5_3) ───┐
  GPU 6 ─┤                       ├─ PCIe Switch 3
  GPU 7 ─┤                       │
         └───────────────────────┘
```

## 10.2 NIC-GPU 亲和性

NCCL 的 Channel 分配核心原则：**让每个 Channel 使用物理上最近的 NIC**。

```
Channel 0 (Ring: GPU0→GPU1→GPU4→GPU5→...):
  跨机段由 GPU0 发送 → 选择 NIC 0 (同一 PCIe Switch) → GDR 走 PCIe Switch 内部路径

Channel 1 (Ring: GPU2→GPU3→GPU6→GPU7→...):
  跨机段由 GPU2 发送 → 选择 NIC 1 (同一 PCIe Switch) → GDR 走 PCIe Switch 内部路径
```

如果选错 NIC（比如 GPU0 的数据走 NIC 3），数据要穿越 CPU Root Complex，带宽降低且延迟增加。

## 10.3 Multi-rail 流量分配

4 个 NIC，每个 25GB/s，通过 4 个 Channel 并行传输：

```
总跨机带宽 = 4 × 25GB/s = 100GB/s

Channel 0 → NIC 0 → 25GB/s 的数据流
Channel 1 → NIC 1 → 25GB/s 的数据流
Channel 2 → NIC 2 → 25GB/s 的数据流
Channel 3 → NIC 3 → 25GB/s 的数据流
```

> **架构师洞察**：Multi-rail 的关键挑战不是"分流量"，而是**负载均衡**。
> 如果某个 Channel 的数据量偏大，对应 NIC 就成为瓶颈，其他 NIC 空闲。
> NCCL 通过让所有 Channel 传等量的数据（Chunk 均分）来保证均衡。
>
> 另一个挑战是 **PXN (PCIe cross-NUMA)**：当节点有更多 GPU 而 NIC 数量不够时，
> 部分 GPU 必须使用非亲和的 NIC。NCCL 2.12+ 引入 PXN 优化来处理这种情况——
> 先通过 NVLink 把数据搬到亲和 NIC 旁的 GPU，再由该 GPU 负责跨机传输。

## 10.4 关键环境变量


| 变量                            | 作用                             |
| ----------------------------- | ------------------------------ |
| `NCCL_IB_HCA`                 | 指定使用哪些 IB 网卡，如 `mlx5_0,mlx5_1` |
| `NCCL_NET_GDR_LEVEL`          | 控制 GDR 使用的激进程度 (0=关闭, 5=最激进)   |
| `NCCL_NCHANNELS_PER_NET_PEER` | 每对节点使用多少 Channel               |
| `NCCL_IB_QPS_PER_CONNECTION`  | 每个连接的 QP 数                     |


---

# 第十一章：拓扑发现与路径规划

## 11.1 发现过程

```
1. 扫描 sysfs 获取硬件信息:
   /sys/class/infiniband/          → IB 设备
   /sys/bus/pci/devices/           → PCIe 拓扑
   /proc/driver/nvidia/gpus/       → GPU 信息
   /sys/class/nvlink/              → NVLink 连接

2. 构建拓扑图 (ncclTopoGraph):
   节点: GPU, NIC, PCIe Switch, CPU, NVSwitch
   边: 物理连接, 带权值 (带宽)

3. 路径规划:
   用图搜索算法找最大带宽路径
   为每个 Channel 生成 Ring/Tree 的节点排列顺序
   核心约束: 跨机段必须经过 NIC, 且要选亲和的 NIC
```

## 11.2 拓扑类型标识

NCCL_DEBUG=INFO 输出中的拓扑类型：


| 标识      | 含义                           | 带宽级别     |
| ------- | ---------------------------- | -------- |
| **NVL** | NVLink 直连                    | ~600GB/s |
| **NVB** | NVLink 经过 NVSwitch           | ~600GB/s |
| **PIX** | PCIe 同 Switch                | ~32GB/s  |
| **PXB** | PCIe 跨 Switch (同 NUMA)       | ~16GB/s  |
| **PHB** | PCIe 跨 NUMA (经过 Host Bridge) | ~16GB/s  |
| **SYS** | QPI/UPI 跨 Socket             | ~16GB/s  |
| **NET** | 跨机网络                         | 取决于网卡    |


## 11.3 调试拓扑

```bash
# 查看 NCCL 探测到的完整拓扑和 Ring/Tree 路径
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH ./your_app

# 输出示例:
# NCCL INFO Trees [0] 1/-1/-1->0->4 [1] 1/-1/-1->0->5
# NCCL INFO Channel 00 : 0[0] -> 1[1] -> 2[2] -> 3[3] -> 4[4] -> ...
```

> **架构师洞察**：拓扑发现是 NCCL 最"网络化"的模块。它本质上做的事情和**网络路由协议（OSPF/IS-IS）** 一样：
> 发现拓扑 → 构建链路状态数据库 → 计算最短/最宽路径 → 安装路由表（Channel 配置）。
> 区别在于 NCCL 是静态发现（一次性），网络路由是动态的。

---

# 第十二章：SHARP 与网内计算

## 12.1 什么是 SHARP

**SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** 是 NVIDIA/Mellanox 的网内计算技术，让 InfiniBand 交换机参与集合通信的归约计算。

```
传统 AllReduce (Tree):
  Leaf GPUs → send data to Root GPU → Root computes reduce → broadcast result

SHARP AllReduce:
  All GPUs → send data to IB Switch → Switch computes reduce → send result back
```

## 12.2 NCCL 中的 CollNet

NCCL 通过 **CollNet** 接口抽象 SHARP 能力：

```c
typedef struct {
  ncclResult_t (*init)(...);
  ncclResult_t (*devices)(...);
  ncclResult_t (*listen)(...);
  ncclResult_t (*connect)(...);      // 连接到 SHARP aggregation node
  ncclResult_t (*reduceSupport)(...); // 查询支持的归约操作
  ncclResult_t (*regMr)(...);
  ncclResult_t (*iallreduce)(...);    // 直接提交 AllReduce 到 SHARP!
  ncclResult_t (*iflush)(...);
  ncclResult_t (*test)(...);
  ...
} ncclCollNet_v8_t;
```

### 与 ncclNet 的区别


| 对比     | ncclNet               | ncclCollNet       |
| ------ | --------------------- | ----------------- |
| 抽象级别   | 点对点 (send/recv)       | 集合操作 (iallreduce) |
| 网络行为   | 应用自己编排 Ring/Tree 通信模式 | 交换机硬件执行整个集合操作     |
| 需要硬件支持 | 任意网卡                  | 仅 SHARP 交换机       |


## 12.3 NVLS — 节点内的"SHARP"

**NVLS (NVLink SHARP)** 是 NVLink 4.0 / NVSwitch 引入的类似能力：

```
传统节点内 AllReduce:
  GPU0 → NVLink → GPU1 → NVLink → GPU2 → ... (Ring)

NVLS AllReduce:
  GPU0 ──→ NVSwitch ←── GPU1
  GPU2 ──→ NVSwitch ←── GPU3
              │
         NVSwitch 内部完成 Reduce
              │
  GPU0 ←── result ──→ GPU1
  GPU2 ←── result ──→ GPU3
```

环境变量 `NCCL_NVLS_ENABLE=1` 启用。

---

# 第十三章：性能模型 — 网络架构师的分析框架

## 13.1 基本性能公式

```
集合通信时间 = 延迟项 + 带宽项

Ring AllReduce:
  T = 2(N-1) × α + 2(N-1)/N × S/BW
  ≈ 2N × α + 2S/BW     (N 大时)

  α = 单步延迟 (网络 RTT + GPU-CPU 交互)
  S = 数据总量
  BW = 单链路带宽 (最慢的那段)
  N = GPU 数量

Tree AllReduce:
  T = 2log₂(N) × α + 2log₂(N) × S/BW
```

## 13.2 瓶颈分析框架

```
性能不及预期时, 按顺序排查:

1. 带宽瓶颈?
   → 实际带宽 vs 理论带宽 (nccl-tests 输出的 busBw)
   → busBw < 理论带宽的 80%: 网络可能有问题

2. 延迟瓶颈?
   → 小消息 (<1KB) 的延迟是否合理
   → 延迟 > 50μs: 检查是否走了 Socket 而非 RDMA

3. 算法瓶颈?
   → 强制 NCCL_ALGO=RING 和 TREE 分别测试
   → 对比选择是否合理

4. 协议瓶颈?
   → 强制 NCCL_PROTO=SIMPLE/LL 分别测试
   → 小消息用 Simple 会慢很多

5. GDR 是否生效?
   → NCCL_DEBUG 日志中查找 "GDR" 字样
   → 无 GDR 时延迟明显偏高

6. 多轨是否均衡?
   → 检查每个 NIC 的流量计数器
   → 是否所有 NIC 都在工作
```

## 13.3 关键性能数字（心中有数）


| 场景                  | 期望 busBw  | 备注             |
| ------------------- | --------- | -------------- |
| 节点内 NVLink (8×A100) | ~240 GB/s | NVSwitch 全互联   |
| 节点内 PCIe P2P        | ~25 GB/s  | PCIe 4.0 x16   |
| 跨机 IB HDR (200Gbps) | ~24 GB/s  | 单卡             |
| 跨机 IB NDR (400Gbps) | ~48 GB/s  | 单卡             |
| 跨机 4×HDR Rail       | ~90 GB/s  | 4 卡聚合          |
| 跨机 RoCE 100Gbps     | ~11 GB/s  | 受 PFC/ECN 影响更大 |


---

# 第十四章：网络问题诊断方法论

## 14.1 三板斧

### 第一斧：NCCL_DEBUG 日志

```bash
# 最常用: 看初始化过程和拓扑选择
NCCL_DEBUG=INFO ./your_app

# 深入网络层: 看每次 send/recv
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=NET ./your_app

# 看拓扑发现和路径规划
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH ./your_app
```

### 第二斧：nccl-tests 基准

```bash
# AllReduce 性能扫描 (8B ~ 1GB, 2倍递增)
./all_reduce_perf -b 8 -e 1G -f 2 -g 8

# 关键输出列:
#   busBw: 算法带宽 (已除以算法系数, 可直接和物理带宽对比)
#   algBw: 原始算法带宽

# 对比实验:
NCCL_P2P_DISABLE=1 ./all_reduce_perf ...    # 禁用 P2P, 看纯网络性能
NCCL_NET_GDR_LEVEL=0 ./all_reduce_perf ...   # 禁用 GDR, 看 GDR 的收益
NCCL_ALGO=RING ./all_reduce_perf ...         # 强制 Ring
NCCL_ALGO=TREE ./all_reduce_perf ...         # 强制 Tree
```

### 第三斧：系统级工具

```bash
# IB 链路状态
ibstatus

# IB 端口流量计数器
perfquery -x mlx5_0 1

# PCIe 带宽监控
nvidia-smi topo -m         # GPU-NIC 拓扑矩阵
nvidia-smi nvlink -s       # NVLink 状态

# RDMA 基准对比
ib_write_bw -d mlx5_0      # 原始 RDMA 带宽, 和 NCCL 结果对比
```

## 14.2 常见问题模式


| 现象             | 可能原因              | 排查方法                             |
| -------------- | ----------------- | -------------------------------- |
| busBw 远低于理论值   | GDR 未生效           | NCCL_DEBUG 查 "GDR"               |
| 小消息延迟异常高       | 走了 Socket 而非 RDMA | 检查 NCCL_DEBUG 中的 transport 类型    |
| 多节点性能不随节点数线性增长 | 网络拓扑瓶颈            | 检查交换机是否过载                        |
| 性能抖动大          | PFC 风暴 / ECMP 不均衡 | 检查交换机 PFC 计数器                    |
| 某些节点明显慢        | NIC 亲和性错误         | 检查 NCCL 选择的 NIC vs GPU 的 PCIe 拓扑 |


---

# 第十五章：关键环境变量速查

## 调试类


| 变量                  | 作用      | 常用值                         |
| ------------------- | ------- | --------------------------- |
| `NCCL_DEBUG`        | 日志级别    | INFO / WARN / TRACE         |
| `NCCL_DEBUG_SUBSYS` | 子系统过滤   | INIT, NET, GRAPH, COLL, ALL |
| `NCCL_DEBUG_FILE`   | 日志输出到文件 | /tmp/nccl_%h_%p.log         |


## 算法和协议


| 变量           | 作用   | 常用值                   |
| ------------ | ---- | --------------------- |
| `NCCL_ALGO`  | 强制算法 | RING / TREE / COLLNET |
| `NCCL_PROTO` | 强制协议 | SIMPLE / LL / LL128   |


## 网络相关 ★


| 变量                            | 作用                | 常用值                        |
| ----------------------------- | ----------------- | -------------------------- |
| `NCCL_SOCKET_IFNAME`          | Bootstrap 使用的网络接口 | eth0, ib0, ^docker0 (排除)   |
| `NCCL_IB_HCA`                 | 指定 IB 网卡设备        | mlx5_0, mlx5_0,mlx5_1 (多卡) |
| `NCCL_IB_GID_INDEX`           | RoCE 的 GID 索引     | 通常 3 (RoCEv2)              |
| `NCCL_IB_QPS_PER_CONNECTION`  | 每连接 QP 数          | 默认 1                       |
| `NCCL_NET_GDR_LEVEL`          | GDR 使用级别          | 0(关闭)~5(最激进)               |
| `NCCL_NET_GDR_READ`           | 允许 NIC 从 GPU 读数据  | 0/1                        |
| `NCCL_NCHANNELS_PER_NET_PEER` | 每对节点 Channel 数    | 默认自动                       |


## P2P 与 SHM


| 变量                 | 作用         | 常用值             |
| ------------------ | ---------- | --------------- |
| `NCCL_P2P_DISABLE` | 禁用 GPU P2P | 0/1             |
| `NCCL_P2P_LEVEL`   | P2P 使用级别   | NVL/PIX/PXB/PHB |
| `NCCL_SHM_DISABLE` | 禁用共享内存传输   | 0/1             |


## 插件


| 变量                  | 作用              | 常用值          |
| ------------------- | --------------- | ------------ |
| `NCCL_NET_PLUGIN`   | 指定网络插件          | ofi, ib, ... |
| `NCCL_TUNER_PLUGIN` | 指定调优插件          | 自定义 .so 路径   |
| `NCCL_NVLS_ENABLE`  | 启用 NVLink SHARP | 0/1          |


---

# 第十六章：学习路线 — RDMA 开发者专属

> **核心原则**：你已经懂 DMA、MR、QP、CQ。不需要从 CUDA Hello World 开始。
> 直接从网络路径切入，以你最熟悉的 RDMA 知识为锚点展开。

## 环境需求总览

### 各阶段硬件需求

```
阶段        ┃ 最低要求                        ┃ 理想配置
━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一 黑盒观测  ┃ 单机 2×GPU (PCIe 即可)          ┃ 2+节点, 每节点 4~8 GPU + RDMA NIC
二 源码阅读  ┃ 任意笔记本 (纯读代码)            ┃ 同阶段一的环境 (可实验验证)
三 写 Plugin ┃ 单机 2×GPU + 编译环境            ┃ 2 节点 + RDMA NIC (跑 RDMA 版本)
四 GDR 内核  ┃ 1×GPU + 1×RDMA NIC + root 权限  ┃ 同左 (硬件无法替代)
五 性能调优  ┃ 2+节点 RDMA 集群                ┃ 8+节点生产级集群
```

### 硬件选型建议

**GPU**：

- 入门够用：任意 NVIDIA GPU (RTX 3090/4090, A30, A100)，关键是 **数量 ≥ 2**
- NVLink 是加分项但非必须——阶段一用 `NCCL_P2P_DISABLE=1` 强制走网络，PCIe 直连也能学到核心内容
- 建议优先关注 Compute Capability ≥ 7.0 (Volta+)，因为 LL128 协议依赖此特性

**RDMA NIC**：

- 强烈建议 ConnectX-5 及以上 (cx5/cx6/cx7)，NCCL 默认走的是 Mellanox 驱动栈
- ConnectX-4 可用但缺少部分 GDR 优化特性
- RoCEv2 和 InfiniBand 均可；IB 更省心（不需要折腾 PFC/ECN），RoCE 更贴近生产环境
- 如果你手上只有 RoCE 环境，确保交换机已配好 PFC (Priority Flow Control)，否则丢包会导致 NCCL hang

**拓扑注意**：

- GPU 和 NIC 尽量在同一个 PCIe Root Complex / NUMA Node 下——这直接影响 GDR 能否启用
- 验证命令：`nvidia-smi topo -m` 查看 GPU-NIC 亲和关系，`PIX`/`PXB` 表示同 Root Complex

### 软件栈 (所有阶段通用)


| 组件             | 版本建议                | 安装方式                       | 说明                                 |
| -------------- | ------------------- | -------------------------- | ---------------------------------- |
| NVIDIA Driver  | ≥ 525.x             | 官网 .run 或 apt/yum          | 需要支持 CUDA 12.x                     |
| CUDA Toolkit   | 12.x                | apt/yum 或 runfile          | nvcc 编译 NCCL 和 nccl-tests          |
| MLNX_OFED      | ≥ 5.8 (推荐 24.x LTS) | Mellanox ISO 安装            | 包含 libibverbs, rdma-core, perftest |
| NCCL           | ≥ 2.18 (推荐源码编译)     | `make -j src.build`        | 源码编译方便调试、改代码                       |
| nccl-tests     | 最新 main 分支          | `make MPI=1 NCCL_HOME=...` | 带 MPI 可跑多节点测试                      |
| nvidia-peermem | 内核模块                | MLNX_OFED 自带 或 独立编译        | 阶段四必需，`modprobe nvidia-peermem`    |
| OpenMPI / PMIx | ≥ 4.x               | apt/yum 或源码                | 多节点启动 nccl-tests 用                 |


**验证命令清单** (装完后跑一遍确认环境正常)：

```bash
# GPU 状态
nvidia-smi                              # 能看到所有 GPU
nvidia-smi topo -m                      # 拓扑矩阵: GPU 之间、GPU-NIC 关系

# RDMA 状态
ibstatus                                # 端口 State: Active, Rate: 100/200 Gb/sec
ibv_devinfo                             # 设备详情: fw_ver, phys_port_cnt
ib_write_bw -d mlx5_0                   # 单边带宽基线 (双机各跑一端)

# CUDA + NCCL
nvcc --version                          # CUDA 版本
ls /usr/lib/x86_64-linux-gnu/libnccl*   # NCCL 库是否存在 (路径因系统而异)

# GDR 验证
lsmod | grep nvidia_peermem             # 模块是否加载
cat /sys/kernel/mm/memory_peers/        # peermem 注册状态 (内核 5.x+)

# nccl-tests 快速验证 (单机 2 GPU)
./build/all_reduce_perf -b 8 -e 256M -f 2 -g 2
```

### 没有硬件？替代方案


| 方案                        | 适合阶段   | 成本    | 说明                      |
| ------------------------- | ------ | ----- | ----------------------- |
| **云厂商 GPU 实例**            | 一~五    | 按小时计费 | 见下方推荐                   |
| **纯源码阅读**                 | 二      | 免费    | 笔记本 + IDE 足够            |
| **单 GPU + Socket Plugin** | 三 (部分) | 低     | 用 loopback 跑通 Plugin 逻辑 |


**云实例推荐** (按性价比排序)：

```
┌──────────────────────┬──────────┬─────────────┬────────────────────────────┐
│ 实例类型             │ GPU      │ 网络        │ 适合阶段                   │
├──────────────────────┼──────────┼─────────────┼────────────────────────────┤
│ AWS p3.8xlarge       │ 4×V100   │ 25G ENA     │ 一、二 (入门够用)          │
│ AWS p4d.24xlarge     │ 8×A100   │ 4×100G EFA  │ 一~五 (全阶段理想环境)     │
│ AWS p5.48xlarge      │ 8×H100   │ 32×100G EFA │ 五 (生产级调优)            │
│ Azure ND96amsr_v4    │ 8×A100   │ 8×200G IB   │ 一~五 (IB 环境首选)        │
│ GCP a2-highgpu-4g    │ 4×A100   │ 100G gVNIC  │ 一、二、三                 │
│ AutoDL / 矩池云      │ 按需选   │ 共享网络    │ 一、二 (国内低成本入门)     │
└──────────────────────┴──────────┴─────────────┴────────────────────────────┘

提示:
  - 阶段一、二: 2~4×GPU 的按小时实例即可，用完释放，成本可控
  - 阶段三 Socket Plugin: 单机 2×GPU 即可，最便宜的多 GPU 实例就行
  - 阶段四、五: 需要 RDMA 网络 (EFA/IB)，只有 p4d+/ND96+ 等高端实例才有
  - 国内用户: AutoDL、矩池云有 A100/V100 实例，但网络通常不支持 RDMA，
    适合阶段一~三，阶段四、五需要自建或找有 IB 网络的机房
```

### 环境搭建检查清单

按顺序逐项确认，**全部打勾后再开始阶段一**：

- `nvidia-smi` 能看到 ≥ 2 块 GPU，驱动版本 ≥ 525
- `nvcc --version` 返回 CUDA 12.x
- `ibstatus` 显示端口 Active (如果有 RDMA NIC)
- NCCL 源码已 clone 并编译成功 (`make -j src.build CUDA_HOME=/usr/local/cuda`)
- nccl-tests 编译通过，`all_reduce_perf -g 2` 能跑出结果
- (可选) `nvidia-peermem` 模块已加载 (阶段四前做)
- (可选) OpenMPI 安装完成，`mpirun --version` 正常 (多节点前做)

---

## 阶段一：黑盒观测（1 周）

**目标**：看到 NCCL 的网络行为，建立直觉。

### 动手任务

1. 编译 [nccl-tests](https://github.com/NVIDIA/nccl-tests)
2. 运行 `NCCL_DEBUG=INFO ./all_reduce_perf -b 8 -e 1G -f 2 -g N`
3. 对比实验矩阵：


| 实验      | 环境变量                 | 观察什么              |
| ------- | -------------------- | ----------------- |
| 基线      | 默认                   | busBw 数字、选择的算法/协议 |
| 禁用 P2P  | NCCL_P2P_DISABLE=1   | 强制走网络，busBw 变化    |
| 禁用 GDR  | NCCL_NET_GDR_LEVEL=0 | GDR 的收益           |
| 强制 Ring | NCCL_ALGO=RING       | 大小消息的性能差异         |
| 强制 Tree | NCCL_ALGO=TREE       | 和 Ring 的交叉点在哪里    |
| LL 协议   | NCCL_PROTO=LL        | 小消息延迟改善           |


### 验证标准

- 能看懂 NCCL_DEBUG 输出中的拓扑信息（PIX/NVL/NET 等）
- 能根据 busBw 判断当前走的是 NVLink、PCIe 还是 RDMA
- 能用 `ib_write_bw` 结果和 NCCL 结果做对比，解释差异

## 阶段二：源码阅读 — 网络路径优先（3 周）

**目标**：白盒理解 NCCL 如何使用 RDMA。从你最熟悉的 `ibv_post_send` 往上追溯。

### 准备工作（Day 0）

```bash
# 1. 克隆源码，建议 checkout 到一个稳定的 release tag
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git tag -l 'v2.*' | sort -V | tail -5     # 看有哪些版本
git checkout v2.21.5-1                      # 选一个较新的稳定版

# 2. 先摸清源码规模，心里有数
find src -name '*.cc' -o -name '*.h' | wc -l          # 文件数 (~100+)
wc -l src/transport/net_ib.cc                          # IB 层代码行数
wc -l src/proxy.cc                                     # Proxy 层代码行数

# 3. 生成全局索引 (任选一个)
#    方式 A: ctags (轻量，终端跳转)
ctags -R src/
#    方式 B: compile_commands.json (用于 VSCode/CLion 的精确跳转)
bear -- make -j src.build CUDA_HOME=/usr/local/cuda 2>/dev/null
#    方式 C: 直接用 VSCode + clangd 插件打开 src/ 目录
```

> **方法论**：读大型 C 项目不要从 main() 开始通读——你会迷失在初始化的几十个分支里。
> 对 RDMA 开发者来说，正确的入口是 **你认识的 Verbs API 调用**。
> 从 `ibv_post_send` 开始往上追，每一层都能和你已有的知识对接。

### Week 1：传输层 — 从你最熟悉的 Verbs 开始（5 天）

这一周只看 **一个文件**：`src/transport/net_ib.cc`。不要贪多。

#### Day 1-2：找到 ibv_post_send，理解 NCCL 如何发数据

```bash
# 第一步：在整个源码树里搜 ibv_post_send，确认入口
grep -rn 'ibv_post_send' src/
# 预期：只在 net_ib.cc 中出现，被 ncclIbIsend 包装

# 第二步：找到 ncclIbIsend 函数定义
grep -n 'ncclIbIsend' src/transport/net_ib.cc
# 打开这个函数，从上往下读，回答以下问题：
```

**带着问题读代码** (逐个回答，写在你的笔记里)：

```
Q1: ibv_post_send 的 opcode 是什么？IBV_WR_RDMA_WRITE 还是 IBV_WR_SEND？
    → 你会发现是 RDMA WRITE——思考为什么不用 SEND

Q2: wr.sg_list 指向的地址是 GPU 显存还是 Host 内存？
    → 追踪 lkey 的来源，找到 MR 是怎么注册的

Q3: 每次 isend 发多大的数据？谁决定切分粒度？
    → 找 size 参数的来源，往上层追

Q4: remote_addr 和 rkey 从哪来？什么时候交换的？
    → 这就是 NCCL 的 "连接建立" 过程，记住入口，Week 3 再深入
```

#### Day 3：追踪 MR 注册和缓存

```bash
# 找 MR 相关函数
grep -n 'ibv_reg_mr\|RegMr\|MrCache\|dereg' src/transport/net_ib.cc
```

**带着问题读代码**：

```
Q5: ibv_reg_mr 在哪被调用？传入的 addr 是 GPU 地址还是 CPU 地址？
    → GPU 地址 + GDR: 走 nvidia-peermem 路径
    → CPU 地址: 普通注册

Q6: MR Cache 用什么数据结构？Key 是什么？
    → 找 cache lookup 逻辑：通常用 (addr, size) 做 key
    → 对比你写 RDMA 应用时怎么管理 MR

Q7: MR 什么时候被释放？有淘汰机制吗？
    → 找 deregMr / cache eviction 逻辑
    → 思考：如果 PyTorch 释放了 tensor 但 MR Cache 里还有旧条目，会怎样？
```

#### Day 4-5：QP 创建和连接建立

```bash
# 找 QP 相关函数
grep -n 'ibv_create_qp\|ibv_modify_qp\|IBV_QPS_' src/transport/net_ib.cc
```

**带着问题读代码**：

```
Q8: QP type 是 RC 还是 UD？
    → 你会发现是 RC——和你写 RDMA 应用一样

Q9: QP 状态转换 INIT→RTR→RTS 在代码里怎么做的？
    → 找 ibv_modify_qp 的调用点和参数

Q10: 两端的 QPN、GID 是怎么交换的？走什么通道？
     → NCCL 用 Socket 做 out-of-band 交换 (类似你用 CM 做连接管理)
     → 找 ncclIbConnect / ncclIbAccept 中的 socket 操作
```

#### Week 1 产出物

读完后，你应该能画出这张图（手画在纸上就行）：

```
ncclSend(GPU_ptr, size)
  │
  ▼
ncclIbIsend()
  ├─ MR Cache 查找 → 命中: 复用 lkey
  │                 → 未命中: ibv_reg_mr (GDR 路径走 nvidia-peermem)
  ├─ 构造 ibv_send_wr
  │    opcode   = IBV_WR_RDMA_WRITE
  │    sg_list  = [{addr=GPU_ptr, length=chunk_size, lkey=cached_lkey}]
  │    wr.rdma  = {remote_addr=对端Scratch_Buffer, rkey=对端rkey}
  ├─ ibv_post_send(qp, &wr)
  │
  ▼
完成检测: ibv_poll_cq → 通知上层 test() 返回 done
```

### Week 2：代理层 — 理解 GPU 和网络之间的桥梁（5 天）

Week 1 你看到了 `ncclIbIsend` 被调用，但**谁调用它**？不是 GPU Kernel 直接调的——GPU
不能执行 Verbs。这就是 Proxy Thread 的角色。

#### Day 1-2：Proxy Thread 主循环

```bash
# 找 Proxy 主循环
grep -n 'proxyProgress\|proxyService\|pthread_create' src/proxy.cc | head -20
```

**带着问题读代码**：

```
Q11: Proxy Thread 是什么时候创建的？每个 GPU 一个还是全局一个？
     → 找 pthread_create 的调用点

Q12: Proxy 的主循环在做什么？
     → 你会看到一个 while 循环，反复调用 progress function
     → 本质上就是你写 RDMA 应用时的 "polling loop"

Q13: Proxy 怎么知道 GPU 有新任务要发？
     → 这就是 Mailbox 机制：GPU 更新 Head 计数器 → Proxy 轮询发现变化 → 发起网络 I/O
     → 对比你写 RDMA 应用时的工作提交方式 (ibv_post_send 是同步调用，这里多了一层异步)
```

#### Day 3-4：NET Send / Recv 的 Progress Function

这两个文件是 Proxy 调用链的下一层，连接 Proxy 主循环和 Week 1 看到的 IB Verbs：

```bash
# 找 progress function 的实现
grep -n 'netSendProxy\|netRecvProxy\|Progress' src/transport/net_send.cc
grep -n 'netSendProxy\|netRecvProxy\|Progress' src/transport/net_recv.cc
```

**理解状态机**：

```
Progress Function 是一个状态机，每次被 Proxy 主循环调用时推进一步：

  net_send.cc 的状态流转:
  ┌──────────┐    ┌───────────┐    ┌────────码阅读技巧──┐    ┌──────────┐
  │ 等待GPU  │ →  │ 拷贝到     │ →  │ 调用      │ →  │ 等待CQ   │
  │ 数据就绪  │    │ StagingBuf │    │ ncclNet   │    │ 完成通知  │
  │          │    │ (如需要)   │    │ isend()   │    │          │
  └──────────┘    └───────────┘    └──────────┘    └──────────┘
       ↑ Mailbox                                         │
       │              更新 Tail → GPU 可以送下一批         │
       └──────────────────────────────────────────────────┘

每个状态对应代码里的一个 case/if 分支。
带着这个状态机的预期去读代码，你会清晰很多。
```

**核心问题**：

```
Q14: 什么情况下需要 Staging Buffer 中转？什么情况下 GPU 显存直接 DMA？
     → 有 GDR: GPU_ptr 直接作为 ibv_post_send 的源地址
     → 无 GDR: 先 cudaMemcpy 到 Host Staging Buffer，再 post_send
     → 这就是 GDR 的核心收益——省掉一次 Host 内存拷贝

Q15: Tail 计数器更新后，GPU 端怎么感知到？
     → 回到第六章：volatile 轮询 + 内存屏障
```

#### Day 5：串联 — 画出完整调用链

把 Week 1 和 Week 2 拼起来：

```
   GPU Kernel (Persistent)                Proxy Thread (CPU)
   ═══════════════════════               ═══════════════════
1. 从 FIFO 读到 ncclWork
2. 把用户数据搬到 Scratch Buffer
   (通过 NVLink/PCIe)
3. 更新 Head 计数器 ──────────→  4. 轮询发现 Head 变化
                                 5. 调用 progress function
                                 6. 读 Scratch Buffer 内容
                                 7. 调用 ncclIbIsend()
                                    → ibv_post_send()
                                 8. ibv_poll_cq() 等完成
                                 9. 更新 Tail ──────────→  10. GPU 感知到 Tail 变化
                                                               可以复用 Scratch Buffer
```

> **RDMA 类比**：这个 GPU → Proxy → NIC 的三级流水线，
> 本质上和高性能 RDMA 应用里的 "应用线程填 Buffer → I/O 线程 post_send → CQ 线程 poll" 一样。
> NCCL 只是把 "应用线程" 换成了 GPU Kernel。

### Week 3：拓扑和初始化 — 理解全局蓝图（5 天）

前两周你看清了 "一个 Channel 上一次 Send 怎么走"。这周看**全局**：
NCCL 初始化时怎么发现拓扑、规划路径、建立连接。

#### Day 1-2：拓扑发现

```bash
# 入口函数
grep -n 'ncclTopoGetSystem\|ncclTopoCompute' src/graph/topo.cc | head -20

# NCCL 探测哪些系统信息
grep -rn '/sys/class\|/proc/\|nvidia-smi\|ibv_get' src/graph/topo*.cc
```

**带着问题读代码**：

```
Q16: NCCL 怎么发现 GPU 和 NIC 的亲和关系？
     → 它读 sysfs: /sys/class/infiniband/mlx5_X/device/numa_node
     → 和 /sys/bus/pci/devices/XXXX:XX:XX.X/ 下的拓扑信息
     → 对比 nvidia-smi topo -m 的输出，你会发现信息来源一致

Q17: 拓扑图在内存里是什么数据结构？
     → 找 ncclTopoSystem / ncclTopoNode / ncclTopoLink
     → 本质是一个带权重的有向图：节点 = GPU/NIC/CPU/PCI Switch，边 = 链路 + 带宽
```

#### Day 3-4：路径规划（Search）

```bash
grep -n 'ncclTopoCompute\|ncclTopoSearch\|Ring\|Tree' src/graph/search.cc | head -30
```

**带着问题读代码**：

```
Q18: Ring 路径是怎么规划的？
     → 在拓扑图上找一个 Hamilton 回路，使得最慢链路带宽最大化
     → 类似网络中的 "最大带宽路径" 问题

Q19: 什么时候选 Ring，什么时候选 Tree？
     → 找决策逻辑：通常看消息大小和 GPU 数量
     → 回到第三章的分析框架验证你的理解
```

#### Day 5：初始化全流程

```bash
grep -n 'ncclCommInitRank\|bootstrapInit\|transportSetup\|Connect' src/init.cc | head -30
```

用时序图整理 `ncclCommInitRank` 的完整过程：

```
ncclCommInitRank(comm, nranks, id, rank)
  │
  ├─ 1. Bootstrap ─────────── 所有 rank 通过 TCP 互联，交换基础信息
  │     (类似 MPI_Init 的握手过程)
  │
  ├─ 2. Topo Discovery ────── 每个 rank 探测本地 GPU/NIC/PCIe 拓扑
  │     ncclTopoGetSystem()
  │
  ├─ 3. Graph Search ──────── 在拓扑图上规划 Ring/Tree 路径
  │     ncclTopoCompute()
  │     决定: 每个 Channel 上谁发给谁
  │
  ├─ 4. Transport Setup ───── 为每条路径选择传输方式并建立连接
  │     ├─ 同 GPU:  无需传输
  │     ├─ 同机 NVLink: P2P transport → cuMemcpy 直通
  │     ├─ 同机无 NVLink: SHM transport → 走共享内存 + cudaMemcpy
  │     └─ 跨机: NET transport → 走到你 Week 1 看的 ncclIbConnect()
  │           创建 QP, 交换 QPN/GID, INIT→RTR→RTS
  │
  ├─ 5. Proxy Setup ──────── 为 NET transport 创建 Proxy Thread
  │     分配 Scratch Buffer, 注册 MR
  │
  └─ 6. Kernel Launch ─────── 启动 Persistent Kernel
        GPU Kernel 开始轮询 FIFO，等待第一个 ncclWork
```

### 源码阅读技巧（通用）


| 技巧               | 做法                                          | 为什么有效           |
| ---------------- | ------------------------------------------- | --------------- |
| **Verbs 锚点法**    | 搜 `ibv_`* 调用，从你认识的 API 往上追                  | RDMA 开发者的知识直接复用 |
| **日志辅助法**        | `NCCL_DEBUG=TRACE` 跑测试，对照日志和代码              | 日志行号直接定位到源码位置   |
| **单 Channel 简化** | `NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1` | 去掉并行干扰，只看一条路径   |
| **数据结构优先**       | 先读 `src/include/` 下的头文件里的结构体定义              | 理解数据结构后代码自然清晰   |
| **二分缩范围**        | 不确定走哪个分支？加 `printf` 或看 TRACE 日志             | 比猜测快 10 倍       |


```bash
# 实用 grep 命令模板
grep -rn 'ibv_post_send\|ibv_post_recv' src/         # 找所有 Verbs 数据面调用
grep -rn 'ibv_reg_mr\|ibv_dereg_mr' src/              # 找 MR 生命周期
grep -rn 'ibv_create_qp\|ibv_modify_qp' src/          # 找 QP 管理
grep -rn 'NCCL_NET_\|ncclNet_v' src/include/           # 找 Plugin 接口定义
grep -rn 'proxyProgress\|proxyService' src/            # 找 Proxy 逻辑入口

# 日志辅助: 跑一次 AllReduce，只开 1 个 Channel，输出到文件慢慢看
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=NET \
NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 \
./build/all_reduce_perf -b 1M -e 1M -g 2 2>&1 | tee nccl_trace.log

# 在日志里搜关键事件
grep 'ibv_post_send\|posted\|completed\|connect\|QP' nccl_trace.log
```

### 验证标准

完成三周后，你应该能独立回答以下问题（建议写成笔记）：

- 画出 `ncclAllReduce()` → GPU Kernel → Proxy → `ibv_post_send` → CQ → Tail 更新 的完整时序图
- 解释为什么 NCCL 用 RDMA WRITE 而非 SEND（提示：WRITE 不消耗对端 Recv WR，适合单边通信模型）
- 解释 MR Cache 的查找、插入、淘汰逻辑，以及 GPU 显存被释放后 stale MR 的处理
- 解释没有 GDR 时数据多走了哪一步拷贝，性能差多少（对比阶段一的实验数据）
- 说出 `ncclCommInitRank` 的 6 个阶段，每个阶段的输入和输出是什么

## 阶段三：写一个简单的 NCCL Net Plugin（2 周）

**目标**：通过实现来深入理解。这是你作为网络开发者能做的最有价值的练习。

### 方案

基于 POSIX Socket 写一个最简 ncclNet 插件（不需要 RDMA，先跑通逻辑）：

```
实现 ncclNet_v8 接口:
  init        → socket()
  devices     → return 1
  listen      → bind + listen
  connect     → socket + connect
  accept      → accept()
  regMr       → no-op (Socket 不需要 MR)
  isend       → non-blocking send() + 维护状态机
  irecv       → non-blocking recv() + 维护状态机
  test        → poll/select 检查完成
  close*      → close()
```

### 进阶

如果有 RDMA 硬件，把 Socket 替换为 ibv_post_send/ibv_post_recv。

### 验证标准

- 插件能被 NCCL 加载并完成 AllReduce
- 能通过 nccl-tests 跑出结果（性能不重要，跑通为主）
- 能解释插件中每个回调的调用时序和生命周期

## 阶段四：GPUDirect RDMA 内核层（2 周）

**目标**：攻克 nvidia-peermem，打通从用户态到内核态的完整链路。

### 动手任务

1. 阅读 nvidia-peermem 内核模块源码
2. 找到 `ib_register_peer_memory_client` 的 hook 点
3. 追踪 `nvidia_p2p_get_pages` → BAR 地址映射的完整路径
4. 理解 mmu_notifier 如何在 GPU 显存释放时撤销 MR

### 验证标准

- 能画出完整时序图：ncclSend(GPU_Ptr) → IB Driver → PeerMem → GPU Driver → DMA
- 能解释 BAR 空间不够时的降级行为
- 能解释为什么 iflush 在 GDR 模式下是必需的

## 阶段五：性能调优实战（持续）

**目标**：在真实集群上定位和解决 NCCL 网络性能问题。

### 典型场景


| 场景                       | 方法                 |
| ------------------------ | ------------------ |
| 跨机 AllReduce 带宽只有理论值 60% | 用诊断方法论排查（第十四章）     |
| 训练 step time 抖动大         | 检查 PFC 风暴、ECN 标记率  |
| 加节点后性能不线性增长              | 检查 Ring 路径是否经过了慢链路 |


---

# 第十七章：关键参考资料

## 必看视频

- [NCCL: High-Speed Inter-GPU Communication (GTC) - Sylvain Jeaugey](https://www.youtube.com/watch?v=Dk8XH0nSccA)
  - **14:30** Ring 动画 | **25:00** Tree 动画 | **35:00** Channel 和 Multi-rail
- [Demystifying NCCL (Hot Interconnects 2023)](https://www.youtube.com/watch?v=M2SPJ1uA-Eg)
  - Protocol 时序图 | GDR 路径分析 | 性能模型

## 核心文档

- [NCCL Developer Guide (PDF)](https://docs.nvidia.com/deeplearning/nccl/assets/nccl-developer-guide.pdf)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL Network Plugin API](https://github.com/NVIDIA/nccl/blob/master/src/include/nccl_net.h) — 插件接口定义

## 源码

- [NVIDIA/nccl (GitHub)](https://github.com/NVIDIA/nccl) — 主仓库
- [NVIDIA/nccl-tests (GitHub)](https://github.com/NVIDIA/nccl-tests) — 性能测试工具
- [aws-ofi-nccl (GitHub)](https://github.com/aws/aws-ofi-nccl) — ★ 插件接口的最佳学习材料
- [nvidia-peermem (GitHub)](https://github.com/Mellanox/nvidia-peermem) — GDR 内核模块

## 深入阅读

- [Massively Scale with NCCL 2.4 (Blog)](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)
- [CUDA C++ Programming Guide - 3.2.4 Page-Locked Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [InfiniBand SHARP Technology](https://docs.nvidia.com/networking/display/shaborev270/) — SHARP 官方文档

---

# 学习进度跟踪


| 章节                 | 状态    | 完成日期 | 笔记  |
| ------------------ | ----- | ---- | --- |
| 第一章：NCCL 是什么       | ⬜ 待开始 |      |     |
| 第二章：核心概念           | ⬜ 待开始 |      |     |
| 第三章：通信算法           | ⬜ 待开始 |      |     |
| 第四章：软件架构           | ⬜ 待开始 |      |     |
| 第五章：ncclNet 插件接口 ★ | ⬜ 待开始 |      |     |
| 第六章：数据面详解          | ⬜ 待开始 |      |     |
| 第七章：控制面详解          | ⬜ 待开始 |      |     |
| 第八章：通信协议           | ⬜ 待开始 |      |     |
| 第九章：GPUDirect RDMA | ⬜ 待开始 |      |     |
| 第十章：多轨网络           | ⬜ 待开始 |      |     |
| 第十一章：拓扑发现          | ⬜ 待开始 |      |     |
| 第十二章：SHARP 与网内计算   | ⬜ 待开始 |      |     |
| 第十三章：性能模型          | ⬜ 待开始 |      |     |
| 第十四章：问题诊断          | ⬜ 待开始 |      |     |
| 第十五章：环境变量速查        | ⬜ 待开始 |      |     |
| 阶段一实战：黑盒观测         | ⬜ 待开始 |      |     |
| 阶段二实战：源码阅读         | ⬜ 待开始 |      |     |
| 阶段三实战：写 Net Plugin | ⬜ 待开始 |      |     |
| 阶段四实战：GDR 内核层      | ⬜ 待开始 |      |     |
| 阶段五实战：性能调优         | ⬜ 待开始 |      |     |


