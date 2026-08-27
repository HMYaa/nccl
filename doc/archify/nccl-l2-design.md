# NCCL L2 组件设计说明

## 1. 文档定位

本文档把现有 L1 中的三个 NCCL Rank Runtime 容器继续拆解为 L2 Component View：

1. Host NCCL Runtime
2. Proxy Progress Runtime
3. GPU Device Runtime

每个容器配套两种视图：

- Component View：回答内部有哪些稳定职责、状态归谁持有、组件之间交换什么。
- Sequence View：回答一个典型工作如何依次经过这些组件。

本文档不是函数目录，也不是把调用栈改画成方框。L2 组件表示一组具有稳定职责和状态边界的实现单元，函数名只作为源码入口。

源码基线：NCCL `v2.31.2-1`，提交 `7b83616df3ae082a1f32bb74c27458bfe8153a13`。

证据标记：

- `[源码事实]`：可以在固定提交的文件、结构或函数中定位。
- `[设计归纳]`：为了理解和维护，从多个源码单元抽象出的组件边界。
- `[待运行验证]`：必须通过日志、GPU/HCA 环境或性能实验才能确认。

## 2. 从 L1 到 L2

| L1 容器                | L2 要回答的问题                                            | 主要状态对象                                                              | 配套行为场景                                          |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------- |
| Host NCCL Runtime      | 请求如何从 API 语义变成可提交的执行计划                    | `ncclInfo`、`ncclTaskColl`、`ncclKernelPlanner`、`ncclKernelPlan` | communicator 初始化；AllReduce 入队、tuning 与 launch |
| Proxy Progress Runtime | Host 如何建立 transport 资源，并持续推进跨节点 operation   | proxy connection、posted ops、`ncclProxyArgs`、transport request        | setup/connect/register；send/recv progress            |
| GPU Device Runtime     | device work 如何进入 collective 算法和 protocol primitives | `ncclDevWorkColl`、work batch、connector/FIFO state                     | Ring + Simple AllReduce 的 chunk/slice/step 推进      |

严格 C4 边界如下：

```text
Host Runtime L2
    只拆 Host NCCL Runtime
    Proxy 和 GPU 作为外部容器

Proxy Runtime L2
    只拆 Proxy Progress Runtime
    Host、GPU connector、ncclNet/NIC 作为外部依赖

Device Runtime L2
    只拆 GPU Device Runtime
    Host launch、Proxy 和 peer/network path 作为外部依赖
```

## 3. Host NCCL Runtime

### 3.1 组件职责

| 组件                      | 职责                                                                | 输入                                         | 输出或持久状态                               |
| ------------------------- | ------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------- |
| API & Validation          | 接收 collective 参数，构造`ncclInfo` 并完成 communicator/参数检查 | 用户参数、comm、stream                       | 栈上`ncclInfo`                             |
| Group Lifecycle           | 管理隐式/显式 Group 嵌套与统一提交边界                              | 已验证请求、Group depth                      | Group 内待统一处理的 communicator 集合       |
| Task Admission            | 过滤特殊分支并把普通 collective 持久化                              | `ncclInfo`、device redop                   | `ncclTaskColl`                             |
| Planner & Task Queues     | 持有待调度任务并组织后续计划                                        | `ncclTaskColl`                             | sorter、task queues、plan queues             |
| Tuning & Cost Selection   | 在合法候选中选择 algorithm、protocol 和 channel/CTA 配置            | collective、消息大小、拓扑图、tuning context | 已选择的执行参数                             |
| Work & Plan Builder       | 将 Host task 编码为 GPU 可消费的 work，并按 kernel 划分计划         | 已调优 task                                  | `ncclDevWorkColl`、`ncclKernelPlan`      |
| Launch Coordinator        | 在 Group 收口后启动 Proxy 工作并提交 CUDA kernel                    | kernel plans、proxy ops、stream              | 已提交但不保证已完成的异步工作               |
| Communicator Init & State | 建立 rank 级持久状态、channel 和内存池                              | bootstrap 信息、设备环境                     | communicator 持久状态                        |
| Topology & Graph Builder  | 发现物理拓扑、计算路径并搜索可用 collective graph                   | GPU/NIC/PCI/NVLink 信息                      | `ncclTopoSystem`、paths、`ncclTopoGraph` |

这些组件是 `[设计归纳]`。其中的数据对象和函数入口是 `[源码事实]`。

### 3.2 两个时间尺度

Communicator 初始化期：

```text
ncclCommInitRank
→ ncclTopoGetSystem
→ ncclTopoComputePaths
→ ncclTopoCompute(Ring / Tree / ...)
→ 保存到 communicator
```

每次普通多 Rank AllReduce：

```text
用户参数
→ ncclInfo
→ ncclEnqueueCheck
→ taskAppend
→ collTaskAppend
→ ncclTaskColl
→ planner/collSorter
→ ncclPrepareTasks
→ ncclGetAlgoInfo
→ ncclDevWorkColl + ncclKernelPlan
→ Proxy start + CUDA launch
```

`[源码事实]` `ncclInfo` 是临时请求描述，`ncclTaskColl` 是 planner 持有的持久任务。持久化的主要原因是请求需要跨越 API 栈帧和 Group 收口点，不是因为每个网络进度状态都记录在 `ncclTaskColl` 中。

### 3.3 关键接口和不变量

```text
ncclInfo
    表达 API 请求语义
    生命周期短

ncclTaskColl
    表达 Host planner task
    由 communicator pool 分配并进入 sorter

ncclDevWorkColl
    表达 GPU collective work
    已包含 device 执行需要的算法/协议相关字段

ncclKernelPlan
    表达一次 kernel 提交批次
    可包含多个 device work batch
```

必须保持的边界：

- `[源码事实]` `ncclAllReduce()` 只提供 AllReduce 语义，不能证明最终选择 Ring。
- `[源码事实]` `collTaskAppend()` 成功只证明 task 已进入 planner，不证明 kernel 已启动。
- `[源码事实]` Host API 返回遵循 CUDA stream 异步提交语义，不证明 GPU 工作完成。
- `[设计归纳]` sendbuff/recvbuff 的安全复用边界属于 stream completion，而不是 `ncclInfo` 或 `ncclTaskColl` 的 C++ 生命周期。

### 3.4 源码锚点

- `src/collectives.cc:192` `ncclAllReduceConfigImpl`
- `src/include/info.h:17` `ncclInfo`
- `src/enqueue/enqueue.cc:3383` `ncclEnqueueCheck`
- `src/enqueue/enqueue.cc:3247` `taskAppend`
- `src/enqueue/enqueue.cc:2694` `collTaskAppend`
- `src/enqueue/enqueue.cc:2783` `ncclTaskCollSorterInsert`
- `src/include/comm.h:199` `ncclTaskColl`
- `src/include/comm.h:342` `ncclKernelPlan`
- `src/include/comm.h:469` `ncclKernelPlanner`
- `src/include/device.h:285` `ncclDevWorkColl`
- `src/enqueue/enqueue.cc:414` `ncclPrepareTasks`
- `src/enqueue/enqueue.cc:2097` `ncclGetAlgoInfo`
- `src/enqueue/enqueue.cc:1661` `ncclLaunchPrepare`
- `src/enqueue/enqueue.cc:1852` `ncclLaunchKernel`
- `src/graph/topo.cc:1814` `ncclTopoGetSystem`
- `src/graph/paths.cc:751` `ncclTopoComputePaths`
- `src/graph/search.cc:1105` `ncclTopoCompute`

## 4. Proxy Progress Runtime

### 4.1 两条执行轨道

Proxy 不能只画成一个“CPU Proxy 线程”。固定版本中至少存在两条职责不同的执行轨道。

控制轨道：

```text
Host transport setup
→ ncclProxyCallBlocking / ncclProxyCallAsync
→ Proxy Service Thread
→ setup / connect / register
→ transport resource handle
```

推进轨道：

```text
ncclProxySaveOp
→ ncclProxyPost
→ posted operation queue
→ ncclProxyGetPostedOps
→ progressOps
→ sendProxyProgress / recvProxyProgress
→ ncclNet isend / irecv / test
→ connector/FIFO progress
```

### 4.2 组件职责

| 组件                   | 职责                                    | 不负责什么                           |
| ---------------------- | --------------------------------------- | ------------------------------------ |
| Proxy Client API       | 提交 blocking/async 控制请求            | 不持续轮询 collective request        |
| Service Thread         | 分派连接和资源管理请求                  | 不作为 collective data progress loop |
| Connection & Resources | 建立 connection、MR 和 transport 资源   | 不选择 collective algorithm          |
| Operation Submission   | 保存并发布 Proxy operation              | 不表示 operation 已完成              |
| Posted Operation Queue | 在提交方与 progress thread 之间交接工作 | 不承载用户 payload 语义              |
| Progress Thread        | 获取 posted ops 并循环推进 active ops   | 不重新执行 Host tuning               |
| Active Op Scheduler    | 持有 operation 的阶段和状态             | 不拥有用户 buffer 的应用生命周期     |
| Transport Progress     | 调用 NET send/recv progress callback    | 不保证数据经过 CPU staging buffer    |

### 4.3 与 RDMA 的映射

```text
Proxy operation
→ transport progress callback
→ ncclNet request
→ net_ib backend
→ QP / WQE / CQ / MR
→ NIC
```

- `[源码事实]` Proxy progress callback 调用 ncclNet 接口并测试 request completion。
- `[设计归纳]` 从 RDMA 视角看，Proxy progress thread 更接近 software progress engine，而不是 memcpy worker。
- `[待运行验证]` 目标机器是否走 GDR、DMA 的具体源/目的地址以及 NIC rail 选择，不能由这张静态图证明。

### 4.4 源码锚点

- `src/proxy.cc:1341` `ncclProxyCallAsync`
- `src/proxy.cc:1436` `ncclProxyCallBlocking`
- `src/proxy.cc:591` `ncclProxySaveOp`
- `src/proxy.cc:476` `ncclProxyPost`
- `src/proxy.cc:830` `ncclProxyGetPostedOps`
- `src/proxy.cc:796` `progressOps`
- `src/proxy.cc:951` `ncclProxyProgress`
- `src/proxy.cc:1013` `ncclProxyStart`
- `src/transport/net.cc:1324` `sendProxyProgress`
- `src/transport/net.cc:1493` `recvProxyProgress`

## 5. GPU Device Runtime

### 5.1 组件层次

```text
Device Work Buffer
→ Kernel Entry & Loader
→ Work Batch Dispatcher
→ Collective Dispatcher
→ Algorithm Layer
→ Protocol Primitives
→ Connector/FIFO
→ Peer GPU 或 NET path
```

| 层次            | 当前示例          | 可替代实现                             |
| --------------- | ----------------- | -------------------------------------- |
| Collective 语义 | AllReduce         | Broadcast、AllGather、ReduceScatter 等 |
| Algorithm       | Ring              | Tree、NVLS、CollNet 等                 |
| Protocol        | Simple            | LL、LL128                              |
| Parallel lane   | 多 channel        | 由 Host planner/tuning 决定数量        |
| Transport path  | P2P/NET connector | SHM、P2P、NET 等连接类型               |

`[源码事实]` device kernel 消费 Host 已规划的 work。Device 端 specialization 执行给定的 collective、algorithm 和 protocol 组合，不在 `runRing()` 内重新运行 Host cost model。

### 5.2 Ring + Simple 主路径

```text
ncclKernelPlan
→ ncclDevWorkColl
→ ncclKernelMain
→ RunWorkBatch
→ RunWorkColl<AllReduce, Ring, Simple>
→ runRing
→ Simple Primitives
→ recvReduceSend
→ connector/FIFO
→ next peer
```

`recvReduceSend` 的物理模型：

```text
等待 credit / step 可用
→ 读取 peer 或 NET buffer
→ 读取 local input
→ 执行 reduction
→ 写入下一跳 buffer
→ 发布 step / tail
```

这里的 `step`、`slice`、`chunk` 是 pipeline 和流控粒度，不是新的 collective 语义层。

### 5.3 完成语义

- `[源码事实]` connector/FIFO state 用于 GPU primitive 与 peer/Proxy 路径之间的进度协调。
- `[设计归纳]` GPU kernel launch、primitive 完成、CUDA event/stream 完成是不同的观察点。
- `[待运行验证]` 某个 chunk 实际经 NVLink、PCIe P2P、GDR 或 staging path，必须结合 topology、NCCL_DEBUG 和 profiler 证明。

### 5.4 源码锚点

- `src/device/common.h:397` `ncclKernelMain`
- `src/device/common.h:305` `RunWorkBatch`
- `src/device/common.h:266` `RunWorkColl`
- `src/device/all_reduce.h:14` `runRing`
- `src/device/all_reduce.h:86` `runTreeUpDown`
- `src/device/all_reduce.h:229` Ring + Simple AllReduce specialization
- `src/device/all_reduce.h:763` Ring + LL AllReduce specialization
- `src/device/all_reduce.h:777` Ring + LL128 AllReduce specialization
- `src/device/prims_simple.h:989` Simple `recvReduceSend`
- `src/device/prims_ll.h:403` LL `recvReduceSend`
- `src/device/prims_ll128.h:422` LL128 `recvReduceSend`

## 6. 跨容器契约

| From               | To                 | 主要契约                                   | 完成边界                                |
| ------------------ | ------------------ | ------------------------------------------ | --------------------------------------- |
| Host Runtime       | GPU Device Runtime | CUDA launch 参数、device work、kernel plan | launch 返回不等于 stream 完成           |
| Host Runtime       | Proxy Runtime      | transport setup 请求、Proxy ops            | op 发布不等于 network completion        |
| GPU Device Runtime | Proxy Runtime      | connector/FIFO 共享进度状态                | state 更新只代表对应 step 可推进        |
| Proxy Runtime      | ncclNet Backend    | setup/connect/register、isend/irecv/test   | request test/completion 由 backend 定义 |
| GPU Device Runtime | Peer/Fabric        | peer buffer、local fabric 或 NET connector | 实际路径必须运行验证                    |

## 7. 非目标与后续扩展

本轮不展开：

- CE collective、RMA、GIN、CFT 和 enqueue rearchitecture。
- NVLS、CollNet、PAT 的内部 L3 设计。
- `net_ib` 内 QP/CQ/MR 资源模型和具体 WQE 状态机。
- 具体版本阈值和目标机器上的算法选择结论。
- 性能数值、GDR 启用状态和 NIC rail 映射。

后续进入 L3 时，建议一次只选择一个 L2 组件：

1. Host：`Task Admission + Planner` 的对象生命周期。
2. Proxy：`Active Op Scheduler + NET Progress` 的状态机。
3. Device：`Ring + Simple Primitives + Connector` 的 step/slice/chunk 数据流。

## 8. 视图索引

| 视图               | Component View                                                                                     | Sequence View                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Host Runtime       | [HTML](nccl-l2-host-runtime.architecture.html) · [JSON](nccl-l2-host-runtime.architecture.json)     | [HTML](nccl-l2-host-runtime.sequence.html) · [JSON](nccl-l2-host-runtime.sequence.json)     |
| Proxy Runtime      | [HTML](nccl-l2-proxy-runtime.architecture.html) · [JSON](nccl-l2-proxy-runtime.architecture.json)   | [HTML](nccl-l2-proxy-runtime.sequence.html) · [JSON](nccl-l2-proxy-runtime.sequence.json)   |
| GPU Device Runtime | [HTML](nccl-l2-device-runtime.architecture.html) · [JSON](nccl-l2-device-runtime.architecture.json) | [HTML](nccl-l2-device-runtime.sequence.html) · [JSON](nccl-l2-device-runtime.sequence.json) |

上层入口：

- [L0 System Context](nccl-l0-system-context.html)
- [L1 Rank Runtime](nccl-l1-control-data-plane.html)
- [AllReduce 总览序列](nccl-allreduce-sequence.html)

## 9. 设计验收问题

1. 为什么 communicator 初始化期生成 topology/graph，而每次 AllReduce 仍需 tuning？
2. 为什么 `ncclTaskColl` 已进入 planner 仍不能说明 GPU kernel 已启动？
3. Proxy Service Thread 和 Progress Thread 分别解决什么问题？
4. 为什么 Proxy progress 不等价于 CPU memcpy？
5. AllReduce、Ring、Simple、channel 和 NET 分别属于哪一层？
6. `recvReduceSend` 的 credit、数据和 completion 分别由谁提供或观察？
