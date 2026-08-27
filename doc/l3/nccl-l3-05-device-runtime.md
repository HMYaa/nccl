# NCCL L3-05：GPU Device Runtime

## 1. 文档范围

本卷描述 Host 已选定 collective、algorithm、protocol 和 channel 后，GPU kernel 如何
加载 work 并通过 primitives 推进数据。重点主线是 AllReduce + Ring + Simple。

## D1. Kernel Entry、Work Batch 加载与分发

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 从 kernel args/work buffer 加载 Host 规划结果，并分发到对应 collective specialization |
| 非目标 | Device 端不重新运行 Host cost model，不选择 NIC |
| 输入 | ncclDevKernelArgs、work batches、channel/block、specialized function id |
| 输出与停止边界 | 调用 RunWorkColl specialization；停止在每个 work batch 执行完 |
| 上下游接口 | 上游为 Host kernel launch；下游为 Ring/Tree algorithm implementation |
| 核心对象 | ncclDevKernelArgs、ncclDevWorkBatch、ncclDevWorkColl、ncclShmem |
| 主流程 | ncclKernelMain → load comm/channel → 遍历 work batch → RunWorkBatch → RunWorkColl |
| 状态与生命周期 | work 由 Host 编码并在 kernel 生命周期内消费；shared state 为 block/channel 局部 |
| 并发、同步与所有权 | CUDA block 通常对应 channel work；线程/warp 角色由 specialization 划分 |
| 异常与退出 | Device 路径主要依赖预先合法化 work；非法组合应在 Host 规划或 specialization 阶段避免 |
| 性能与可观测 | kernel grid、blocks/channels、warps、work batch 数；使用 Nsight Systems/Compute |
| 源码证据与遗留 | 源码事实：ncclKernelMain、RunWorkBatch、RunWorkColl。待运行：block/channel 实际映射 |

~~~plantuml
@startuml
title D1 Device Work 分发
rectangle "ncclKernelMain" as Main
rectangle "RunWorkBatch" as Batch
rectangle "RunWorkColl\n<Fn,T,RedOp,Algo,Proto>" as Coll
rectangle "ncclDevWorkColl" as Work
rectangle "ncclShmem\ncomm/channel/groups" as Shmem
Main --> Shmem : load shared state
Main --> Batch : iterate work batches
Work --> Batch
Batch --> Coll : specialization dispatch
@enduml
~~~

### 设计说明与源码锚点

Host 已把本次选择编码进 work 和 kernel specialization。RunWorkBatch 负责把线程分配给
sub-work，RunWorkColl 才进入具体 collective algorithm。语义层和执行模板层不能混淆。

- src/device/common.h：ncclKernelMain、RunWorkBatch、RunWorkColl
- src/device/common_kernel.h：kernel specialization 支撑
- src/include/device.h：ncclDevWorkColl、ncclDevKernelArgs

## D2. Ring AllReduce Device Schedule

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 在每个 channel 上按 Ring 顺序完成 ReduceScatter 和 AllGather 数据推进 |
| 非目标 | 不决定为什么 tuning 选择 Ring；不负责建立 peer connector |
| 输入 | ncclDevWorkColl、Ring prev/next、count/chunk、protocol specialization |
| 输出与停止边界 | recvbuff 获得该 channel 覆盖的数据结果；停止在 Ring 全部 loop 完成 |
| 上下游接口 | 上游为 D1；下游为 protocol Primitives |
| 核心对象 | channel ring userRanks、chunkCount、gridOffset、Primitives、work |
| 主流程 | 初始 send → p-2 次 recvReduceSend → recvReduceCopySend → p-2 次 recvCopySend → final recv |
| 状态与生命周期 | 每个 chunk 在 ReduceScatter 后拥有最终 reduce 结果，再沿 AllGather 传播 |
| 并发、同步与所有权 | 多 channel 处理不同数据区间；Ring step 通过 connector/FIFO 协调 |
| 异常与退出 | Device kernel 本身不返回 Host ncclResult；错误由异步 CUDA/comm 路径观察 |
| 性能与可观测 | 2(p-1) 逻辑阶段、chunk/slice pipeline、channel 并行和链路带宽 |
| 源码证据与遗留 | 源码事实：runRing 及 Ring+Simple/LL/LL128 specialization。待运行：实际 algo/channel |

~~~plantuml
@startuml
title D2 Ring AllReduce 一个 Channel
participant "Rank r local input" as Local
participant "Ring Primitive" as Prim
participant "Next Rank" as Next
group ReduceScatter
  Local -> Prim : initial chunk
  loop p-2 intermediate chunks
    Prim -> Prim : recvReduceSend
  end
  Prim -> Prim : recvReduceCopySend
end
group AllGather
  loop p-2 chunks
    Prim -> Next : recvCopySend
  end
  Prim -> Local : final recv/copy
end
@enduml
~~~

### 设计说明与源码锚点

Ring 的逻辑通信阶段是 p-1 个 ReduceScatter 加 p-1 个 AllGather。代码通过 chunk loop
和 primitive 调用表达流水，而不是显式写一个 2(p-1) 的简单循环。多 channel 会把
消息切分并并行推进，但不会改变 AllReduce 语义。

- src/device/all_reduce.h：runRing、Ring+Simple、Ring+LL、Ring+LL128 specialization
- src/device/primitives.h：统一 primitive 接口

## D3. Tree AllReduce Device Schedule

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 沿 Tree 完成上行 reduction 和下行 broadcast，降低大规模 Rank 的逻辑深度 |
| 非目标 | 不声称 Tree 总比 Ring 快；不展开 NVLS/CollNet tree |
| 输入 | ncclDevWorkColl、tree parent/children、chunk、protocol |
| 输出与停止边界 | 每个 Rank recvbuff 获得 reduce 结果；停止在 up/down traversal 完成 |
| 上下游接口 | 上游为 D1；下游为 protocol Primitives |
| 核心对象 | channel tree peers、Fan、Primitives、chunkCount |
| 主流程 | leaf/internal/root 分角色执行 reduce up → root 形成结果 → broadcast down |
| 状态与生命周期 | chunk 在上行合并，在 root 完成 reduction，再下行复制 |
| 并发、同步与所有权 | parent/children 形成 fan-in/fan-out；每层依赖 connector step |
| 异常与退出 | 非法 tree peer 应在 init graph 阶段避免；device async error 由 Host 观察 |
| 性能与可观测 | 深度约为 O(log p) 的拓扑直觉；实际成本受链路、fanout、protocol 影响 |
| 源码证据与遗留 | 源码事实：runTreeUpDown 和 Tree specializations。待运行：小消息 Tree 选择与时序 |

~~~plantuml
@startuml
title D3 Tree AllReduce
participant "Children" as Child
participant "Current Rank" as Rank
participant "Parent/Root" as Parent
group Reduce Up
  Child -> Rank : chunks
  Rank -> Rank : reduce child + local
  Rank -> Parent : reduced chunk
end
group Broadcast Down
  Parent -> Rank : final chunk
  Rank -> Child : copy/send final chunk
end
@enduml
~~~

### 设计说明与源码锚点

Tree schedule 的关键不是“步数公式”，而是 parent/children 的 fan-in 和 fan-out。
Leaf、internal node 和 root 采用不同 primitive 组合。最终选择仍由 Host tuning 决定。

- src/device/all_reduce.h：runTreeUpDown、Tree+Simple、Tree+LL、Tree+LL128 specialization
- src/include/channel.h：Tree peer 结构

## D4. Simple Protocol Primitive Pipeline

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 使用 step/slice/chunk 和 connector credit 高吞吐地完成 recv、reduce、copy、send 组合 |
| 非目标 | 不把 Simple 当作 algorithm；不规定实际 Transport |
| 输入 | local source/destination、recv/send conn、SlicePerChunk、StepPerSlice、work element range |
| 输出与停止边界 | 指定元素被接收/归约/发送并发布 step；停止在 primitive call 完成 |
| 上下游接口 | 上游为 Ring/Tree schedule；下游为 connector buffers/FIFO 和 reduction copy |
| 核心对象 | Primitives ProtoSimple、waitPeer、postPeer、connFifo、stepSize |
| 主流程 | waitPeer 获得 credit/data → 准备 source/destination pointers → reduce/copy → postPeer 发布进度 |
| 状态与生命周期 | step 单调递增并按 NCCL_STEPS 环形复用；slice 是单次流水推进粒度 |
| 并发、同步与所有权 | warp/thread 分工等待、搬运和同步；生产者发布后消费者才能复用 |
| 异常与退出 | GPU 等待依赖 peer/proxy 正确推进；卡死需结合 FIFO/head/tail 定位 |
| 性能与可观测 | stepSize、slice/chunk、unroll、线程数、memory transaction 和链路 backpressure |
| 源码证据与遗留 | 源码事实：ProtoSimple Primitives、waitPeer、postPeer、recvReduceSend。待运行：stall 原因 |

~~~plantuml
@startuml
title D4 Simple recvReduceSend
participant "Previous Peer/Proxy" as Prev
participant "waitPeer" as Wait
participant "Reduce/Copy Workers" as Work
participant "postPeer" as Post
participant "Next Peer/Proxy" as Next
Prev -> Wait : data ready / tail advances
Wait -> Wait : credit + pointer selection
Wait -> Work : remote src + local src + dst
Work -> Work : reduce and store
Work -> Post : data stored
Post -> Next : publish FIFO/tail
Post -> Prev : release head/credit
@enduml
~~~

### 设计说明与源码锚点

chunk 是 algorithm 对消息的分工粒度，slice 是 primitive 的流水粒度，step 是
connector 环形 buffer 的流控位置。三者相关但不等价。Simple 通常偏向带宽吞吐，
仍需运行数据验证具体消息区间。

- src/device/prims_simple.h：ProtoSimple Primitives、waitPeer、postPeer、recvReduceSend
- src/device/primitives.h：protocol step size
- src/include/device.h：NCCL_STEPS、ncclConnInfo

## D5. LL/LL128 协议布局与推进机制

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 使用协议特定 line/flag 布局降低同步开销，并在吞吐和延迟之间提供不同实现 |
| 非目标 | 不把 LL/LL128 解释为 Ring/Tree；不记忆固定消息阈值 |
| 输入 | algorithm schedule、LL 或 LL128 specialization、conn buffer、step |
| 输出与停止边界 | 协议 line 带有效标记到达 peer 并推进 step；停止在 primitive 完成 |
| 上下游接口 | 上游为 Ring/Tree；下游为 LL/LL128 conn buffers |
| 核心对象 | ProtoLL、ProtoLL128、ncclLLFifoLine、data/flag layout、stepLines |
| 主流程 | 等待 line flag/credit → 读有效 payload → reduce/copy → 写 payload 和 flag |
| 状态与生命周期 | flag 与 step 协同防止读到旧 slot；buffer 仍按 NCCL_STEPS 复用 |
| 并发、同步与所有权 | 生产者必须在发布有效标记前完成 payload 写入；消费者验证标记后读取 |
| 异常与退出 | 协议不匹配或 flag 不推进会导致 wait；由 async error/watchdog 辅助定位 |
| 性能与可观测 | LL 偏低延迟，LL128 提升 payload efficiency；最终选择由 tuning 决定 |
| 源码证据与遗留 | 源码事实：prims_ll.h、prims_ll128.h 和对应 specialization。待运行：阈值与平台效率 |

~~~plantuml
@startuml
title D5 LL 系列 Line 发布规则
participant Producer
database "LL/LL128 Line\npayload + flag" as Line
participant Consumer
Producer -> Line : 写 payload
Producer -> Line : 发布当前 step flag
Consumer -> Line : 轮询期望 flag
Line --> Consumer : payload valid
Consumer -> Consumer : reduce/copy
note over Producer,Consumer : flag 建立 payload 可见性边界
@enduml
~~~

### 设计说明与源码锚点

LL 与 LL128 都把数据布局和同步标记耦合到协议 line，但 payload 比例、line 结构和
线程协作不同。它们是 protocol，不改变 AllReduce 的数学语义，也不决定使用 Ring 还是 Tree。

- src/device/prims_ll.h：ProtoLL Primitives、recvReduceSend
- src/device/prims_ll128.h：ProtoLL128 Primitives、recvReduceSend
- src/device/all_reduce.h：LL/LL128 algorithm specializations
- src/include/device.h：LL 常量和 connector state

## D6. Connector/FIFO、Step/Credit 与 GPU-Proxy 同步

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 在有限环形 slots 上协调 GPU producer/consumer、peer GPU 和 Proxy，防止覆盖未消费数据 |
| 非目标 | 不决定 buffer 的物理 Transport；不替代 CUDA stream completion |
| 输入 | ncclConnInfo buffs/head/tail/connFifo/step、NCCL_STEPS |
| 输出与停止边界 | slot 可写、数据可读或 credit 释放；停止在一个 step 的 ownership 交接完成 |
| 上下游接口 | 上游为 protocol primitive；下游为 peer GPU 或 Proxy progress |
| 核心对象 | ncclConnInfo、head、tail、connFifo、step、stepSize |
| 主流程 | producer 等 credit → 写 buffer/FIFO → 发布 tail → consumer 读取 → 发布 head |
| 状态与生命周期 | slot index 为 step mod NCCL_STEPS；逻辑 step 单调增长避免 ABA |
| 并发、同步与所有权 | head/tail 是跨执行域同步点；发布顺序必须保证 payload/FIFO 可见 |
| 异常与退出 | head/tail 停滞表现为 backpressure 或 hang；不可仅看一个计数器判断根因 |
| 性能与可观测 | FIFO occupancy、head-tail distance、step stalls、Proxy polling 和 GPU wait |
| 源码证据与遗留 | 源码事实：ncclConnInfo、Simple/LL/LL128 wait/post。待运行：GPU 与 Proxy timeline |

~~~plantuml
@startuml
title D6 Step Slot 所有权循环
state Free
state "Producer Owns" as Prod
state Ready
state "Consumer Owns" as Cons
Free --> Prod : head 提供 credit
Prod --> Ready : payload/FIFO 写入后发布 tail
Ready --> Cons : consumer 观察 tail/flag
Cons --> Free : 完成读取后发布 head
note right of Free : slot = step mod NCCL_STEPS
@enduml
~~~

### 设计说明与源码锚点

从 RDMA 视角看，connector FIFO 是 NCCL 自己的数据面 credit/progress 契约；
ncclNet request/WQE/CQ 是 backend 传输契约。Proxy 在两者之间做状态转换。

- src/include/device.h：NCCL_STEPS、ncclConnInfo、ncclConnector
- src/device/prims_simple.h：waitPeer、postPeer
- src/device/prims_ll.h、src/device/prims_ll128.h：协议 step/credit
- src/transport/net.cc：Proxy 对 head/tail/connFifo 的推进

## 本卷分层总结

~~~plantuml
@startuml
title Device 执行分层
[Kernel / Work Dispatch] --> [Collective Algorithm\nRing or Tree]
[Collective Algorithm\nRing or Tree] --> [Protocol\nSimple or LL or LL128]
[Protocol\nSimple or LL or LL128] --> [Primitive\nrecv/reduce/copy/send]
[Primitive\nrecv/reduce/copy/send] --> [Connector FIFO\nstep/credit]
[Connector FIFO\nstep/credit] --> [P2P/SHM/NET Path]
@enduml
~~~
