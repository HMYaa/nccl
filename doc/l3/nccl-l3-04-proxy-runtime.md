# NCCL L3-04：Proxy Runtime

## 1. 文档范围

本卷把 Proxy 拆成控制面和数据推进面。控制面负责连接与资源请求；推进面负责把
已发布 operation 持续推进到 transport request 完成。Proxy 不是单纯的 CPU memcpy 线程。

## P1. Proxy Client、Service Thread 与控制面

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 在 Rank 进程内外传递 setup/connect/register 等控制请求，并把请求分派给目标 Proxy connection |
| 非目标 | 不持续推进 collective chunk；不执行 Host tuning |
| 输入 | proxy connector、request type、request buffer、response buffer、blocking/async mode |
| 输出与停止边界 | transport connection/resource handle 或控制请求完成；停止在 response 返回 client |
| 上下游接口 | 上游为 Transport setup；下游为 Proxy service 和 transport proxy callbacks |
| 核心对象 | ncclProxyState、ncclProxyConnection、proxy connector、UDS/socket messages |
| 主流程 | client call → 发送 request → service dispatch → transport callback → 返回 response |
| 状态与生命周期 | connection 在 communicator 初始化/连接期建立；service thread 随 proxy state |
| 并发、同步与所有权 | blocking call 等待 response；async call 用异步 response/state 交接 |
| 异常与退出 | socket/service/callback 错误写入 response 并传回 caller；销毁时停止 service |
| 性能与可观测 | PROXY/NET setup 日志和控制请求延迟；不使用 data bandwidth 衡量 |
| 源码证据与遗留 | 源码事实：ncclProxyCallAsync、ncclProxyCallBlocking、ncclProxyConnection。待运行：跨进程 Proxy 部署形态 |

~~~plantuml
@startuml
title P1 Proxy 控制面
participant "Transport Client" as Client
participant "ncclProxyCallAsync/Blocking" as Call
participant "Proxy Service Thread" as Service
participant "Transport Callback" as Callback
Client -> Call : setup/connect/register request
Call -> Service : control message
Service -> Callback : dispatch(connection,type)
Callback --> Service : response/resource
Service --> Call : response
Call --> Client : result
note over Client,Service : 控制请求完成不等于 collective data 完成
@enduml
~~~

### 设计说明与源码锚点

Proxy control call 负责在正确的 Proxy context 中执行连接和资源操作。Blocking 与
async 的区别是控制请求的等待方式，不是 collective 数据算法的区别。

- src/proxy.cc：ncclProxyCallAsync、ncclProxyCallBlocking、Proxy service loop
- src/include/proxy.h：ncclProxyState、ncclProxyConnection、Proxy message types
- src/transport/net.cc：NET setup/connect proxy callbacks

## P2. Proxy Operation 生成、保存和 Posted Queue

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 将 kernel plan 所需的网络工作编码为 Proxy operation，并安全交给 progress thread |
| 非目标 | op 被 post 不代表 ncclNet request 已完成 |
| 输入 | ncclKernelPlan 中的 proxy op、channel/peer/step/slice/chunk 参数 |
| 输出与停止边界 | operation 发布到 posted queue；停止在 progress thread 可见 |
| 上下游接口 | 上游为 Host plan builder/launch；下游为 ncclProxyGetPostedOps |
| 核心对象 | ncclProxyOp、ncclProxyArgs、ncclProxyOpsPool、postedOps、opCount |
| 主流程 | ncclProxySaveOp 聚合/保存 → ncclProxyStart → ncclProxyPost 发布 index range |
| 状态与生命周期 | saved op 属于 plan 准备；posted 后由 progress runtime 接管并转换为 active args |
| 并发、同步与所有权 | Host 为 producer，progress thread 为 consumer；queue/index 发布必须有内存可见性 |
| 异常与退出 | pool 资源不足或无效 connection 时不发布；错误沿 launch/group 返回 |
| 性能与可观测 | 观察 posted op 数、pool pressure、唤醒次数；posted 数不是 WQE 数 |
| 源码证据与遗留 | 源码事实：ncclProxySaveOp、ncclProxyPost、ncclProxyStart、ncclProxyOpsPool。待运行：批量聚合效果 |

~~~plantuml
@startuml
title P2 Proxy Op Producer-Consumer 交接
participant "Host Plan Builder" as Host
database "Proxy Ops Pool" as Pool
queue "Posted Queue" as Posted
participant "Progress Thread" as Progress
Host -> Pool : ncclProxySaveOp
Host -> Pool : 保存 channel/peer/chunk params
Host -> Posted : ncclProxyPost(index range)
Posted -> Progress : ncclProxyGetPostedOps
note right of Posted : publish 只表示工作可见
@enduml
~~~

### 设计说明与源码锚点

ncclProxyOp 是 Host 规划结果的一部分；ncclProxyArgs 是 progress 侧推进一组 operation
所需的活动状态。它们都不拥有应用 payload，只持有地址、连接和推进参数。

- src/proxy.cc：ncclProxySaveOp、ncclProxyPost、ncclProxyStart、ncclProxyGetPostedOps
- src/include/proxy.h：ncclProxyOp、ncclProxyArgs、ncclProxyOpsPool、ncclProxyOps
- src/enqueue/enqueue.cc：ncclAddProxyOpIfNeeded

## P3. Active Operation Scheduler

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 将 posted operations 转换为 active args，并在公平性、合并和阶段约束下反复调用 progress |
| 非目标 | 不重新计算 algorithm/protocol；不直接解析 collective 用户语义 |
| 输入 | posted queue、ncclProxyArgs、connection progress callback |
| 输出与停止边界 | args state 达到 done 并回收；停止在 active op 从 scheduler 移除 |
| 上下游接口 | 上游为 P2；下游为 transport-specific progress callback |
| 核心对象 | ncclProxyProgressState、active list、ncclProxyArgs、state、idle counters |
| 主流程 | get posted → append active → progressOps 遍历 → callback 更新 state → done 回收 |
| 状态与生命周期 | Ready/Progress/None 等 callback 状态驱动 args；active list 跨多次轮询 |
| 并发、同步与所有权 | 单 progress loop 持有 active 调度；GPU/FIFO 和 backend completion 异步变化 |
| 异常与退出 | callback error 记录为 async error 并终止对应 op；thread stop 时清理 active state |
| 性能与可观测 | polling 占用、active op 数、idle backoff、公平性；过多 channel 会增加 progress 压力 |
| 源码证据与遗留 | 源码事实：ncclProxyGetPostedOps、progressOps、ncclProxyProgress。待运行：CPU core 和 polling 行为 |

~~~plantuml
@startuml
title P3 Active Op 状态机
[*] --> Posted
Posted --> Active : getPostedOps
Active --> Progressing : progress callback
Progressing --> Progressing : partial / request pending
Progressing --> Active : yield to other ops
Progressing --> Done : args->state done
Done --> Recycled : return pool/resources
Recycled --> [*]
Active --> Failed : callback error
Progressing --> Failed : async transport error
Failed --> Recycled
@enduml
~~~

### 设计说明与源码锚点

Progress loop 的职责类似 software progress engine：观察 GPU connector/FIFO 和
ncclNet request completion，满足条件时推进一个 slice/step，再让出执行机会。
这与驱动中的 CQ polling 类似，但其调度对象是 NCCL proxy args，而不是裸 CQE。

- src/proxy.cc：ncclProxyGetPostedOps、progressOps、ncclProxyProgress
- src/include/proxy.h：ncclProxyArgs、ncclProxyProgressState

## P4. Send/Recv Progress 与完成检测

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 在 GPU connector 条件和 ncclNet request 条件满足时，按 step/slice/chunk 推进 send/recv |
| 非目标 | 不保证每次 callback 都 post WQE；不把 request complete 等同于 CUDA stream complete |
| 输入 | ncclProxyArgs、send/recv resources、connFifo/head/tail、ncclNet comm/request |
| 输出与停止边界 | 更新 posted/transmitted/done step 和 connector head/tail；停止在 args 全部完成 |
| 上下游接口 | 上游为 P3；下游为 ncclNet isend/irecv/test 和 GPU connector |
| 核心对象 | subArgs、base/posted/transmitted/done、requests[NCCL_STEPS]、connFifo |
| 主流程 | 检查 GPU ready/credit → isend/irecv → test request → 更新 FIFO/head/tail → 下一 step |
| 状态与生命周期 | 每个 slot request 从 null 到 in-flight 再归 null；step 计数单调推进 |
| 并发、同步与所有权 | GPU 与 Proxy 是 producer/consumer；head/tail 发布必须遵守内存顺序 |
| 异常与退出 | ncclNet call/test 失败转 async error；不得提前释放 in-flight request/resource |
| 性能与可观测 | slice/chunk、NCCL_STEPS、request latency、CQ progress、CPU polling；需与 NIC counters 联合 |
| 源码证据与遗留 | 源码事实：sendProxyProgress、recvProxyProgress。待运行：每次 callback 到 WQE/CQE 的对应关系 |

~~~plantuml
@startuml
title P4 NET Send Progress 单个 Step
participant "GPU Primitive" as GPU
participant "sendProxyProgress" as Send
participant "ncclNet.isend" as Isend
participant "ncclNet.test" as Test
participant "Connector FIFO" as FIFO
GPU -> FIFO : 发布 ready data / tail
Send -> FIFO : 检查 step ready
Send -> Isend : buffer,size,mhandle
Isend --> Send : request or pending
loop request in flight
  Send -> Test : test(request)
end
Test --> Send : done
Send -> FIFO : 更新 head/credit
FIFO --> GPU : slot 可复用
@enduml
~~~

### 设计说明与源码锚点

send 和 recv progress 都是多阶段流水：posted 表示已向 backend 提交，transmitted 表示
数据传输阶段满足条件，done 表示该 operation 的完成计数推进。实际字段和分支随协议、
GDR read/write 和 flush 条件变化，静态图只表达共同骨架。

- src/transport/net.cc：sendProxyProgress、recvProxyProgress
- src/include/proxy.h：ncclProxyArgs、ncclProxySubArgs
- src/include/device.h：ncclConnInfo
- src/include/plugin/nccl_net.h：isend、irecv、test

## 本卷控制面与推进面边界

~~~plantuml
@startuml
title Proxy 双轨设计
rectangle "Control Rail" {
  [Client Call] --> [Service Thread]
  [Service Thread] --> [Setup/Connect/Register]
}
rectangle "Progress Rail" {
  [Save/Post Op] --> [Posted Queue]
  [Posted Queue] --> [Active Scheduler]
  [Active Scheduler] --> [Send/Recv Progress]
}
[Setup/Connect/Register] --> [Send/Recv Progress] : 提供 connection/resources
@enduml
~~~
