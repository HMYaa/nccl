# NCCL L3-07：横切运行时契约

## 1. 文档范围

本卷统一描述跨 Host、GPU、Proxy 和 ncclNet 的所有权、完成、错误与证据边界。
它不替代各主流程卷，只定义跨层必须一致理解的契约。

## C1. 跨层对象所有权与生命周期

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 明确请求描述、任务、计划、Proxy operation、NET request 和用户 buffer 的所有者与释放条件 |
| 非目标 | 不把 C++ 对象生命周期等同于数据 buffer 安全复用期 |
| 输入 | API request、comm pools、plan queues、proxy pools、backend request pools、CUDA stream |
| 输出与停止边界 | 每类对象有唯一 owner、handoff 和回收条件；停止在 stream completion 和资源回收契约清楚 |
| 上下游接口 | 横跨 H1-H8、P2-P4、D6、N3-N4 |
| 核心对象 | ncclInfo、ncclTaskColl、ncclDevWorkColl、ncclKernelPlan、ncclProxyOp/Args、ncclIbRequest、buffers |
| 主流程 | stack request → planner task → device work/plan → proxy args → network request → completion/recycle |
| 状态与生命周期 | 描述对象逐层转换；用户 buffer 地址被引用但 payload 不被 task 复制 |
| 并发、同步与所有权 | Host、GPU、Proxy、NIC 可同时持有对 buffer/resource 的使用权，释放需等待最晚消费者 |
| 异常与退出 | 失败路径仍需等待或取消 in-flight work 后再释放 MR、plan 和 comm resources |
| 性能与可观测 | pool pressure、registration cache、in-flight request、stream event；生命周期过长会占资源 |
| 源码证据与遗留 | 源码事实：各对象定义和 pool/queue；设计归纳：统一 ownership ledger。待运行：实际回收时序 |

~~~plantuml
@startuml
title C1 对象与 Buffer 生命周期
object "ncclInfo\nAPI stack" as Info
object "ncclTaskColl\nPlanner pool" as Task
object "ncclDevWorkColl\nWork buffer" as Work
object "ncclKernelPlan\nPlan pool" as Plan
object "ncclProxyArgs\nProxy active state" as Proxy
object "ncclIbRequest\nBackend pool" as Req
object "User Buffer\nApplication owned" as Buf
Info --> Task : copy description
Task --> Work : encode
Work --> Plan : batch
Plan --> Proxy : proxy operation
Proxy --> Req : isend/irecv
Buf ..> Info : address
Buf ..> Task : address
Buf ..> Work : address/offset
Buf ..> Req : registered DMA range
note bottom of Buf : 安全复用由 stream/application 同步决定
@enduml
~~~

### Ownership Ledger

| 对象 | 创建者/所有者 | 跨层交接 | 回收条件 |
|---|---|---|---|
| ncclInfo | API 栈 | 内容复制给 task | ncclEnqueueCheck 返回 |
| ncclTaskColl | communicator task pool / Planner | scheduler 消费 | task 编码并完成 Planner 回收 |
| ncclDevWorkColl | Planner/work storage | GPU kernel 读取 | 对应 plan 不再需要 work storage |
| ncclKernelPlan | Planner plan pool | launch path | launch 后满足 plan 回收约束 |
| ncclProxyOp/Args | Proxy pool/progress runtime | posted → active | args state done 或失败清理 |
| ncclIbRequest | net_ib request pool | NIC/CQ completion | ncclIbTest 确认所有 event 完成 |
| MR handle | NET/backend registration owner | request 引用 | 无 in-flight request 且 cache/ref 允许 |
| sendbuff/recvbuff | Application | NCCL 异步引用 | CUDA stream/event 建立完成关系后 |

### 源码锚点

- src/include/info.h：ncclInfo
- src/include/comm.h：ncclTaskColl、ncclKernelPlan、ncclKernelPlanner
- src/include/device.h：ncclDevWorkColl、ncclConnInfo
- src/include/proxy.h：ncclProxyOp、ncclProxyArgs
- src/transport/net_ib/common.h：ncclIbRequest

## C2. Host、Proxy、GPU、NET 异步完成与错误传播

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 区分各层完成点，并定义同步依赖和错误向 communicator/application 的传播路径 |
| 非目标 | 不把某层 done 推广成所有层 done；不承诺错误立即同步返回 API |
| 输入 | API return、group result、launch result、proxy args state、ncclNet test、CUDA async error |
| 输出与停止边界 | 应用通过 stream/event 或 async error API 获得可依赖状态；停止在 buffer 可安全复用或错误已处理 |
| 上下游接口 | Host launch、Proxy progress、CUDA runtime、ncclNet backend、application |
| 核心对象 | comm asyncResult、plan state、proxy state、backend request、CUDA stream/event |
| 主流程 | Host 完成提交后 API 可返回；Proxy/GPU 并发推进；backend request 与 kernel completion 仅按 connector 协议建立偏序；stream completion 构成应用可依赖边界 |
| 状态与生命周期 | 各层 completion 是偏序关系，不是单一全局 boolean |
| 并发、同步与所有权 | connector head/tail、CQE、CUDA stream/event 分别建立不同执行域的 happens-before |
| 异常与退出 | 同步检查直接返回；异步错误记录到 comm 并由 ncclCommGetAsyncError/后续 API 观察 |
| 性能与可观测 | 分别测 launch latency、proxy progress、network request 和 stream latency |
| 源码证据与遗留 | 源码事实：ncclEnqueueCheck、group launch、progress error、ncclIbTest、async error API。待运行：故障注入 |

~~~plantuml
@startuml
title C2 完成偏序
object "Host 提交完成" as Submit
object "API 返回\n仅表示提交边界" as Return
object "Proxy 推进" as Proxy
object "NET Request Done\nCQE/test" as NetDone
object "GPU Kernel 推进" as GPU
object "Kernel Done" as KernelDone
object "CUDA Stream/Event Done" as StreamDone
object "应用可复用 Buffer" as Reuse

Submit --> Return : 调用可以返回
Submit --> Proxy : 使能异步推进
Submit --> GPU : kernel 已入 stream
Proxy --> NetDone : 某个 request 完成
Proxy <--> GPU : connector/FIFO\n建立具体协议依赖
GPU --> KernelDone
KernelDone --> StreamDone
StreamDone --> Reuse

note bottom of Return : 不对 Proxy、NET 或 GPU 的完成时刻排序
note bottom of NetDone : 只证明对应 backend request 完成
note bottom of StreamDone : 应用依赖的完成边界
@enduml
~~~

### 完成点对照

| 观察点 | 能证明什么 | 不能证明什么 |
|---|---|---|
| collTaskAppend 成功 | Planner 接收了 Host task | kernel 已 launch |
| ncclLaunchKernel 返回 | kernel 已提交到 CUDA stream | kernel 已执行完 |
| ncclNet test done | 对应 backend request 完成 | 整个 collective 或 stream 完成 |
| Proxy args done | 对应 Proxy operation 完成 | 所有 GPU work 已完成 |
| CUDA event/stream complete | 此 stream 上先前工作完成 | 其他 stream/comm 自动完成 |

### 源码锚点

- src/enqueue/enqueue.cc：ncclEnqueueCheck、ncclLaunchKernel
- src/group.cc：group prepare/launch/error handling
- src/proxy.cc：progressOps、ncclProxyProgress
- src/transport/net.cc：sendProxyProgress、recvProxyProgress
- src/transport/net_ib/p2p.cc：ncclIbTest
- src/init.cc：ncclCommGetAsyncError 等 communicator error API

## C3. 参数、日志、Profiler 与运行证据边界

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 建立从静态设计结论到目标系统运行结论的可重复验证路径 |
| 非目标 | 不用单一日志行证明端到端性能；不记录与本设计无关的全部环境变量 |
| 输入 | NCCL 参数、DEBUG/SUBSYS、NVTX/Profiler、CUDA trace、NIC/QP counters、benchmark |
| 输出与停止边界 | 每个结论有 source/config/runtime/perf 证据链；停止在可复现和可证伪 |
| 上下游接口 | 连接所有 L3 功能章节和实验报告 |
| 核心对象 | env params、debug masks、profiler events、NCCL tests metrics、hardware counters |
| 主流程 | 提出假设 → 固定版本/config → 采集多层证据 → 对时 → 反例检查 → 结论分级 |
| 状态与生命周期 | 配置在 comm init 或调用期生效；日志和 trace 属于一次实验 snapshot |
| 并发、同步与所有权 | 多进程日志需带 rank/time；GPU/CPU/NIC timeline 需要统一关联字段 |
| 异常与退出 | 缺失某层证据时降级结论，不以推断填补；日志采集失败需明确记录 |
| 性能与可观测 | algo/proto/channel、proxy op、kernel、WQE/CQE、NIC bytes/ECN/PFC、algbw/busbw |
| 源码证据与遗留 | 源码事实：debug/profiler/param plumbing。待运行：GPU/HCA 环境和标准实验矩阵 |

~~~plantuml
@startuml
title C3 证据链
rectangle "Source Evidence\nfile/symbol/branch" as Source
rectangle "Config Evidence\nenv + comm config" as Config
rectangle "Runtime Evidence\nlogs + trace" as Runtime
rectangle "Hardware Evidence\nNIC/QP/GPU counters" as Hardware
rectangle "Performance Evidence\nlatency/algbw/busbw" as Perf
Source --> Config
Config --> Runtime
Runtime --> Hardware
Hardware --> Perf
note bottom of Source : 证明代码具备路径
note bottom of Runtime : 证明本次运行采用路径
note bottom of Perf : 证明结果，不自动证明根因
@enduml
~~~

### 验证矩阵

| 结论 | 最低证据 |
|---|---|
| API 语义为 AllReduce | 源码/API 参数 |
| 本次选择 Ring + Simple | TUNING/INFO 日志或对应 trace |
| 使用 8 channels | launch/tuning 日志和 kernel grid |
| 使用 NET/IB | NET backend/connection 日志 |
| 使用指定 NIC rail | graph/netDev 日志加 NIC counters |
| 使用 GDR | registration/path 日志加 trace/counters |
| 性能瓶颈位于 NIC | NIC 利用率/拥塞加 GPU/Proxy 无其他饱和的排除证据 |

### 源码锚点

- src/debug.cc、src/include/debug.h：日志和 subsystem
- src/misc/param.cc、src/param/param.cc、src/include/param.h：参数
- src/plugin/profiler.cc、src/include/profiler.h：Profiler events
- src/init.cc、src/enqueue/enqueue.cc、src/proxy.cc、src/transport/net.cc：各层日志点

## 横切不变量

1. 源码事实、设计归纳和待运行验证必须显式区分。
2. 任何对象回收都不得早于其最晚异步消费者。
3. Task 入队、plan 生成、kernel launch、Proxy done、NET request done 和 stream complete
   是不同事件。
4. 应用 buffer 的安全复用由 stream/application synchronization 决定，不由 ncclInfo
   或 ncclTaskColl 的 C++ 生命周期决定。
5. algbw/busbw 是性能视角，不能反推 algorithm、protocol、Transport、GDR 或瓶颈。
