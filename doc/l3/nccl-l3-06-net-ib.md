# NCCL L3-06：ncclNet 与 net_ib

## 1. 文档范围

本卷从 ncclNet ABI 进入 IB backend，追踪到 QP/CQ/MR/request 的资源和状态。它使用
RDMA 术语直接描述，不重复解释 Verbs 基础。resiliency、GIN 和 EFA GDA 不在第一版范围。

## N1. ncclNet ABI、Backend、NIC 与 Rail 选择

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 通过版本化 ncclNet ABI 加载网络 backend，枚举 net devices 并向 topology/tuning 暴露属性 |
| 非目标 | 不在 plugin load 阶段为每次 collective 建 QP；不证明使用 IB 而非 socket |
| 输入 | plugin name/env、动态库、commId/config、backend init/devices/getProperties |
| 输出与停止边界 | comm->ncclNet、device properties、NIC/rail topology 节点；停止在 backend 可供连接使用 |
| 上下游接口 | 上游为 communicator init；下游为 topology、NET Transport 和 connection setup |
| 核心对象 | ncclNet_t ABI、plugin state、ncclNetProperties_t、ncclIbDev/ncclIbMergedDev、railId |
| 主流程 | plugin load/version adapt → init → devices → getProperties → topology NIC population |
| 状态与生命周期 | backend context 和 device inventory 随 comm/process；connection 独立创建 |
| 并发、同步与所有权 | plugin 初始化受全局/comm 锁约束；device properties 作为只读能力输入 |
| 异常与退出 | plugin 不可用时进入 fallback backend；device 初始化失败则对应 NET 能力不可用 |
| 性能与可观测 | NET/IB init 日志、device name/speed/pciPath/port/rail；属性不是实测吞吐 |
| 源码证据与遗留 | 源码事实：ncclNetPluginInit、ncclNetInit、ncclIbInitDevices、ncclIbGetProperties。待运行：实际 backend 和 rail |

~~~plantuml
@startuml
title N1 ncclNet Backend 发现
participant "ncclNetInit" as Init
participant "Plugin Loader" as Plugin
participant "ncclNet ABI" as ABI
participant "NET/IB Backend" as IB
database "Topology NIC Nodes" as Topo
Init -> Plugin : load requested plugin
Plugin -> ABI : resolve supported version
ABI -> IB : init / devices / getProperties
IB --> ABI : dev name/speed/pciPath/rail
ABI -> Topo : populate NIC capabilities
note right of Topo : 能力输入，不是单次运行选择
@enduml
~~~

### 设计说明与源码锚点

ncclNet 是 NCCL runtime 与具体网络实现之间的 ABI。内置 IB backend 通过同一函数表
暴露 init/devices/listen/connect/regMr/isend/irecv/test。Rail 属性进入 topology 后，
还要结合 graph/channel 和 runtime 配置才能形成实际路径。

- src/plugin/net.cc：ncclNetPluginInit、ncclNetInit
- src/include/plugin/nccl_net.h：版本化 ncclNet ABI
- src/transport/net_ib/common.cc：IB function table
- src/transport/net_ib/init.cc：ncclIbInitDevices、ncclIbInit、ncclIbDevices、ncclIbGetProperties
- src/transport/net_ib/common.h：ncclIbDev、ncclIbMergedDev

## N2. Listen/Connect、QP/CQ 资源建立

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 在 send/recv peer 间交换连接元数据，建立 CQ/QP 并把 QP 推进到可传输状态 |
| 非目标 | 不发布 collective payload WQE；不选择 collective channel 数 |
| 输入 | listen handle、device、peer handle、nQpsPerDev、comm config |
| 输出与停止边界 | ncclIbSendComm/ncclIbRecvComm 及 ready QP/CQ；停止在 connect/accept 完成 |
| 上下游接口 | 上游为 NET Transport Proxy setup；下游为 regMr 和 isend/irecv |
| 核心对象 | ncclIbListenComm、ncclIbSendComm、ncclIbRecvComm、ncclIbNetCommBase、QP、CQ、connection metadata |
| 主流程 | listen → connect/accept state machine → 建 CQ/QP → 交换 QPN/PSN/address → RTR/RTS |
| 状态与生命周期 | listenComm 用于握手；send/recv comm 随 connector；QP/CQ 在 close 时销毁 |
| 并发、同步与所有权 | connect/accept 可非阻塞分阶段推进；双方元数据和 QP 状态必须匹配 |
| 异常与退出 | socket 握手、CQ/QP 创建或 modify_qp 失败时逆序释放部分资源 |
| 性能与可观测 | QP 数、CQ size、GID/LID、MTU、SL/TC、timeout/retry；连接日志需脱敏 |
| 源码证据与遗留 | 源码事实：ncclIbListen、ncclIbConnectImpl、ncclIbAcceptImpl、ncclIbCommState。待运行：实际 QP 参数 |

~~~plantuml
@startuml
title N2 IB Connection 建立
participant Sender
participant "ncclIbConnectImpl" as Connect
participant "Handshake Socket" as Sock
participant "ncclIbAcceptImpl" as Accept
participant Receiver
Sender -> Connect : remote handle
Receiver -> Accept : listenComm
Connect -> Connect : create CQ/QP\nQP INIT
Accept -> Accept : create CQ/QP\nQP INIT
Connect -> Sock : local QPN/PSN/GID
Sock -> Accept : sender metadata
Accept -> Sock : receiver metadata
Sock -> Connect : receiver metadata
Connect -> Connect : QP RTR -> RTS
Accept -> Accept : QP RTR -> RTS
Connect --> Sender : sendComm
Accept --> Receiver : recvComm
@enduml
~~~

### 设计说明与源码锚点

连接状态机是控制面。QP/CQ ready 后还没有 collective payload WQE；Proxy progress
后续调用 isend/irecv 才创建 request 并 post work。一个逻辑 NET connection 可能包含
多个 device/QP，不能简单等同单个 QP。

- src/transport/net_ib/connect.cc：ncclIbCommState、ncclIbListen、ncclIbConnectImpl、
  ncclIbAcceptImpl、QP/CQ 初始化
- src/transport/net_ib/connect.h：ncclIbConnectionMetadata
- src/transport/net_ib/common.h：ncclIbSendComm、ncclIbRecvComm、ncclIbNetCommBase

## N3. MR/GDR 注册、缓存与注销生命周期

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 将 Host/GPU buffer 注册到参与传输的 IB devices，生成 backend mhandle 并安全注销 |
| 非目标 | 注册成功本身不证明实际 DMA 绕过 CPU；不管理应用 buffer 的分配生命周期 |
| 输入 | comm、data、size、memory type、可选 dma-buf fd/offset、MR flags |
| 输出与停止边界 | 每 device ibv_mr 集合包装为 mhandle；停止在 backend 可用于 post send/recv |
| 上下游接口 | 上游为 NET Transport registration；下游为 isend/irecv SGE/WQE 构造 |
| 核心对象 | ncclIbMrHandle、ibv_mr、lkey/rkey、dma-buf、registration cache record |
| 主流程 | 规范化范围 → 为每 device 注册 MR/dma-buf MR → 保存 handles → dereg 逐项释放 |
| 状态与生命周期 | MR 必须覆盖所有 in-flight WQE；注销晚于 request completion |
| 并发、同步与所有权 | cache 命中需保护引用关系；应用拥有 memory，backend 拥有 registration handle |
| 异常与退出 | 部分 device 注册失败时回滚已注册 MR；不留下可被复用的悬空 lkey |
| 性能与可观测 | 注册延迟、cache hit、page pin、dma-buf/GDR capability；结合 ibv/NIC 计数器验证 |
| 源码证据与遗留 | 源码事实：ncclIbRegMr、ncclIbRegMrDmaBufInternal、ncclIbDeregMr。待运行：GDR path 和 cache |

~~~plantuml
@startuml
title N3 MR 生命周期
actor Application
participant "NET Transport" as Net
participant "ncclIbRegMr*" as Reg
participant "IB Devices" as Devs
participant "mhandle wrapper" as Handle
Application -> Net : buffer address + size
Net -> Reg : memory type / dma-buf metadata
loop each IB device in virtual dev
  Reg -> Devs : ibv_reg_mr or dma-buf registration
  Devs --> Reg : ibv_mr / lkey / rkey
end
Reg -> Handle : aggregate device MRs
Handle --> Net : use for requests
Net -> Reg : dereg after all requests complete
Reg -> Devs : ibv_dereg_mr
@enduml
~~~

### 设计说明与源码锚点

GDR 的静态必要条件包括 GPU memory 可注册、backend 支持对应 path 和 Transport
选择正确 buffer，但这些条件仍不能证明一次实际运行没有经过 staging。需要联合
NCCL_DEBUG、Profiler、dma-buf/GDR 日志和硬件计数器。

- src/transport/net_ib/reg.cc：ncclIbRegMrDmaBufInternal2、ncclIbRegMr、
  ncclIbRegMrDmaBuf、ncclIbDeregMr
- src/transport/net_ib/gdr.cc：GDR capability 支撑
- src/transport/net_ib/common.h：MR wrapper 和 comm device base
- src/register/：NCCL 上层 registration/cache

## N4. isend/irecv/test 到 WQE/CQ Completion

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 将 Proxy 提交的异步 send/recv 转换为 IB request、SGE/WQE，并通过 CQE 完成 request |
| 非目标 | 一个 NCCL request 不必严格对应一个 WQE；request complete 不等于 CUDA stream complete |
| 输入 | send/recv comm、buffers/sizes/tags、mhandles、request pool |
| 输出与停止边界 | ncclIbRequest 从 in-flight 到 done，test 返回完成；停止在 request 可回收 |
| 上下游接口 | 上游为 send/recv Proxy progress；下游为 ibv_post_send/recv 和 CQ polling |
| 核心对象 | ncclIbRequest、completion records、sendReqs slots、QP、CQ、WR ID |
| 主流程 | 获取 request → 匹配 recv/send → 组 SGE/WQE → post → poll CQ → 累计 events → test done |
| 状态与生命周期 | request pool slot 从 free 到 pending/events complete 再 free；WR ID 关联 request |
| 并发、同步与所有权 | Proxy thread poll CQ；NIC 异步生产 CQE；request 未完成前 MR/buffer 不得注销 |
| 异常与退出 | post 或 WC error 转 nccl error；必须区分 retry/pending 与 fatal completion |
| 性能与可观测 | WQE batching、CQ moderation/polling、nQps、request slots、latency counters |
| 源码证据与遗留 | 源码事实：ncclIbGetRequest、ncclIbIsend、ncclIbIrecv、ncclIbTest。待运行：request-WQE-CQE 基数 |

~~~plantuml
@startuml
title N4 IB Request 数据面
participant "Proxy Progress" as Proxy
participant "ncclIbIsend/Irecv" as Submit
database "Request Pool" as Pool
participant "ibv_post_*" as Post
participant NIC
queue CQ
participant "ncclIbTest" as Test
Proxy -> Submit : buffers/tags/mhandles
Submit -> Pool : allocate ncclIbRequest
Submit -> Post : SGE + WR + wr_id
Post -> NIC : WQE
NIC -> CQ : CQE
Proxy -> Test : request
Test -> CQ : poll
CQ --> Test : WC / wr_id
Test -> Pool : update completion records
Test --> Proxy : done when all events complete
@enduml
~~~

### 设计说明与源码锚点

从驱动视角，ncclIbRequest 是软件聚合状态。它可以等待多个 device/QP event，
ncclIbTest 通过 CQ polling 和 completion record 判断整体完成。不要用 request 数量
直接推导 WQE 或包数量。

- src/transport/net_ib/p2p.cc：ncclIbGetRequest、ncclIbIsend、ncclIbIrecv、ncclIbTest
- src/transport/net_ib/common.h：ncclIbRequest、ncclIbRequestCompletionRecord
- src/transport/net_ib/connect.cc：QP/CQ resource

## N5. 多 NIC、多 Rail 与多 QP 映射

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 将 topology/channel 的网络并行度映射到 virtual devices、rails 和多个 QP，兼顾局部性与带宽 |
| 非目标 | 不承诺平均分流；不把 railId 当作物理流量计数器 |
| 输入 | NIC properties/railId、graph channel、rank、NCCL_CROSS_NIC、merged device、nQpsPerDev |
| 输出与停止边界 | channel 对 netDev/rail 的选择及 request 到 QP 的映射；停止在可 post 的 comm/QP 集合 |
| 上下游接口 | 上游为 topology/NET Transport；下游为 N2/N4 |
| 核心对象 | ncclIbDev、ncclIbMergedDev、railId、planeId、nQps、qp index mapping |
| 主流程 | 枚举/合并 NIC → topology 选择 netDev → 建多 QP comm → request id 映射到 QP |
| 状态与生命周期 | device/rail inventory 持久；channel 和 connection 映射随 communicator |
| 并发、同步与所有权 | 多 channel/request 可并行使用不同 NIC/QP；CQ/resource 归属必须明确 |
| 异常与退出 | rail 不一致或 cross-NIC 约束不满足时降级/拒绝候选；不可跨错误 fabric 建链 |
| 性能与可观测 | 每 NIC bytes/packets/ECN/PFC、QP/CQ counters、channel-to-NIC 日志 |
| 源码证据与遗留 | 源码事实：railId 属性、merged devices、graph netDev 选择、QP request mapping。待运行：真实流量分布 |

~~~plantuml
@startuml
title N5 Channel 到 Rail/QP 映射
object "Channel 0" as C0
object "Channel 1" as C1
object "Topology Graph" as Graph
object "Rail 0 / NIC 0" as R0
object "Rail 1 / NIC 1" as R1
object "QP Set 0" as Q0
object "QP Set 1" as Q1
Graph --> C0 : select netDev
Graph --> C1 : select netDev
C0 --> R0
C1 --> R1
R0 --> Q0 : nQpsPerDev
R1 --> Q1 : nQpsPerDev
note bottom of Graph : 静态映射需用 NIC/QP counters 验证实际流量
@enduml
~~~

### 设计说明与源码锚点

Rail 是拓扑和 fabric 局部性的标识；channel 是 NCCL 并行 lane；QP 是 Verbs 传输资源。
三者不应一一等同。一个 channel 可经 virtual device 使用多个物理 device/QP，
多个 channel 也可能共享同一 NIC。

- src/transport/net_ib/init.cc：ncclIbInitDevices、merged device、railId
- src/transport/net_ib/common.h：ncclIbDev、ncclIbMergedDev、QP collections
- src/transport/net_ib/p2p.cc：request 到 QP 的映射
- src/graph/search.cc、src/graph/connect.cc：rail/channel topology 约束
- src/include/plugin/net/net_v12.h：railId/planeId properties

## 本卷端到端边界

~~~plantuml
@startuml
title Proxy 到 NIC 的静态数据链
[sendProxyProgress] --> [ncclNet.isend]
[ncclNet.isend] --> [ncclIbRequest]
[ncclIbRequest] --> [SGE/WQE]
[SGE/WQE] --> [QP]
[QP] --> [NIC/Fabric]
[NIC/Fabric] --> [CQE]
[CQE] --> [ncclIbTest]
[ncclIbTest] --> [Proxy step done]
@enduml
~~~
