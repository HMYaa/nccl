# NCCL L3-03：Transport 与 Connector

## 1. 文档范围

本卷描述 NCCL 如何为 channel peer 选择 Transport，并把 transport-specific 资源映射到
统一 connector。它不讨论 collective algorithm 的选择，也不深入 net_ib 的 WQE 状态机。

## T1. Transport 选择与 Connector 生命周期

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 为每个 channel/peer/方向选择可连接的 Transport，并建立统一 connector 状态 |
| 非目标 | 不选择 Ring/Tree；不把 setup 完成当作数据传输完成 |
| 输入 | comm、ncclTopoGraph、channelId、peer、send/recv type、connIndex |
| 输出与停止边界 | ncclConnector 绑定 transportComm、transportResources 和 ncclConnInfo；停止在 connect 完成 |
| 上下游接口 | 上游为 topology/channel setup；下游为 P2P、SHM 或 NET transport callbacks |
| 核心对象 | ncclTransport、ncclTransportComm、ncclConnector、ncclConnect、ncclConnInfo |
| 主流程 | 遍历 transport → canConnect → setup → 交换 connectInfo → connect → 标记 connected |
| 状态与生命周期 | connector 随 communicator/channel 存续；transport resource 在 comm teardown 时 free |
| 并发、同步与所有权 | 多 Rank 必须成对建立 send/recv；setup 元数据通过 bootstrap 交换 |
| 异常与退出 | transport 不可用时尝试下一候选；全部失败则 connector setup 失败 |
| 性能与可观测 | NCCL_DEBUG_SUBSYS=NET/P2P/SHM/GRAPH；选择结果是路径证据，不是吞吐证据 |
| 源码证据与遗留 | 源码事实：selectTransport、ncclTransportP2pSetup、ncclTransport。待运行：目标 peer 实际选择 |

~~~plantuml
@startuml
title T1 Transport 选择与 Connector 建立
participant "Channel Setup" as Setup
participant "selectTransport" as Select
participant "ncclTransport[]" as Registry
participant "Peer Rank" as Peer
participant "ncclConnector" as Conn
Setup -> Select : graph/channel/peer/direction
loop P2P -> SHM -> NET candidates
  Select -> Registry : canConnect
end
Select -> Registry : setup
Registry -> Peer : exchange ncclConnect
Peer --> Registry : remote connect info
Registry -> Conn : connect + transport resources
note right of Conn : connected 只表示资源就绪
@enduml
~~~

### 设计说明与源码锚点

ncclTransport 是能力和回调表；ncclConnector 是某个 channel peer 方向上的连接实例。
选择顺序和 canConnect 条件把 topology 能力落到具体实现。Device/Proxy 通过统一
ncclConnInfo 观察 buffer、head、tail、step，而不直接理解每种 Transport 的全部资源。

- src/transport.cc：selectTransport、ncclTransportP2pSetup
- src/include/transport.h：ncclTransport、ncclTransportComm
- src/include/device.h：ncclConnector、ncclConnInfo

## T2. P2P/SHM 本地连接和数据路径

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 在同一主机内优先建立 GPU P2P 或共享内存路径，并向 connector 暴露统一缓冲区 |
| 非目标 | 不负责跨节点 RDMA；不保证任意两 GPU 都支持直接 P2P |
| 输入 | peerInfo、topology path、CUDA IPC/P2P 能力、共享内存能力 |
| 输出与停止边界 | p2pTransport 或 shmTransport resources 及 connector conn；停止在本地 peer 可交换数据 |
| 上下游接口 | 上游为 T1；下游为 Device primitive 和本地 peer |
| 核心对象 | p2pResources/shmResources、ncclConnect、ncclConnInfo、protocol buffers |
| 主流程 | P2P canConnect/setup/connect；不可用时 SHM canConnect/setup/connect |
| 状态与生命周期 | IPC/shared mappings 随 connector；free 时撤销映射和 Host/GPU 资源 |
| 并发、同步与所有权 | GPU producer/consumer 通过 head/tail/FIFO 同步；映射资源不能早于 kernel 回收 |
| 异常与退出 | IPC、mapping 或 P2P access 失败时返回并允许上层选择其他 Transport |
| 性能与可观测 | P2P/SHM 日志、nvidia-smi topo、Profiler；静态路径不能证明实际带宽 |
| 源码证据与遗留 | 源码事实：p2pCanConnect/p2pSendSetup/p2pRecvSetup、shmCanConnect/shmSendSetup/shmRecvSetup。待运行：NVLink/PCIe/SHM 路径 |

~~~plantuml
@startuml
title T2 本地 Transport 选择
start
if (p2pCanConnect?) then (yes)
  :p2pSend/RecvSetup;
  :交换 IPC/P2P connect info;
  :p2pSend/RecvConnect;
else (no)
  if (shmCanConnect?) then (yes)
    :shmSend/RecvSetup;
    :建立共享内存映射;
    :shmSend/RecvConnect;
  else (no)
    :交回上层继续 NET 候选;
  endif
endif
stop
@enduml
~~~

### 设计说明与源码锚点

P2P 和 SHM 都服务于本地主机，但物理路径不同。P2P 可通过 CUDA P2P/IPC 直接映射
peer GPU 资源；SHM 使用共享 Host memory 协调或搬运。两者最终都把协议 buffer 和
同步字段写入 connector，从而让 primitive 使用统一接口。

- src/transport/p2p.cc：p2pCanConnect、p2pSendSetup、p2pRecvSetup、p2pTransport
- src/transport/shm.cc：shmCanConnect、shmSendSetup、shmRecvSetup、shmTransport
- src/include/device.h：ncclConnInfo

## T3. NET Transport 连接、缓冲区与注册契约

### 设计卡

| 字段 | 内容 |
|---|---|
| 设计目标 | 将跨节点 connector 映射到 Proxy 和 ncclNet 资源，建立 protocol buffers、request slots 和注册句柄 |
| 非目标 | 不展开 net_ib 内部 QP 状态转换；不证明实际启用 GDR |
| 输入 | graph/NIC 选择、peerInfo、channel/connIndex、ncclNet properties |
| 输出与停止边界 | send/recv resources、proxy connection、registered buffers 和 connector conn；停止在 NET connector connected |
| 上下游接口 | 上游为 T1；下游为 Proxy progress、ncclNet backend 和 Device FIFO |
| 核心对象 | send/recv resources、ncclNetAttr、proxyConn、ncclConnInfo、buffs/head/tail/connFifo |
| 主流程 | canConnect → send/recv setup → Proxy setup/connect → buffer allocation/register → connector publish |
| 状态与生命周期 | NET resources 随 connector；network request 则按 Proxy operation 循环创建/完成 |
| 并发、同步与所有权 | GPU 与 Proxy 共享 FIFO/head/tail；Proxy 持有 ncclNet request，用户持有 payload 生命周期 |
| 异常与退出 | NET plugin、listen/connect、allocation 或 registration 失败时清理对称资源 |
| 性能与可观测 | NET/PROXY 日志、注册命中、GDR read/write、buffer size；需运行验证真实路径 |
| 源码证据与遗留 | 源码事实：net canConnect/sendSetup/recvSetup、populateCommNetAttrs、netTransport。待运行：GDR 与 staging |

~~~plantuml
@startuml
title T3 NET Transport 资源契约
participant "Transport Setup" as Setup
participant "NET send/recv setup" as Net
participant "Proxy Service" as Proxy
participant "ncclNet Backend" as Backend
participant "ncclConnector" as Conn
Setup -> Net : graph/peer/channel
Net -> Proxy : setup/connect request
Proxy -> Backend : listen/connect/accept
Backend --> Proxy : sendComm/recvComm
Proxy -> Backend : regMr(protocol/user buffers)
Backend --> Proxy : mhandle
Proxy -> Conn : buffs/head/tail/connFifo\ntransportResources
@enduml
~~~

### 设计说明与源码锚点

NET Transport 是 NCCL collective runtime 与 ncclNet ABI 的适配层。它决定哪些 buffer
由 GPU primitive 访问，哪些资源由 Proxy 维护，以及 send/recv progress 回调如何取得
backend comm/request。是否使用 GPU Direct RDMA 取决于注册和 backend 能力，必须由运行证据确认。

- src/transport/net.cc：canConnect、sendSetup、recvSetup、populateCommNetAttrs、netTransport
- src/include/transport.h：NET transport 接口
- src/include/device.h：ncclConnInfo
- src/include/net.h、src/include/plugin/nccl_net.h：ncclNet 接口

## 本卷连接状态机

~~~plantuml
@startuml
title Connector 生命周期
[*] --> Empty
Empty --> Selected : canConnect
Selected --> SetupLocal : setup
SetupLocal --> MetadataExchanged : bootstrap/proxy exchange
MetadataExchanged --> Connected : connect
Connected --> InUse : kernel/proxy operations
InUse --> Connected : operation complete
Connected --> Freed : communicator teardown
Freed --> [*]
Selected --> Failed : no transport/setup error
SetupLocal --> Failed : connect error
@enduml
~~~
