# NCCL 源码学习仓（分支 `nccl-v2.31.2-learning`）

本仓用于深度学习 NCCL 源码。基线 `v2.31.2-1`（`7b83616d`）。用户具备 RDMA 驱动开发经验，目标是吃透拓扑、IB/RoCE 传输、proxy 调度与无锁并发。

## 讲解方式

- 四段式：技术定义（一句话）→ 大白话类比 → 物理逻辑 → 精简代码佐证。拒绝平铺知识点、大段贴码。
- 不扫盲 QP/CQ/WQE/RoCE；把 NCCL 逻辑直接映射到 Verbs / 硬件队列行为。聚焦对齐、cache line、barrier/atomics、控制面 vs 数据面。
- 复杂流程（调用栈、FIFO/CQ/QP、状态机）优先 ASCII 图。
- 用户说「继续」时给规划 + 分层串讲，每层讲完停下等确认；不出自测题/quiz。
- 证据口径：区分「源码事实 / 设计归纳 / 待运行验证」；静态分支不等于运行路径，算法/协议/传输选择必须留给运行证据。

## 学习流程

- 单点突破：一次一个知识点；先「大脑建模」（心智模型），再下钻实现。
- 新模块先建模数据结构（字段按功能分组），再看函数；数据结构优先于函数。
- 核实字段用途：`编号 + 字段名 + 一句话结论 + 证据 文件:行号`，讲清谁写/谁读/生命周期。
- 动手前先 `git log --oneline -- <文件>` 看该结构是否已注释过，避免重复建模。

## 沉淀与提交

- 结构体注释直接写进 `src/` 源码，中文，沿用 `comm.h` 风格：`// ============ 功能分组 ============`、`// ---- (a) 子视图 ----`；不改代码语义。
- 学习笔记在独立仓 `/home/yangxw/yxw/playbook/notes/ccl/summer/`（`00`–`07` + `90-pain-points`），按 README「写作约定」的 02 模板写：正文 150–220 行、每小节 ≤1 个代码块 ≤15 行、引用写 `文件 · 符号名`；细节过「粒度闸门」才进正文，否则放 `deep/`；新篇写完更新 README 阅读顺序表。笔记与 nccl 仓源码注释分开提交。
- 架构图用 Archify 落 `doc/archify/`。
- 学习节点结束：整理改动、挑可提交内容、分批提交；commit message 中文，如「补充 X 结构的注释」。
- 配置 Cursor 只改本地（`~/.cursor/`），不改远端仓库配置。

## 仓库地图

- 架构主链入口：`doc/architecture/nccl-architecture-index.md`（L0/L1）→ `doc/archify/nccl-l2-design.md`（L2）→ `doc/l3/nccl-l3-design-index.md`（L3）。DeepWiki/目录索引只作跳文件用。
- 核心结构：`src/include/comm.h`（`ncclComm`）、`src/include/transport.h`（vtable / `ncclPeerInfo`）、`src/bootstrap.cc`（`bootstrapState`）、`src/include/graph.h`（`ncclTopoGraph`）、`src/include/proxy.h`。
- 重点源码：`src/graph/topo.h`、`src/transport/net_ib/`（v2.31 已拆目录：`init.cc` 设备发现、`connect.cc` QP 建链、`p2p.cc` 数据面、`reg.cc` MR 注册、`gdr.cc`）、`src/proxy.cc`。
- Init 心智轴：成员信息 → 物理拓扑 → 算法拓扑 → Channel → 连接；落地 Bootstrap → Communicator → Topology/channel → Transport(P2P/SHM/NET) → Proxy setup/connect。
- 入门路径 `docs/contrib/architecture/learning/zh/`；上游官方文档 `docs/`；分支 `read_code` 上有 `.cursorrules`（同一讲解范式的完整版）。
- 同分支被 worktree 占用时 `cd` 到对应 worktree，勿在主目录强行 `git checkout`。
- 符号检索用 Serena MCP。刷新 clangd：`make clean && bear --output compile_commands.json -- make -j$(nproc)`。

## Learned User Preferences

- 串讲核心数据结构时沿用「路线表 + 契约 + 快递单」心智模型（Channel/Connector → ConnInfo → ProxyArgs），函数视为在填这三张表。
- 读代码按用户点名的单点（符号/函数）一步步下钻，先对齐理解再展开下一段。
- 长函数/大段代码解释时，先给一个具体例子对齐理解，再展开细节；避免一上来长文。
- 用户反馈难懂或不清时，改用更直白的例子、ASCII 图示或货单类比重讲，勿继续堆长梳理/字段枚举。
- 细节粒度由用户定：用户说「先粗 / 细节后续」时只讲粗主链；细项等用户点名再下钻。
- 用户要求查看 window/终端窗格内容并点名 Herdr 时，用 Herdr skill 读取并直接回答，勿绕路猜测。

## Learned Workspace Facts

- 本仓用 Makefile 构建，无可用 CMakeLists；勿用 CMake Tools 配工程，clangd 依赖 `bear` + `make` 生成 `compile_commands.json`。
- `groupLaunch` 默认走 `groupLaunchLegacy`（`NCCL_ENQUEUE_REARCH_ENABLE` 默认 0）；只有显式设为 1 才进 EnqueueRearch。
