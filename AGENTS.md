## Learned User Preferences

- 讲解 NCCL/底层网络时采用四段式：技术定义（一句话）→ 大白话类比 → 物理逻辑 → 精简代码佐证；拒绝平铺知识点或大段贴码。
- 复杂流程优先用 ASCII 图示（调用栈、FIFO/CQ/QP、状态机）；卡住时用类比降低门槛，同时保持专业准确性。
- 用户具备 RDMA 驱动经验：不要扫盲 QP/CQ/WQE/RoCE 等基础概念；把 NCCL 逻辑直接映射到 Verbs/硬件队列行为。
- 源码解析聚焦物理机制（对齐、cache line、barrier/atomics、控制面 vs 数据面），避免表层语法翻译。
- 讲解抽象概念（进程/rank/communicator、三类 Runtime 等）时例子宜简，先建立心智模型再下钻实现细节。
- 学习节奏：按知识点单点突破；模块啃完后整理复盘 Markdown 到 `doc/`，便于复习。
- 配置 Cursor 时改本地配置，不要改远端仓库配置。
- 回复使用简体中文；技术术语保留英文。

## Learned Workspace Facts

- 本仓库用于深度学习 NCCL 源码，重点包括拓扑图（`src/graph/topo.h`）、IB/RoCE 传输（`src/transport/net_ib.cc`）与 proxy 调度（如 `proxy.cc`）。
- 教学向规则写在 `.cursorrules`（部分学习分支上）：角色为 RDMA/CUDA 资深工程师视角，并规定上述讲解范式与 `doc/` 沉淀流程。
- 架构主链以 `doc/architecture/`（L0/L1）与 `doc/l3/` 为准；DeepWiki/按目录划分的索引仅作跳文件用，不作架构心智模型。
- 学习笔记与串讲文档落在 `doc/`（含 `doc/archify/` 架构图）；入门路径见 `docs/contrib/architecture/learning/zh/`；上游官方文档在 `docs/`。
- 常用学习相关分支包括 `read_code`、`nccl-v2.31.2-learning`；可能通过 `.worktrees/` 挂载其他 checkout（同分支已被 worktree 占用时需 `cd` 到对应 worktree，不能再 `git checkout` 到主目录）。
- 本地工具产物已在 `.gitignore` 忽略（`.serena/`、`.remember/`、`doc/archify/*.visual-check.*`、`*.code-workspace` 等）。
- 已配置 Serena MCP（本仓 project）与 Archify，用于符号检索与架构图生成。
- 刷新 clangd 用的 `compile_commands.json` 可用 `bear --output compile_commands.json -- make -j$(nproc)`（通常先 `make clean`）。
