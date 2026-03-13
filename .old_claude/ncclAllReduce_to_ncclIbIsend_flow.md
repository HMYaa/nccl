# NCCL 执行流程：从 ncclAllReduce 到 ncclIbIsend

本文档追踪了 NCCL AllReduce 操作从用户 API 调用到底层 RDMA 发送的完整执行路径，并标注了 GPU Kernel 与 CPU Proxy 线程之间的关键交接点。

---

## 📊 完整执行流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     第一阶段：用户 API 调用 (CPU Host)                     │
└──────────────────────────────────────────────────────────────────────────┘

[1] 用户调用 ncclAllReduce()
    📍 位置：src/collectives/all_reduce.cc:11
    📌 运行在：CPU Host 线程

    ↓ 创建 ncclInfo 结构体，记录操作参数

[2] ncclEnqueueCheck(&info)
    📍 位置：src/collectives/all_reduce.cc:17
    📌 运行在：CPU Host 线程
    功能：参数检查、启动 Group 机制

    ↓ 进入 Group 模式（支持批量操作聚合）

[3] taskAppend(comm, info)
    📍 位置：src/enqueue.cc:1338-1435
    📌 运行在：CPU Host 线程
    功能：将任务添加到 comm->tasks.collQueue
    创建：ncclTaskColl 结构体，保存到 comm->memScoped 内存栈


┌──────────────────────────────────────────────────────────────────────────┐
│              第二阶段：任务调度与 Kernel 准备 (CPU Host)                   │
└──────────────────────────────────────────────────────────────────────────┘

[4] ncclLaunchPrepare(comm)
    📍 位置：src/enqueue.cc:889-977
    📌 运行在：CPU Host 线程
    功能：规划 Kernel 启动方案

    ↓ 调用 scheduleCollTasksToPlan()

[5] scheduleCollTasksToPlan(comm, plan, &nWorkBudget)
    📍 位置：src/enqueue.cc:469-583
    📌 运行在：CPU Host 线程
    功能：
    - 选择算法（Ring/Tree）和协议（Simple/LL/LL128）
    - 计算 channel 数量、线程数
    - 调用 computeColl() 生成 workElem 和 proxyOp

    ↓

[6] computeColl(info, &workFuncIndex, &workElem, &proxyOp)
    📍 位置：src/enqueue.cc:1162-1265
    📌 运行在：CPU Host 线程
    关键计算：
    - getAlgoInfo(): 选择 Ring/Tree 算法
    - getPatternInfo(): 确定通信模式（ncclPatternRingTwice）
    - 计算 chunkSize、nSteps
    - 填充 proxyOp 结构体（后续传给 Proxy 线程）

    ↓

[7] addCollToPlan(comm, plan, ...)
    📍 位置：src/enqueue.cc:248-332
    📌 运行在：CPU Host 线程
    功能：
    - 将 workElem 添加到 plan->channels[c].workQueue
    - 将 proxyOp 添加到 plan->channels[c].proxyOpQueue
    - 选择负载最小的 channel（负载均衡）


┌──────────────────────────────────────────────────────────────────────────┐
│           第三阶段：上传工作到 GPU & 启动 Kernel (CPU -> GPU)              │
└──────────────────────────────────────────────────────────────────────────┘

[8] uploadWork(comm, plan)
    📍 位置：src/enqueue.cc:746-809
    📌 运行在：CPU Host 线程
    功能：
    - 将 ncclWork 从 Host 内存拷贝到 GPU 可见的 workFifoHeap
    - 设置 plan->workHead 指针（指向 GPU 端的工作队列头）
    关键：workHeap[ix & ixMask] = q->work  // 写入 FIFO

    ↓ cudaMemcpy 到 GPU 可见内存

[9] ncclLaunchKernel(comm, plan)
    📍 位置：src/enqueue.cc:987-996
    📌 运行在：CPU Host 线程
    功能：启动 CUDA Kernel

    关键代码：
    ```cpp
    dim3 grid = {plan->channelCount, 1, 1};  // 每个 channel 一个 block
    dim3 block = {plan->threadPerBlock, 1, 1};
    void *args[3] = {&comm->devComm, &plan->channelMask, &plan->workHead};
    ncclStrongStreamLaunchKernel(..., plan->kernelFn, grid, block, args, 0);
    ```

    ↓ Kernel 函数：NCCL_KERN_NAME(AllReduce, RING, SIMPLE, Sum, dtype)


┌──────────────────────────────────────────────────────────────────────────┐
│  ⚡ 交接点 #1：GPU Kernel 开始执行，从 workHead 读取任务                   │
└──────────────────────────────────────────────────────────────────────────┘

[10] GPU Kernel 执行（所有 NCCL GPU 线程）
     📍 位置：src/collectives/device/all_reduce.cu（编译后的 kernel）
     📌 运行在：GPU Device

     Kernel 内部流程：
     ① 每个 block 对应一个 channel
     ② 从 workHead 读取 ncclWork 结构体
     ③ 调用 Primitives<T, RedOp, Fan, Direct, Proto>::run()
     ④ 执行 Ring AllReduce 算法：
        - 通过 sizesFifo/offsFifo 与 Proxy 线程通信
        - 使用 Primitives 进行 send/recv/reduce

     关键数据结构（GPU 可见）：
     - ncclConnInfo::sizesFifo  // GPU 写入，告诉 Proxy 要发送的数据大小
     - ncclConnInfo::offsFifo   // Proxy 写入，告诉 GPU 数据在 buffer 的偏移
     - ncclConnInfo::tail/head  // 用于 GPU-Proxy 同步的进度指针


┌──────────────────────────────────────────────────────────────────────────┐
│              第四阶段：Proxy 线程启动 (CPU Host)                           │
└──────────────────────────────────────────────────────────────────────────┘

[11] uploadProxyOps(comm, plan)
     📍 位置：src/enqueue.cc:811-843
     📌 运行在：CPU Host Stream（或 hostStreamPlanCallback）
     功能：
     - 将 proxyOp 提交到 comm->proxyState
     - 调用 ncclProxySaveOp() 保存操作

     ↓

[12] ncclProxySaveOp(comm, op, NULL)
     📍 位置：src/proxy.cc:375-424
     📌 运行在：CPU Host 线程
     功能：根据 pattern（Ring/Tree）确定需要 Proxy 的方向

     示例（Ring AllReduce）：
     - pattern = ncclPatternRingTwice
     - SaveProxy(channel, proxyRecv, ring->prev, op, 0)  // 从前一个 rank 接收
     - SaveProxy(channel, proxySend, ring->next, op, 0)  // 向后一个 rank 发送

     ↓

[13] ncclLocalOpAppend(comm, proxyConn, proxyOp)
     📍 位置：src/proxy.cc:293-352
     📌 运行在：CPU Host 线程
     功能：
     - 将 proxyOp 添加到 proxyOps->pool
     - 调用 ncclProxyPost() 唤醒 Proxy 线程

     ↓

[14] ncclProxyPost(pool, nextOps, nextOpsEnd)
     📍 位置：src/proxy.cc:280-291
     📌 运行在：CPU Host 线程
     功能：
     - 更新 pool->nextOps
     - pthread_cond_signal(&pool->cond)  // 唤醒 Proxy 线程


┌──────────────────────────────────────────────────────────────────────────┐
│  ⚡ 交接点 #2：Proxy 线程被唤醒，开始轮询 GPU 的 FIFO                      │
└──────────────────────────────────────────────────────────────────────────┘

[15] Proxy 线程主循环（独立的 CPU 线程）
     📍 位置：src/proxy.cc（Proxy 线程函数，未在前面展示）
     📌 运行在：CPU Proxy 线程（每个 localRank 一个）

     工作机制：
     ① 从 pool->nextOps 获取 proxyOp
     ② 调用 ProxyAppend(state, op)
     ③ 创建 ncclProxyArgs，调用 progress 函数
     ④ progress 函数轮询：
        - 读取 GPU 写入的 sizesFifo（GPU 告诉 Proxy：有数据要发送）
        - 调用底层传输层函数（ncclIbIsend）
        - 完成后更新 offsFifo（Proxy 告诉 GPU：传输完成）


┌──────────────────────────────────────────────────────────────────────────┐
│         第五阶段：IB Verbs 调用（底层 RDMA 发送）(CPU Proxy)               │
└──────────────────────────────────────────────────────────────────────────┘

[16] ncclIbIsend(sendComm, data, size, tag, mhandle, request)
     📍 位置：src/transport/net_ib.cc:1117-1185
     📌 运行在：CPU Proxy 线程

     关键流程：
     ① 等待对端 Recv 就绪（通过 FIFO 机制）
        - 检查 comm->fifo[slot][0].idx（对端是否 post recv）
        - 如果未就绪，返回 NULL，下次轮询时重试

     ② 创建 ncclIbRequest
        - req->type = NCCL_NET_IB_REQ_SEND
        - req->send.data = data  // GPU 内存地址（GDR）
        - req->send.lkey = mr->lkey  // Memory Region key

     ③ 调用 ncclIbMultiSend(comm, slot)
        📍 位置：src/transport/net_ib.cc:1080-1115
        功能：分片发送（支持多 QP）

        关键代码：
        ```cpp
        for (int q=0; q<comm->nqps; q++) {
          for (int r=0; r<nreqs; r++) {
            // 设置 RDMA Write Work Request
            comm->sges[r].addr = (uint64_t)reqs[r]->send.data + offset;
            comm->sges[r].lkey = reqs[r]->send.lkey;
            comm->wrs[r].wr.rdma.remote_addr = slots[r].addr + offset;
            comm->wrs[r].wr.rdma.rkey = slots[r].rkey;  // 对端的 rkey
          }

          // ⚡ 关键调用：提交到 IB Verbs 驱动
          ibv_post_send(comm->qps[q], comm->wrs, &bad_wr);
        }
        ```

     ④ ibv_post_send() 将 Work Request 提交到 QP 的 Send Queue
        - 硬件（RDMA NIC）接管，执行 RDMA Write
        - DMA 引擎直接从 GPU 内存读取数据（通过 GDR）
        - 数据通过 PCIe -> IB HCA -> Network -> 对端 HCA -> 对端 GPU 内存
```

---

## 🔄 GPU Kernel 与 CPU Proxy 的交接点详解

### **交接点 #1：GPU 写入 sizesFifo**

**位置：** GPU Kernel 内部（Primitives）

**机制：**
```cpp
// GPU 端（Device 代码）
__device__ void sendPeer(int offset, int nelem) {
  connInfo->sizesFifo[step % NCCL_STEPS] = nelem * sizeof(T);  // 写入大小
  __threadfence_system();  // 确保对 CPU 可见
  connInfo->tail++;  // 更新进度
}
```

**CPU 端（Proxy 线程）：**
```cpp
// Proxy 轮询 sizesFifo
while (connInfo->tail > state->step) {
  int size = connInfo->sizesFifo[state->step % NCCL_STEPS];  // 读取大小
  ncclIbIsend(..., size, ...);  // 调用 IB send
  state->step++;
}
```

### **交接点 #2：Proxy 更新 offsFifo**

**CPU 端（Proxy 线程）：**
```cpp
// RDMA 完成后
connInfo->offsFifo[step % NCCL_STEPS] = offset;  // 告诉 GPU buffer 位置
connInfo->head++;  // 更新进度
```

**GPU 端（Device 代码）：**
```cpp
// GPU 等待 Proxy 完成
__device__ void waitSend() {
  while (connInfo->head <= expectedStep) {  // 等待 Proxy 更新
    // spin wait
  }
}
```

---

## 🏗️ 硬件视角的关键路径

```
用户数据 (Host Memory)
   ↓ cudaMemcpy
GPU 内存 (Device Memory)
   ↓ GDR (GPU Direct RDMA) 注册
IB HCA DMA Engine  ← ibv_post_send() 触发
   ↓ PCIe TLP (Transaction Layer Packet)
IB HCA Send Queue (QP SQ)
   ↓ RDMA Write 操作
InfiniBand Network / RoCE
   ↓ 交换机转发（PFC 流控）
对端 IB HCA Receive Queue
   ↓ RDMA Write 直接写入
对端 GPU 内存 (GDR)
```

---

## 📌 关键文件位置总结

| 阶段 | 文件 | 行号 | 功能 |
|------|------|------|------|
| API 入口 | `src/collectives/all_reduce.cc` | 11-18 | ncclAllReduce() |
| 任务入队 | `src/enqueue.cc` | 1338-1435 | taskAppend() |
| 调度规划 | `src/enqueue.cc` | 1162-1265 | computeColl() |
| 上传工作 | `src/enqueue.cc` | 746-809 | uploadWork() |
| 启动 Kernel | `src/enqueue.cc` | 987-996 | ncclLaunchKernel() |
| GPU Kernel | `src/collectives/device/primitives.h` | 1-142 | Primitives 类 |
| Proxy 保存 | `src/proxy.cc` | 375-424 | ncclProxySaveOp() |
| Proxy 追加 | `src/proxy.cc` | 243-278 | ProxyAppend() |
| IB 发送 | `src/transport/net_ib.cc` | 1117-1185 | ncclIbIsend() |
| IB Verbs | `src/transport/net_ib.cc` | 1104 | ibv_post_send() |

---

## 🔑 关键概念总结

### Rank 到 QP 的映射
- 每个 rank pair 使用独立的 QP（Queue Pair）
- QP 在 `ncclIbConnect()` 时创建（`src/transport/net_ib.cc`）
- 支持多 QP 并行传输（`comm->nqps`，通常为 1-4）

### 内存注册（GDR）
- GPU 内存通过 `ibv_reg_mr()` 注册到 IB 网卡
- 返回 `lkey`（本地访问）和 `rkey`（远程访问）
- 注册后 RDMA NIC 可直接访问 GPU 内存（绕过 CPU）

### Proxy 线程与 ibv_post_send 的交互
- Proxy 线程轮询 GPU 的 `sizesFifo`（检查是否有新数据）
- 一旦检测到数据就绪，调用 `ncclIbIsend()`
- `ncclIbIsend()` 检查对端 recv 是否就绪（通过共享 FIFO）
- 就绪后调用 `ibv_post_send()` 提交 RDMA Write Work Request

### FIFO 通信机制
- `sizesFifo`: GPU → Proxy，传递数据大小
- `offsFifo`: Proxy → GPU，传递 buffer 偏移
- `tail`/`head`: 双向进度同步指针
- 采用环形缓冲区（`% NCCL_STEPS`）避免竞争

---

## 📚 扩展阅读

### 相关数据结构
- `ncclWork`: GPU 可见的工作描述符
- `ncclProxyOp`: Proxy 线程的操作描述符
- `ncclConnInfo`: GPU-Proxy 通信的共享内存结构
- `ncclIbRequest`: IB 传输请求

### 算法模式
- `ncclPatternRing`: Ring 单向传递
- `ncclPatternRingTwice`: Ring AllReduce（两次环形传递）
- `ncclPatternTreeUpDown`: Tree AllReduce（二叉树上行下行）

### 协议类型
- `NCCL_PROTO_SIMPLE`: 标准协议，适用于大消息
- `NCCL_PROTO_LL`: Low Latency 协议，适用于小消息
- `NCCL_PROTO_LL128`: LL128 协议，折衷方案

---

**文档生成时间:** 2026-01-03
**NCCL 版本:** 基于 2.14.3-1
**作者:** Claude Code Analysis
