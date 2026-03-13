# NCCL net_ib.cc 核心数据面与资源管理总结

> 本文档总结了 NCCL IB 传输层 (`net_ib.cc`) 中关于建联、数据面发送逻辑和内存资源管理的核心知识点。
> 采用“大白话 + 物理逻辑 + 代码佐证”的范式进行解析。

---

## 知识点一：FIFO 动态握手机制

### 1. 是什么？为什么？
**大白话**：传统发快递，寄件人得先打个电话（TCP握手）问收件人：“你的新地址是哪？用哪把钥匙开门？”这太慢了。NCCL 的做法是：收件人（Receiver）在自己家准备好仓库后，直接跑到寄件人（Sender）家门口的“信箱”（FIFO）里，塞一张小纸条：“货往这个地址发，这是钥匙（rkey），这次是第 N 批货（idx）”。寄件人一开门看到纸条，二话不说直接发货。
**物理逻辑**：
- 在 AI 训练中，因为每次 AllReduce 传入的 Tensor 显存地址可能不同，如果每次都用 TCP Socket 交换这些信息，系统调用开销会极其恐怖。
- **解法**：通过 `RDMA WRITE`，把接收端的 `addr` 和 `rkey` 等元数据，单向暴力写入发送端的预分配内存（FIFO）中。这叫做 **Out-of-Band (OOB) 到 In-Band 硬件握手**的降维打击。
- **防鬼连（ABA 问题）与乱序撕裂**：用单调递增的 `idx` 避免发送端读到旧数据；强制使用 `__sync_synchronize()` 内存屏障，且保证纸条大小是 32B 对齐的，防止 PCIe Relaxed Ordering 造成的“信纸被撕成两半，只看到后半截”的窘境。

### 2. 代码佐证
接收端（填纸条并扔进信箱）：
```cpp
// 接收端本地填好纸条
localElem[i].addr = (uint64_t)data[i];
localElem[i].rkey = mr->rkey;
// ...
localElem[i].idx = comm->remFifo.fifoTail + 1; // 递增的 idx

// 用 RDMA_WRITE 把纸条扔进 Sender 的信箱 (fifo)
wr.opcode = IBV_WR_RDMA_WRITE;
NCCLCHECK(wrap_ibv_post_send(comm->qps[0], &wr, &bad_wr));
```
发送端（低头看信箱）：
```cpp
// 期待的下一封信的编号
int idx = comm->fifoHead + 1;
// 轮询信箱里的 idx，只要没对上就 return 走人（非阻塞）
if (slots[0].idx != idx) return ncclSuccess; 

// 看到了！马上加内存屏障，防止后面的 addr/rkey 被 CPU 乱序提前读了旧的
__sync_synchronize(); 
uint64_t remote_addr = slots[r].addr;
```

---

## 知识点二：CQE 轮询与完成机制 (`ncclIbTest`)

### 1. 是什么？为什么？
**大白话**：货发出去或者收进来后，怎么知道干完了？传统程序会留个电话让网卡“打电话（发中断）”通知 CPU。但打电话太慢了，CPU 不如直接搬个小板凳坐在网卡办事处门口，死死盯着告示板（CQ），只要告示板上贴出单子（CQE），CPU 立刻拿走处理。
**物理逻辑**：
- 高性能网络必须避免内核中断，采用用户态轮询（Polling）。
- Send 和 Recv 在建联时被绑定到了**同一个 CQ（完成队列）**。
- **发送端完成**：等待自己刚才发出的那条带有 `SIGNALED` 标志的 WR 的 CQE。
- **接收端完成**：等待对端发过来的 `WRITE_WITH_IMM` 触发本地的一个 Recv CQE。
- **events 计数器做屏障**：如果开了多个 QP（多条货车），每个货车回来都会触发一个 CQE，`events` 减 1。只有当 `events == 0` 时，才向更上层报告“这次集体发送/接收全部完成”。

### 2. 代码佐证
```cpp
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
    // 如果计步器归零，说明这批货（无论多少个QP）都搞定了
    if (r->events == 0) {
        *done = 1;
        return ncclSuccess;
    }

    // 死盯着告示板（拉取 CQE）
    NCCLCHECK(wrap_ibv_poll_cq(r->verbs->cq, 4, wcs, &wrDone));
    if (wrDone == 0) return ncclSuccess; // 啥也没发生，下次再来

    for (int w=0; w<wrDone; w++) {
        struct ibv_wc *wc = wcs+w;
        // ... 从 wc->wr_id 里解出是哪个请求
        
        if (req->type == NCCL_NET_IB_REQ_SEND) {
            req->events--; // 发送端：发出去的货车回来了，计步器减一
        } else {
            if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                // 接收端：收到对端的 IMM 通知，把里面夹带的大小(size)拿出来
                sizes[0] = wc->imm_data;
            }
            req->events--; // 接收端：对方的一辆货车卸完货了，计步器减一
        }
    }
}
```

---

## 知识点三：TCP OOB 与 QP 建联状态机

### 1. 是什么？为什么？
**大白话**：要让两张网卡能通过高速公路（RDMA）直连，一开始它们是互不认识的。它们必须先找个“传统的电话亭”（TCP Socket）互换一下车牌号、高速公路入口（QPN, LID, GID）。交换完毕后，各自回到自己车上，把状态拨到“准备收货（RTR）”和“准备发货（RTS）”，这才算建交完成。
**物理逻辑**：
- NCCL 没有使用官方的 `rdma_cm` 库来建联，因为那玩意太重了，且偏向阻塞模型。NCCL 极其强调非阻塞状态机。
- **建联核心数据 (`ncclIbQpInfo`)**：包含了 QPN、LID、GID（RoCE 需要）、MTU，以及**发送端最初的那个 FIFO 信箱的物理地址和 rkey**（这就把前面的 FIFO 握手给串起来了）。
- **延迟建联 (`ncclSendCheck`)**：发送端通过 TCP 把自己的信息发给接收端后，不能傻等着对方回复（会阻塞）。所以发送端直接 return。等后来第一次准备调 `isend` 发数据时，再去偷看一眼 TCP 里面有没有收到对方的回信。收到了，才把 QP 的状态推到 RTS（Ready To Send）。

### 2. 代码佐证
通过 TCP 交换 `ncclIbQpInfo`，并推进 QP 状态机：
```cpp
// 接收端 (Accept)
// 1. 从 TCP Socket 里拿出发送端的车牌号等信息
NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->sock, &remQpInfo, ...));

// 2. 拿到对方的 QPN 后，把自己的 QP 状态推到 RTR (准备收) 和 RTS (准备发)
for (int q=0; q<rComm->nqps; q++) {
    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn[q], &remQpInfo));
    NCCLCHECK(ncclIbRtsQp(qp));
}

// 3. 最关键的一步：接收端把发送端 FIFO 信箱的地址和 rkey 记在心里！
// 以后它就往这个地址里扔 FIFO 握手纸条！
rComm->remFifo.rkey = remQpInfo.fifoRkey;
rComm->remFifo.addr = remQpInfo.fifoAddr;

// 4. 把自己的车牌号通过 TCP 发回给发送端
NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->sock, &localQpInfo, ...));
```

---

## 知识点四：多 QP (Multi-QP) 与 128B 对齐切分

### 1. 是什么？为什么？
**大白话**：发一批货（比如4KB），不是用一辆大卡车送，而是把它拆成好几截，用多辆小货车（多个 QP）同时送。而且货物的打包标准必须是 128 字节一托盘（对齐），不能有半个托盘。
**物理逻辑**：
- **多 QP 提速**：在同一条物理连接上，分配多个 QP。不同 QP 的报文会有不同的路由哈希特征，能够充分利用交换机的多条路径（ECMP/LAG），提高聚合带宽。
- **128B 对齐**：NCCL 内部的协议层（LL / LL128）在逻辑上把数据按照 128 字节一块来处理。如果不按 128B 切分，接收端就会收到“半块”数据，处理逻辑极其复杂。按 128B 切分保证了多 QP 分段的边界刚好落在协议层的处理边界上。

### 2. 代码佐证
在 `ncclIbMultiSend` 中，同一条发送指令会在 `comm->nqps` 个 QP 上循环发送：
```cpp
const int align = 128;
for (int q=0; q<comm->nqps; q++) {
    for (int r=0; r<nreqs; r++) {
        // 先按 QP 数均分，再向上取整到 128 的倍数
        int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, comm->nqps), align) * align;
        // 算出当前 QP 负责的这一段有多长
        int length = std::min(reqs[r]->send.size - reqs[r]->send.offset, chunkSize);
        
        // ... 填充 WR 的地址和长度
    }
    // 把属于当前 QP 的这一段提交给网卡
    NCCLCHECK(wrap_ibv_post_send(comm->qps[q], comm->wrs, &bad_wr));

    // 移动指针，下一轮给下一个 QP 发下一段
    for (int r=0; r<nreqs; r++) {
        reqs[r]->send.offset += chunkSize;
        comm->sges[r].addr += chunkSize;
        comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }
}
```

---

## 知识点二：Adaptive Routing (AR) 与 WR 分离

### 1. 是什么？为什么？
**大白话**：发送端发了一堆数据，最后得告诉接收端“发完了”。正常情况下，这个“发完了”的纸条是贴在最后一个包裹上的。但在启用了自适应路由（AR）的快递网络里，包裹可能乱序到达，贴着纸条的包裹反而先到了，这会引起误会。所以，必须把纸条单独剥离出来，用一辆绝对排在最后面的车送。
**物理逻辑**：
- 接收端依靠带有 `imm_data` 的 `WRITE_WITH_IMM` 报文来触发 CQE，作为“数据已全部到达”的完成信号。
- 如果数据 WR 和带 IMM 的 WR 混在一起，大包会被网卡拆分。在 AR 环境下，带有 IMM 的最后一个分片可能比前面的纯数据分片更早到达对端，导致接收端提前读取到未写完的脏数据。
- **解法**：剥离！发送纯数据的 `RDMA_WRITE` 链，最后追加一个独立的、0 字节的 `WRITE_WITH_IMM`。利用同一 QP 的发送队列（SQ）严格保序的物理特性，确保 IMM 报文一定在所有数据报文都发射完毕后再发射。

### 2. 代码佐证
```cpp
// 找到最后一个 WR 的位置
struct ibv_send_wr* lastWr = comm->wrs + nreqs - 1;

// 如果开了多 recv，或者包太大（超过 AR 阈值），执行 WR 分离
if (nreqs > 1 || reqs[0]->send.size > ncclParamIbArThreshold()) {
    lastWr++; // 凭空多造出一个 WR
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
}

// 这个最后追加的 WR 专门用来做完成通知，不带数据（0字节）
lastWr->wr_id = wr_id;
lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
lastWr->imm_data = immData;
lastWr->next = NULL;
lastWr->send_flags = IBV_SEND_SIGNALED; // 只有它是 signaled 的，产生发送完成事件
```

---

## 知识点三：MR 注册三条路径与池化复用 (MR Cache)

### 1. 是什么？为什么？
**大白话**：办“网卡通行证”（注册 MR）是一件要跑遍系统多个部门、极慢的事情（涉及锁页表、写 IOMMU）。如果每次发数据都去办一次，网络就卡死了。所以 NCCL 建了一个“通行证缓存中心”（MR Cache），办过一次的通行证不轻易销毁，给好几个人同时借着用。
**物理逻辑**：
- `ibv_reg_mr` 开销巨大，属于控制面慢速路径。
- **按页对齐化零为整**：以物理页为单位进行缓存，只要处于同一页的访问，都算命中。
- **智能指针式的生命周期 (refs)**：AI 训练中，一块大 Tensor 会被切成多块，交给多个 Channel 并发处理。这导致同一块物理缓存在同一时刻被多个实体“借走”。`refs` 就是用来追踪当前有几个 Channel 正在依赖这块内存。只有当所有人都不用了（`refs == 0`），才真正找操作系统销毁。

### 2. 代码佐证
注册时的缓存命中与引用计数增加：
```cpp
uintptr_t addr = (uintptr_t)data & -pageSize; // 对齐到页
size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;

pthread_mutex_lock(&ncclIbDevs[verbs->dev].lock);
for (int slot=0; /*true*/; slot++) {
    // 命中缓存：这块页以前注册过
    if (cache->slots[slot].addr == addr && cache->slots[slot].pages == pages) {
        cache->slots[slot].refs += 1; // 多了一个通道在用，引用加一
        *mhandle = (void*)cache->slots[slot].mr; // 直接拿旧钥匙走人，极速返回
        res = ncclSuccess;
        goto returning;
    }
    // ... 如果没命中，走 dma-buf / Relaxed Ordering / 标准注册 三条路之一，并存入 slots 且 refs=1
}
```

释放时的虚假注销（只减引用）：
```cpp
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
    for (int i=0; i < cache->population; i++) {
        if (mhandle == cache->slots[i].mr) {
            // 引用计数减一，只有等于 0 时，才真正调底层注销
            if (0 == --cache->slots[i].refs) {
                wrap_ibv_dereg_mr(...);
                // 把最后一个元素移过来填补空洞，O(1) 删除
                memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
            }
            return ncclSuccess; // 大多数情况下直接返回了
        }
    }
}
```

### 3. 三种底层注册路径
如果 Cache 没命中，NCCL 会根据硬件能力选路：
1. `ibv_reg_dmabuf_mr`：驱动支持的 GPU 显存直通车，性能最好。
2. `ibv_reg_mr_iova2` (+ `IBV_ACCESS_RELAXED_ORDERING`)：开启 PCIe 乱序特权，提高总线吞吐。
3. `ibv_reg_mr`：标准的兜底方案。

### 4. 深度释疑：为什么需要维持多达 N 个的 refs 计数？
**疑问**：既然是复用，每次发完还回去不就行了？为什么同一块内存的 `refs` 可能会涨到 8 甚至更高，他们真的在“同时”使用吗？

**物理逻辑解谜**：
- **化整为零的并发分发**：在 AI 训练中，框架发给 NCCL 的往往是一块几百 MB 甚至 GB 级的巨大 Tensor。为了打满网络带宽，NCCL 不会只用 1 个 Channel 傻发，而是把这块大内存切成无数个几百 KB 的小块（Chunks），交给 8 个或更多的 Channel **同时并发发送**。
- **空间与时间的重叠**：这 8 个并发的 Channel，它们各自持有一个小 Chunk 的首地址去调 `regMr`。但在底层，这些小块地址经过**“页对齐”**后，落入的完全是同一个大物理页范围！
- **生命周期的保护伞**：因为是并发发送，Channel 0 可能跑得快，瞬间把它的 Chunk 发完了；但此时 Channel 1 还在苦苦往外发。如果 Channel 0 发完就直接找系统 `ibv_dereg_mr`（解除网卡和物理内存的绑定），那 Channel 1 正在飞行的 DMA 请求会瞬间触发“内存保护错误（Memory Protection Fault）”，网卡当场宕机。
- **结论**：`refs` 记录的是**当前还有几个正在运转的独立流水线（Channel）需要这块物理内存不被操作系统收回**。只有当走得最慢的那个 Channel 也干完活（`refs == 0`），这把“网卡钥匙”才能真正安全归还系统。这在本质上就是一个**给网卡硬件资源（MR）用的 `std::shared_ptr`**：有了并发，就必须有 `std::shared_ptr`（也就是 `refs`）来保护这块内存不被跑得快的那个实体提前释放掉。

---

## 知识点四：GDR Flush（PCIe 写入可见性保证）

### 1. 是什么？为什么？
**大白话**：网卡（快递员）把数据写进了 GPU 显存（老王的地下室），但实际上数据可能还卡在 PCIe 交换机（一楼大厅）的写缓存里，网卡却已经向 CPU 报告说“送到了”。如果 CPU 这时候立刻叫 GPU 去读，GPU 会读到旧数据的乱码。怎么解决？网卡在送完货后，强行发起一次从 GPU 读取 1 个字节的读请求。因为同一通道“不能超车”，PCIe 必须先把前面囤积的写数据全“刷（Flush）”进地下室，才能把这 1 字节读出来。看到这 1 字节，网卡才真正确信：数据落盘了。
**物理逻辑**：
- **Posted Write 语义的鸿沟**：PCIe 总线上的写（Memory Write）是 Posted（邮寄）的，只要途径的交换机缓存收下，就立刻回 ACK 触发 CQE，**此时数据并未真正进入 GPU DRAM**。
- **Non-Posted 读的屏障效应**：PCIe 规定读（Memory Read）是 Non-Posted 的，且必须与前面的写操作**严格保序**。一次针对 GPU 的 RDMA READ，就相当于一次物理级别的 **PCIe Write Flush**。
- **GPU 自己为什么不 Flush？**：因为这批数据是第三方网卡通过 PCIe 塞给 GPU 的，GPU 本身处于无知状态。`__threadfence()` 或 `cudaStreamSynchronize()` 只能同步 GPU 自己发起的读写，管不了网卡。只能靠网卡自己发起读来兜底。

### 2. 深度释疑：为什么是“接收端”发起？什么是 Loopback QP？
**疑问**：数据是发送端发过来的，那是谁负责去发起这个 Flush（去读这一字节）呢？什么是“自己连自己的 Loopback QP”？

**物理逻辑解谜**：
- **是谁发起 Flush？是接收端 (Receiver)！**
  发送端 (Sender) 的任务在发完 `WRITE_WITH_IMM` 后就彻底结束了。数据到达接收端网卡后，是**接收端网卡**通过 PCIe 写入**接收端 GPU 显存**的。所以，卡在 PCIe 缓存里的也是接收端的缓存。必须由**接收端**发起一笔从自己网卡到自己 GPU 的读请求，才能疏通这条拥堵的本地 PCIe 链路。
- **为什么需要 Loopback QP (自己连自己)？**
  - 我们需要网卡发起一笔读操作（RDMA READ）。但在正常通信中，RDMA READ 是用来跨机去读别人内存的。
  - 既然接收端只想通过“读”这个物理动作产生“刷缓存”的副作用，完全不关心读出来的数据交给谁，那最省事的办法就是**在本地网卡上建一个专门的虚假通道（QP），把它初始化的目的端（dest_qpn）强行填成它自己的编号。**
  - 当接收端把 READ 请求发给这个 Loopback QP 时，网卡一看目标是自己，包就不会发到网线上，而是直接在网卡内部绕一圈，对着自己的 GPU 显存发起一笔纯粹的 PCIe Read，完美达到 Flush 的目的！

### 3. 灵魂拷问：为什么要读那块地？读出来的内容有用吗？
**疑问**：既然只是为了产生“副作用”刷缓存，那是随便读哪里都行吗？读出来的那个字节有用吗？

**物理逻辑解谜**：
- **从哪读？（必须是刚写过的那块 GPU 显存）**
  PCIe 规范的保序规则是基于路径和地址的。你必须去读 `data[last]`（刚才接收到的最后一块有效数据的地址）。只有沿着“网卡到这块特定 GPU 显存”的同一条物理路径发起读操作，才能强行把这条路径上囤积的写请求给“挤”下去。如果随便读一块无关的内存，走的是别的 PCIe 通道，完全起不到 Flush 的作用。
- **读到哪去？读什么不重要！（垃圾桶机制）**
  我们发起读操作，不是为了获取数据，而是为了**白嫖“读”这个物理动作的副作用**。
  在代码里，读回来的数据目的地 `sg_list` 被指向了 `comm->gpuFlush.sge`，这块内存是在初始化时分配在主机 CPU 上的一块极小的废空间（`hostMem`）。
  这 1 个字节读回来后，直接扔进这个废纸篓，**没有任何人会去检查它的值**。它的存在仅仅是为了满足网卡 RDMA READ 接口“必须有个目的地”的语法要求。
- **结论**：**读目标必须精准，但读内容一文不值。**我们在乎的是，当网卡发出“这个 READ 操作已完成”的 CQE 时，我们在物理上 100% 确信——货已经老老实实躺在老王的地下室里了。此时再去叫 GPU Kernel 起床干活，绝对不会读到残缺的脏数据。

### 4. 代码佐证
在 `ncclIbIflush` 中，只对最后一块到达的有效数据发起一笔 1 字节的 READ 窃读：
```cpp
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  // ... 找到最后一块有效数据 last
  
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  
  // remote_addr 填 GPU 显存里刚被写过的地址
  wr.wr.rdma.remote_addr = (uint64_t)data[last];
  wr.wr.rdma.rkey = mr->rkey;
  
  // 读出来的数据放到 CPU 一个无关紧要的 gpuFlush.sge 垃圾桶里，我们不关心内容
  wr.sg_list = &comm->gpuFlush.sge;
  wr.num_sge = 1;
  
  // 核心：强制发起 RDMA READ，迫使 PCIe 刷缓存
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  // 用一个自己连自己的 Loopback QP 提交这个读请求
  NCCLCHECK(wrap_ibv_post_send(comm->gpuFlush.qp, &wr, &bad_wr));

  // ...
}
```
**优化点**：这会增加一次额外的 RDMA 往返延迟。因此代码里非常克制，通过 `sizes[i]` 找到 `last`，只在所有 chunk 写完后，对最后一个非空的地址发起一次 Flush，而不是每块都 Flush。