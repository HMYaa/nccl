# NCCL net_ib.cc 核心数据面与资源管理总结

> 本文档总结了 NCCL IB 传输层 (`net_ib.cc`) 中关于数据面发送逻辑和内存资源管理的核心知识点。
> 采用“大白话 + 物理逻辑 + 代码佐证”的范式进行解析。

---

## 知识点一：多 QP (Multi-QP) 与 128B 对齐切分

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