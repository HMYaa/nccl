# NCCL proxy.cc 核心调度层剖析：包工头与邮递员

> 本章节总结了 NCCL Proxy 代理层 (`proxy.cc`) 的核心架构设计。
> 如果说 `net_ib.cc` 是一辆辆负责拉货的卡车，那 Proxy 层就是坐在调度中心、连接 GPU 车间与卡车车队的“超级包工头”。

---

## 知识点一：Proxy 的“双子星”线程模型（控制面 vs 数据面）

### 1. 是什么？为什么？
**大白话**：NCCL 为每个 GPU 配备了两个 CPU 后台线程：一个坐在前台接电话签合同（**Service 线程**），一个在仓库门口死循环调度货车（**Progress 线程**）。
**物理逻辑**：
- **控制面（Service 线程）**：负责低频、可能阻塞的慢速操作。比如通过 TCP 交换建联信息（QPN、MAC），调用底层的 `ncclNetListen` / `ncclNetConnect`。如果把它和发货混在一起，一次 TCP 拥塞就会导致几百 Gbps 的数据传输彻底停转（Hang 挂死）。
- **数据面（Progress 线程）**：负责高频、对延迟极度敏感的数据收发轮询。它是个纯粹的 `while(true)` 死循环，核心法则是**绝对不阻塞**。它负责把 GPU 交代的任务单转换成底层的 `ibv_post_send` 和 `ibv_poll_cq`。

### 2. 代码佐证
在 `proxy.cc` 中，这两个线程是一前一后被拉起来的，名称和职责泾渭分明：
```cpp
// 1. 创建 Service 线程 (负责建联等慢速行政工作)
ncclSetThreadName(comm->proxyState.thread, "NCCL Service %2d", comm->cudaDev);
pthread_create(&comm->proxyState.thread, NULL, ncclProxyService, comm);

// 2. 创建 Progress 线程 (负责高速收发数据的死循环)
ncclSetThreadName(state->thread, "NCCL Progress%2d", comm->cudaDev);
pthread_create(&state->thread, NULL, ncclProxyProgress, comm);
```

---

## 知识点二：高速运转的 Progress 死循环与状态机

### 1. 是什么？为什么？
**大白话**：Progress 线程（快递小哥）手上有成百上千张发货单（几百个 Channel 的收发任务）。笨办法是发了一单就死等对面的签收回执（阻塞）。高手的做法是：给每个发货单建一个“状态栏”。循环跑圈，跑到 1 号单时，看一眼当前状态，能发就按发车键，**绝不等待**，立刻转头去看 2 号单；如果转了一圈回来发现 1 号单的 CQE（完成事件）到了，就在状态栏上打个勾，结单销毁。
**物理逻辑**：
- **非阻塞状态机 (State Machine)**：单线程要喂饱多张几百 Gbps 的网卡，必须极致榨干 CPU。将每个收发任务抽象为一个状态机，每次循环仅推进一步。
- **函数指针多态 (`op->progress`)**：NCCL 内部有很多种底层通道（`net`、`shm`、`p2p`）。主循环不写死具体的发货代码，而是调用多态指针，把具体的逻辑下放到诸如 `sendProxyProgress` 这样的专属状态机函数中处理。

### 2. 代码佐证
**死循环的核心骨架** (`ncclProxyProgress`)：
```cpp
void* ncclProxyProgress(void *comm_) {
  struct ncclProxyProgressState* state = &comm->proxyState.progressState;

  // 快递员光速死循环的心脏
  while (state->stop == 0 && *comm->abortFlag == 0) {
    int idle = 1;

    // 动作 1：遍历执行待办墙（state->active）上的所有任务单，非阻塞推进一步
    ncclResult_t ret = progressOps(comm, state, state->active, &idle);

    // 动作 2：如果刚才一圈跑下来发现没活干（闲置了）
    if (idle) {
      int added = 0;
      // 去 GPU 的发件箱看看有没有新任务，有就贴到待办墙上
      ret = ncclProxyGetPostedOps(comm, &added);

      // 如果连信箱都是空的，主动交出 CPU 时间片，别占着茅坑不拉屎
      if (added == 0) sched_yield(); 
    }
  }
}
```

**任务是如何被推进一步的** (`progressOps`)：
```cpp
static ncclResult_t progressOps(...) {
  struct ncclProxyArgs* op = opStart;
  while (op) {
    // 【核心多态调用】：调用这个任务专属的推进函数！
    // 如果是网络发送任务，它会指向 src/transport/net_send.cc 里的 sendProxyProgress
    // 进而触发 ncclNetIsend -> ibv_post_send (发车) 或 ncclNetTest -> ibv_poll_cq (等完成)
    NCCLCHECK(op->progress(comm, op));

    // 如果这个任务机走到了终点（完成）
    if (op->state == ncclProxyOpNone) {
      // 从待办墙链表上摘除销毁
      NCCLCHECK(removeOp(state, &op, &prevOp));
    } else {
      // 还没干完，留着下一圈循环继续推
      op = op->next;
    }
  }
}
```

### 3. 深度释疑：数据面接力棒的闭环
如果你把 GPU、Proxy 和网卡串联起来，一个包裹的物理旅程是这样的：
1. **GPU 投递**：GPU 算完数据，把发货单（`ncclProxyOp`）塞进信箱。
2. **Proxy 查收**：Proxy 在死循环的 `ncclProxyGetPostedOps` 阶段捞出单子，贴上墙。
3. **推进一步（发车）**：下一圈循环走到这个单子，调 `op->progress`，内部触发 `ncclNetIsend` -> `ibv_post_send`。函数立马返回，不堵塞。
4. **推进一步（等收据）**：再下一圈循环，内部触发 `ncclNetTest` -> 轮询 CQ (`ibv_poll_cq`)。如果拿到了回执（CQE），修改状态为完成。单子被销毁。
---

## 知识点三：GPU 与 CPU 跨界的桥梁：OpsPool 与无锁信箱

### 1. 是什么？为什么？
**大白话**：GPU 是个在封闭车间里干活的黑工，Proxy 是坐在办公室里的包工头。GPU 干完活了，想要包工头派车来拉货。GPU 不能摇电话（发中断）叫包工头，也不能和包工头共用一把锁。唯一的办法就是：在两者之间放一个完全公开的“信箱”（共享内存），大家约定好规则，谁放纸条、谁拿纸条，全程不加锁，完全靠“手速”和“原子操作”来避免抢破头。
**物理逻辑**：
- **无锁队列（Ring/Linked Buffer）**：在高性能网络中，锁（Mutex）是并发的毒药。NCCL 在 CPU Pinned Memory 中开辟了一个预分配的池子（`ncclProxyOpsPool`）。
- **缓存行对齐（Cache Line Alignment）**：这不仅是为了结构整齐。因为信箱是共享内存，多核 CPU 和 PCIe 设备会疯狂读写。如果两张纸条挨得太近，挤在同一个 CPU 缓存行（64 Bytes）里，一个人写 A 纸条，会导致另一个人正在读的 B 纸条的缓存失效（伪共享 False Sharing），引发巨大的内存延迟惩罚。
- **无锁回收机制（CAS）**：读完信件后，空槽位要还给回收站。此时如果有其他线程也在回收，就用硬件级的 `Compare-And-Swap` (CAS) 原子指令来裁决，失败了就重试或特殊处理。

### 2. 代码佐证
**神级优化：64 字节强制对齐** (`src/include/proxy.h`)：
```cpp
struct ncclProxyOp {
  struct ncclProxyConnection* connection; // 走哪条物理通道
  int type;                               // 任务类型 (Send/Recv)
  void* data;                             // 数据源/目的地址
  int size;                               // 发多大
  struct ncclProxyOp *next;               // 连成一长串单子
  // ... 其他属性
};

// 确保每个结构体不多不少正好占据 64 字节（一个主流 CPU Cache Line 大小）
static_assert(sizeof(struct ncclProxyOp) == 64, "Keep ProxyOp aligned with cache lines for effective prefetch");
```

**Proxy 捞单子与 CPU 预取指令（Prefetch）** (`src/proxy.cc`)：
```cpp
// 从信箱里拉出一串单子
for (int opIndex = state->nextOps; opIndex != -1;) {
  struct ncclProxyOp* peerOp = pool->ops + opIndex;
  
  // 【性能神兵】在处理当前单子时，提前向 CPU 下令：
  // 帮我把下一张单子从缓慢的内存吸到极速的 L1 Cache 里！
  if (peerOp->next != -1) __builtin_prefetch(pool->ops + peerOp->next);
  
  // 把单子贴到 Proxy 自己的待办墙 (state->active) 上
  NCCLCHECK(ProxyAppend(state, peerOp));
  
  opIndex = peerOp->next;
}
```

**无锁（Lock-free）还槽位机制**：
```cpp
for (int i=0; i<comm->localRanks; i++) {
  int newFree = freeOp[i];          // 我要还的一串槽位
  int oldFree = pool->freeOps[i];   // 信箱里当前的回收站头

  // 我把我要还的链表的尾巴，接上当前的回收站头
  pool->ops[freeOpEnd[i]].next = oldFree; 

  // 无锁核心：我试图原子地把回收站的头改成我的 newFree
  // 只要这个瞬间没有人动过 freeOps，就成功！
  int swap = __sync_val_compare_and_swap(pool->freeOps+i, oldFree, newFree);
  
  if (swap != oldFree) {
    // 冲突了！CAS 失败，说明就在刚才的几纳秒里，主线程也回收了单子
    // 处理冲突：认怂，这串我不接了，直接暴力的切断
    pool->ops[freeOpEnd[i]].next = -1;
    pool->freeOps[i] = newFree;
  }
}
```

### 3. 深度释疑：为什么 NCCL 快？都在“抠”细节！
看看这段代码都在抠什么：
1. **防伪共享**（`static_assert 64B`）：这属于对 CPU Cache 架构的降维理解。
2. **隐藏内存延迟**（`__builtin_prefetch`）：在代码还没运行到那里时，通过编译器指令强制调度 CPU 内存控制器提前搬运数据。
3. **消除内核挂起**（`__sync_val_compare_and_swap`）：能用 1 条汇编指令（LOCK CMPXCHG）解决的并发冲突，绝不惊动操作系统去睡眠。