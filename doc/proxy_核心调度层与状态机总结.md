# NCCL proxy.cc 核心调度层剖析：包工头与邮递员

> 本章节总结了 NCCL Proxy 代理层 (`proxy.cc`) 的核心架构设计。
> 如果说 `net_ib.cc` 是一辆辆负责拉货的卡车，那 Proxy 层就是坐在调度中心、连接 GPU 车间与卡车车队的“超级包工头”。
> 采用“技术定义 + 大白话 + 物理逻辑 + 代码佐证”的四段式范式进行解析。

---

## 知识点一：Proxy 的“双子星”线程模型（控制面 vs 数据面）

### 1. 技术定义
**一句话概括**：将容易发生系统级阻塞的建联控制流（Service 线程）与对延迟极度敏感的数据收发轮询流（Progress 线程）进行物理隔离的并发架构设计。

### 2. 大白话与物理逻辑
**大白话**：NCCL 为每个 GPU 配备了两个 CPU 后台线程：一个坐在前台接电话签合同（**Service 线程**），一个在仓库门口死循环调度货车（**Progress 线程**）。
**物理逻辑**：
- **控制面（Service 线程）**：负责低频、可能阻塞的慢速操作。比如通过 TCP 交换建联信息（QPN、MAC），调用底层的 `ncclNetListen` / `ncclNetConnect`。如果把它和发货混在一起，一次 TCP 拥塞就会导致几百 Gbps 的数据传输彻底停转（Hang 挂死）。
- **数据面（Progress 线程）**：负责高频、对延迟极度敏感的数据收发轮询。它是个纯粹的 `while(true)` 死循环，核心法则是**绝对不阻塞**。它负责把 GPU 交代的任务单转换成底层的 `ibv_post_send` 和 `ibv_poll_cq`。

### 3. 代码佐证
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

### 1. 技术定义
**一句话概括**：一种基于单线程非阻塞状态机（State Machine）和多态函数指针的事件驱动循环，在极高频的轮询中不断推进所有活跃网络通道的发送与接收进度，杜绝任何线程挂起。

### 2. 大白话与物理逻辑
**大白话**：Progress 线程（快递小哥）手上有成百上千张发货单（几百个 Channel 的收发任务）。笨办法是发了一单就死等对面的签收回执（阻塞）。高手的做法是：给每个发货单建一个“状态栏”。循环跑圈，跑到 1 号单时，看一眼当前状态，能发就按发车键，**绝不等待**，立刻转头去看 2 号单；如果转了一圈回来发现 1 号单的 CQE（完成事件）到了，就在状态栏上打个勾，结单销毁。
**物理逻辑**：
- **非阻塞状态机 (State Machine)**：单线程要喂饱多张几百 Gbps 的网卡，必须极致榨干 CPU。将每个收发任务抽象为一个状态机，每次循环仅推进一步。
- **函数指针多态 (`op->progress`)**：NCCL 内部有很多种底层通道（`net`、`shm`、`p2p`）。主循环不写死具体的发货代码，而是调用多态指针，把具体的逻辑下放到诸如 `sendProxyProgress` 这样的专属状态机函数中处理。

### 3. 代码佐证
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

---

## 知识点三：GPU 与 CPU 跨界的桥梁：OpsPool 与无锁信箱

### 1. 技术定义
**一句话概括**：一块预分配在 CPU 锁页内存中、采用 64B 缓存行严格对齐的共享发货单池，充当 GPU Kernel 与 CPU Proxy 线程间跨界通信的无锁（Lock-free）任务中转站。

### 2. 大白话与物理逻辑
**大白话**：GPU 是个在封闭车间里干活的黑工，Proxy 是坐在办公室里的包工头。GPU 干完活了，想要包工头派车来拉货。GPU 不能摇电话（发中断）叫包工头，也不能和包工头共用一把锁。唯一的办法就是：在两者之间放一个完全公开的“信箱”（共享内存），大家约定好规则，谁放纸条、谁拿纸条，全程不加锁，完全靠“手速”和“原子操作”来避免抢破头。
**物理逻辑**：
- **无锁队列（Ring/Linked Buffer）**：在高性能网络中，锁（Mutex）是并发的毒药。NCCL 在 CPU Pinned Memory 中开辟了一个预分配的池子（`ncclProxyOpsPool`）。
- **缓存行对齐（Cache Line Alignment）**：这不仅是为了结构整齐。因为信箱是共享内存，多核 CPU 和 PCIe 设备会疯狂读写。如果两张纸条挨得太近，挤在同一个 CPU 缓存行（64 Bytes）里，一个人写 A 纸条，会导致另一个人正在读的 B 纸条的缓存失效（伪共享 False Sharing），引发巨大的内存延迟惩罚。

### 3. 代码佐证
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

---

## 知识点四：基于 CAS 的无锁链表回收机制

### 1. 技术定义
**一句话概括**：利用 CPU 硬件级的原子比较并交换指令 (Compare-And-Swap)，在不阻塞线程的情况下，安全地解决多个并发实体竞争同一个链表头指针时的读写冲突。

### 2. 大白话与物理逻辑
**大白话**：假设有一个公共的“回收站”，你（Proxy 线程）手里有一叠废纸箱（处理完的空闲任务单），想放回回收站里。笨办法是加锁；**无锁办法（CAS）**是你先看一眼回收站最上面的纸箱编号（A），把你手里的纸箱绑在A上。然后你施展一招“闪电手”去修改总记录。**规则是**：只要在修改的一瞬间，上面还是A，就修改成功；如果A被人拿走了，说明有冲突，闪电手失败。
**物理逻辑**：
- `__sync_val_compare_and_swap` 在底层会被编译为一条 CPU 汇编指令（如 x86 的 `LOCK CMPXCHG`）。
- 它通过直接锁住内存总线或单行 Cache，在一个极小的硬件时钟周期内完成“读-对比-写”三个动作，绝对不会被操作系统调度打断。
- 相比于 `pthread_mutex` 这种会引发内核态陷入和线程睡眠的重量级锁，CAS 是在用户态自旋，延迟是纳秒级的。

### 3. 代码佐证与神级冲突降级
在 `src/proxy.cc` 的 `ncclProxyGetPostedOps` 中，Proxy 线程需要把用完的槽位还给 `pool->freeOps`。
```cpp
// 准备归还槽位
int newFree = freeOp[i];          // Proxy手里要还的这串槽位的"头"
int oldFree = pool->freeOps[i];   // 看一眼当前回收站的"头"

// 动作1：把手里这串的"尾巴"，接在当前回收站的"头"上
pool->ops[freeOpEnd[i]].next = oldFree; 

// 动作2：施展闪电手 (CAS)
// 去比对 pool->freeOps[i] 还是不是 oldFree，是的话就替换成 newFree。
int swap = __sync_val_compare_and_swap(pool->freeOps+i, oldFree, newFree);

// 动作3：处理冲突
if (swap != oldFree) {
  // 失败了！说明在我绑绳子的那几纳秒里，主线程把回收站里的东西全拿走了！
  if (swap != -1) return ncclInternalError; // NCCL断定主线程必定是一把抓空(变-1)
  
  // 既然旧东西没了，我刚才绑的绳子也没用了，剪断绳子
  pool->ops[freeOpEnd[i]].next = -1;
  // 直接强行覆盖，因为现在回收站是空的，只有我手里的这串槽位了
  pool->freeOps[i] = newFree;
}
```

---

## 知识点五：`__builtin_prefetch` 隐藏内存墙延迟

### 1. 技术定义
**一句话概括**：利用编译器底层内置指令，在处理当前链表节点时，强制 CPU 内存控制器提前将下一节点数据从主存预取至 L1 Cache，以掩盖随机内存访问（指针追逐）带来的巨大延迟。

### 2. 大白话与物理逻辑
**大白话**：你去食堂打饭（CPU 运行代码），笨办法是走到窗口才点菜，然后傻等大妈现做（等内存）。而 `__builtin_prefetch` 就是“提前点单系统”。当你还在排队时，提前用手机下单，等轮到你站到窗口时，大妈直接把菜递给你，**你的等待时间变成了 0**！
**物理逻辑（内存墙与指针追逐）**：
- CPU 从 L1 Cache 读数据只要 ~1 纳秒，但从主存 (RAM) 读数据要 ~100 纳秒（差了百倍）。
- 如果是遍历连续数组，现代 CPU 的硬件预取器很聪明，会自动提前搬运后面的数据。但 NCCL 的发货单池子是一个用 `index` 串起来的**链表**。这种在内存里随机跳跃的读法叫“指针追逐”。硬件预取器对这种跳跃完全蒙圈。
- 所以，在处理当前单子时，NCCL 必须使用**软件预取**指令（对应底层汇编如 `PREFETCHT0`），手动下令把 `next` 节点所在的 64 字节缓存行拉进 Cache。

### 3. 代码佐证与高级参数
```cpp
for (int opIndex = state->nextOps; opIndex != -1;) {
  struct ncclProxyOp* peerOp = pool->ops + opIndex;
  
  // 【软件预取】：在处理 peerOp 之前，提前叫 CPU 把下一张单子搬进来！
  if (peerOp->next != -1) __builtin_prefetch(pool->ops + peerOp->next);
  
  NCCLCHECK(ProxyAppend(state, peerOp)); // 这期间 CPU 就可以去安心干活，不被内存 I/O 阻塞
  opIndex = peerOp->next;
}
```
*注：`__builtin_prefetch` 默认等价于 `__builtin_prefetch(addr, 0, 3)`。参数 0 表示只读不写，参数 3 表示高度局部性（请死死把它留在 L1 Cache 里别踢掉）。在每纳秒必争的高性能轮询中，这一句代码往往能白嫖出极高的性能收益。*

---

## 知识点六：探秘 Proxy 层的多态状态机：`net.cc` 收发闭环

在 Proxy 的光速死循环里，多态函数指针 `op->progress` 针对“网络快递（NET）”专门执行了收发状态机，核心实现在 `src/transport/net.cc` 里的 `sendProxyProgress` 和 `recvProxyProgress`。

### 1. 发送状态机 (`sendProxyProgress`) —— 从 GPU 取货并发往网络

#### 1.1 技术定义
**一句话概括**：
一个基于步数 (Steps) 和滑动窗口驱动的异步发送状态机，它负责轮询 GPU 端写入的信箱，将就绪的本地数据通过 `ncclNetIsend` 异步推给网卡，并通过 `ncclNetTest` 非阻塞地回收发送完成事件。

#### 1.2 大白话与物理逻辑
**大白话**：
你（Proxy 快递小哥）守在发货流水线旁。你每次跑圈路过，只按顺序干三件事，**绝不傻等**：
1. **盯源头（GPU）**：看一眼传送带上，老黑工 (GPU) 有没有把新造好的包裹放上来？如果放上来了，你就给它贴个条（记下地址和长度）。
2. **推给车队（NIC）**：把贴好条的包裹，扔给网卡卡车（调 `Isend` 发车）。
3. **查回执（CQE）**：回头去查一下之前发走的卡车，对方签收了没（调 `Test`）？签收了就在账本上划掉一笔。

**物理逻辑（流水线与 Steps）**：
为了让几百 Gbps 的带宽被打满，NCCL 不可能等 1GB 的 Tensor 全算完再发。它把数据切分成多个块（Chunks/Steps），默认最大窗口是 `NCCL_STEPS`（通常是 8 或 64）。
这三步操作分别对应着状态机里的三个水位线指针：
- `sub->posted`：表示 Proxy 已经发现 GPU 产出数据的步数。
- `sub->transmitted`：表示 Proxy 已经成功交给网卡发送的步数。
- `sub->done`：表示网卡已经彻底发完并收到硬件 CQE 的步数。
**永远满足：`done <= transmitted <= posted`。**

#### 1.3 代码佐证：三段式水位线推进
在 `src/transport/net.cc` 的 `sendProxyProgress` 里，有极其工整的三个 `if` 块：

```cpp
// 动作1：检查能不能继续往下推 (post) 任务
// posted < done + NCCL_STEPS 控制了滑动窗口不会爆掉
if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
  // 把 posted 水位线往前推，相当于 Proxy 说：“这批货我接管了”
  sub->posted += args->sliceSteps; 
  args->idle = 0; // 只要我推了水位线，我就不算闲着
  continue;
}

// 动作2：检查 GPU 数据是否真写进了显存？写进了就交由网卡发送！
if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
  // sizesFifo 里面记录了 GPU 实际要发的数据大小
  if (sizesFifo[buffSlot] != -1 && ((*recvTail > (sub->base+sub->transmitted)) || p == NCCL_PROTO_LL)) {
    // 调底层的 isend！这就是你之前在 net_ib.cc 学的那个口子！
    NCCLCHECK(ncclNetIsend(comm, resources->netSendComm, buff, size, resources->rank, mhandle, sub->requests+buffSlot));  
    
    // 如果发送请求成功提交给网卡了
    if (sub->requests[buffSlot] != NULL) {
      sizesFifo[buffSlot] = -1; // 阅后即焚，把信箱清空
      __sync_synchronize();
      sub->transmitted += args->sliceSteps; // 推进 transmitted 水位线
      args->idle = 0;
      continue;
    }
  }
}

// 动作3：检查之前交给网卡的发送任务，到底发完了没有？
if (sub->done < sub->transmitted) {
  int done;
  // 非阻塞轮询！就是去查 CQE！
  NCCLCHECK(ncclNetTest(comm, sub->requests[buffSlot], &done, NULL));
  if (done) {
    sub->done += args->sliceSteps; // 推进 done 水位线
    // 通知 GPU：这块显存的数据我发完了，你可以覆盖它写下一批数据了！
    volatile uint64_t* sendHead = &resources->sendMem->head;
    *sendHead = sub->base + sub->done;
    args->idle = 0;
  }
}
```

### 2. 接收状态机 (`recvProxyProgress`) —— 从网卡收货并通知 GPU

#### 2.1 技术定义
**一句话概括**：
一个基于预投递机制 (Pre-posted Recv) 的异步接收状态机，它负责提前向下层网卡投递接收描述符 (`ncclNetIrecv`)，在轮询到硬件接收完成 (`ncclNetTest`) 后，通过更新基于共享内存的 `Tail` 指针安全地唤醒 GPU Kernel 消费数据。

#### 2.2 大白话与物理逻辑
**大白话**：
收快递和发快递逻辑是反过来的。网卡（快递员）来送货时，你门卫室必须提前准备好空箱子（Recv Buffer），不然货就全掉地上了（这就是网络里的 RNR NAK 报错）。
所以接收状态机也要干三件事：
1. **扔空箱子（Irecv）**：只要门卫室还有空地，就疯狂往网卡那里塞空纸箱，告诉网卡“包裹来了放这里”。
2. **看单子（Test）**：不停地查账本，看看网卡有没有把某个纸箱填满（也就是收到了发送端的 IMM 通知）。
3. **改公告板（更新 Tail）**：纸箱填满了，你就在大门口的公告板上改个数字（更新 Tail 指针）。老黑工（GPU）一直在盯着公告板，一看到数字变大，立刻扑上来把纸箱里的数据搬走算乘加。

**物理逻辑（防止 RNR 挂死）**：
在 RDMA 中，接收端必须在发送端发起 WRITE_WITH_IMM 或者 SEND 之前，提前把 Recv WQE（接收工作队列元素）下发到网卡的 RQ 里。所以 `recvProxyProgress` 总是非常激进地把 `sub->posted` 往前推，只要不超过环形队列的深度（`NCCL_STEPS`），它就会把底层的 `Irecv` 塞满。

#### 2.3 代码佐证：接收端的反向流水线

在 `src/transport/net.cc` 的 `recvProxyProgress` 里，同样是经典的三段式推进：

```cpp
// 动作1：激进地投递接收请求 (Irecv 扔空箱子)
if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
  // 把空内存地址和长度交给底层网卡
  NCCLCHECK(ncclNetIrecv(comm, resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, requestPtr));  
  if (*requestPtr) {
    sub->posted += args->sliceSteps;
    args->idle = 0;
  }
}

// 动作2：轮询有没有收到数据
if (subGroup->posted > subGroup->received) {
  int done;
  // 查 CQE，看看带 imm 的包到了没
  NCCLCHECK(ncclNetTest(comm, subGroup->requests[step%NCCL_STEPS], &done, sizes));
  if (done) {
    subGroup->received += args->sliceSteps; // 推进 received 水位线
    // 【高能联动】：如果开了 GDR，这里就是触发 Loopback QP RDMA READ Flush 的地方！
    if (useGdr) {
      NCCLCHECK(ncclNetIflush(comm, resources->netRecvComm, subCount, ptrs, sizes, mhandles, ...));
    }
    args->idle = 0;
  }
}

// 动作3：数据已经安全落入显存，立刻修改公告板 (Tail) 通知 GPU
if (subGroup->received > subGroup->transmitted) {
  int done = 1;
  // 如果刚才触发了 Iflush，这里还要等 Flush 的 CQE 回来才算数
  NCCLCHECK(ncclNetTest(comm, request, &done, NULL));
  if (done) {
    sub->transmitted += args->sliceSteps;
    __sync_synchronize(); // 严格的内存屏障
    // 修改公告板！GPU Kernel 里的 while(*tail < XXX) 瞬间被解阻！
    volatile uint64_t* recvTail = &resources->recvMem->tail;
    *recvTail = sub->base + sub->transmitted; 
    args->idle = 0;
  }
}
```

#### 💡 灵魂拷问：单线程怎么扛得住极高的并发？

无论是 `ncclNetIsend`、`ncclNetIrecv` 还是 `ncclNetTest`，这三个由底层的 `net_ib.cc` 提供的函数，它们**没有任何一个是阻塞的**！
- 调用 `Isend`，只要网卡 SQ 有空位，立马写一条 WR 进去就返回，绝不等发完。
- 调用 `Test`，也就是执行底层的 `ibv_poll_cq`，如果硬件没有回执，它直接把 `done` 置为 `0` 然后瞬间返回。
- 只要 `done == 0`，代码根本不会去推那个水位线（`sub->done += args->sliceSteps`），而是直接跑完这个函数的余下逻辑。

这种“榨干每个 CPU 周期去填补所有的 I/O 等待”的思路，正是高性能网络编程的巅峰浪漫！

---

## 知识点七：实战解析：Proxy 发货单池如何以“全量拉取”彻底绞杀 ABA 问题

在《高性能系统并发与锁_物理学笔记》中，我们提到了 Lock-Free (无锁) 队列中极其致命的 **ABA 问题**（CAS 检查指针值相等，但其实对象已经被回收掉包了，导致 `next` 指针野掉）。
但在 `proxy.cc` 的发货单池 (`ncclProxyOpsPool` 的 `freeOps`) 中，NCCL 通过一种极其巧妙的 **“全量拉取 (Take-All)”** 策略，在底层只用最基础的整数索引，就彻底消灭了 ABA 问题！

### 1. 技术定义
**一句话概括**：采用“全量拉取 (Take-All)”策略的单消费者/单生产者（SPSC/MPSC）无锁链表设计，消费者在出队时将 CAS 的目标值固化为常量（`-1`），避免了对 `next` 指针的依赖，从而从物理根源上消除了悬空指针和 ABA 漏洞的生存空间。

### 2. 大白话与物理逻辑
**大白话**：
传统无锁队列是“每次从回收站（`freeOps`）拿最上面那一个纸箱”。你必须先看一眼第一个纸箱，记住它的下一个是谁，然后把全局指针改成下一个。就在这一瞬间，容易发生 ABA 问题（纸箱被别人换了，下一个的指针野了）。
**NCCL 的黑魔法是**：“老子不一个一个拿，老子直接把整个回收站端走（把回收站强行替换成 `-1`）！”。端回自己办公室后，关起门来，再一个一个慢慢拆着用。
由于“拿纸箱”这个动作的最终状态永远是把回收站设为“空 (`-1`)”，它根本不需要在跨线程的原子操作中去读取什么 `next` 指针，所以就算发生了 ABA，它依然能正确地把那一坨不管是什么的东西全端走。

**物理逻辑（Consumer 与 Producer 的完美互补）**：
- `pool->freeOps[localRank]` 是一个共享的整数数组，存储着空闲链表的头节点索引（Index）。
- **消费者 (Main Thread，负责取单子)**：它只执行一步 `XCHG`（或者 CAS 换成 `-1`）。这一步在硬件总线上是将共享的头节点索引直接霸占，并留下一句“这里空了 (`-1`)”。
- **生产者 (Proxy Thread，负责还单子)**：由于它是唯一能往 `freeOps` 里塞新链表的线程。如果它尝试用 CAS 把新链表挂上去时失败了，**只有一种可能：就是消费者刚才把回收站端空了（变成了 `-1`）**。既然消费者已经端走了旧的，那生产者干脆把新链表直接设为全新的回收站头节点即可。

### 3. 代码佐证与底层推演

**【消费者 (Main Thread) —— 暴力端走整个池子】**
位于 `ncclLocalOpAppend` 中，当它发现本地没有空闲单子时：
```cpp
int freeOp;
// 1. 等待回收站有东西
while ((freeOp = pool->freeOps[comm->localRank]) == -1) sched_yield();

int freeOpNew;
// 2. 【无视 ABA 的神级操作】
// 它根本不去读 freeOp->next！它直接尝试用 CAS 把 freeOps 替换成 -1 (空)！
// 即使发生并发冲突，只要替换成功，就相当于把整个链表据为己有。
while ((freeOpNew = __sync_val_compare_and_swap(pool->freeOps+comm->localRank, freeOp, -1)) != freeOp) {
    freeOp = freeOpNew; // 失败了就重试，但目标依然是 -1
}

// 拿到一长串链表后，把第一个拿出来自己用，剩下的塞进本地缓存 proxyOps->freeOp 里慢慢用
opIndex = freeOp;
op = pool->ops+opIndex;
proxyOps->freeOp = op->next;
```

**【生产者 (Proxy Thread) —— 极简的冲突降级】**
位于 `ncclProxyGetPostedOps` 中，当 Proxy 执行完任务，要把空闲单子还回回收站时：
```cpp
int newFree = freeOp[i];          // Proxy手里要还的新链表头
int oldFree = pool->freeOps[i];   // 看一眼回收站现在的头

// 动作1：把新链表的尾巴，接在现在的头上
pool->ops[freeOpEnd[i]].next = oldFree;

if (oldFree == -1) {
    pool->freeOps[i] = newFree;
} else {
    // 动作2：施展 CAS，尝试把回收站的头改成我的新链表
    int swap = __sync_val_compare_and_swap(pool->freeOps+i, oldFree, newFree);
    
    // 动作3：【硬核断言逻辑】如果失败了会发生什么？
    if (swap != oldFree) {
        // 失败了？唯一的可能就是被刚才的消费者 (Main Thread) 把整个池子端成了 -1 ！！
        if (swap != -1) return ncclInternalError; 
        
        // 既然旧东西被端走了，我刚才接的尾巴也就没用了，赶紧断开（防止内存泄露/串链）
        pool->ops[freeOpEnd[i]].next = -1;
        // 因为现在池子肯定是空的 (-1)，我直接把我的新链表放进去就行了！连第二次 CAS 都不需要！
        pool->freeOps[i] = newFree;
    }
}
```

**总结**：正是这种基于 `Index` + `常量 -1 交换` 的模式，让这套极其高频（每秒百万次）的跨界池化系统跑得既没有死锁，也绝不可能爆发 ABA 崩溃。完全对应了并发笔记中“压缩冲突面”的终极奥义！
