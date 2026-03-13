# NCCL 内存分配 API 完整梳理

> 本文档系统梳理 NCCL 中所有内存分配 API，解释它们的区别、使用场景和底层实现。

---

## 📋 快速对比表

| API | 内存位置 | 底层实现 | GPU可访问 | 页对齐 | 主要用途 |
|-----|---------|---------|----------|--------|---------|
| `ncclCalloc` | **Host（CPU）** | `malloc` + `memset` | ❌ 否 | ❌ 否 | CPU端数据结构 |
| `ncclCudaHostCalloc` | **Host（CPU）** | `cudaHostAlloc` (Mapped) | ✅ **是** | ✅ 是 | CPU↔GPU共享数据 |
| `ncclCudaMalloc` | **Device（GPU）** | `cudaMalloc` | ✅ 是 | ✅ 是 | GPU Kernel数据 |
| `ncclCudaCalloc` | **Device（GPU）** | `cudaMalloc` + `cudaMemsetAsync` (同步) | ✅ 是 | ✅ 是 | GPU Kernel数据（需清零） |
| `ncclCudaCallocAsync` | **Device（GPU）** | `cudaMalloc` + `cudaMemsetAsync` (异步) | ✅ 是 | ✅ 是 | GPU Kernel数据（异步清零） |
| `ncclIbMalloc` | **Host（CPU）** | `posix_memalign` (页对齐) | ❌ 否 | ✅ **是** | IB Verbs 内存注册 |

---

## 🔍 详细分类解析

### 一、Host 端内存分配（CPU 内存）

#### 1. `ncclCalloc` - 普通主机内存

**定义位置**：`src/include/alloc.h:43-54`

**底层实现**：
```c
void* p = malloc(nelem*sizeof(T));  // 标准 malloc
memset(p, 0, nelem*sizeof(T));      // 清零
```

**特点**：
- ✅ **CPU 可访问**：CPU 代码可以直接读写
- ❌ **GPU 不可访问**：GPU Kernel 无法直接访问
- ❌ **非固定内存**：可能被操作系统交换到磁盘
- ❌ **非页对齐**：不保证页边界对齐

**使用场景**：
- CPU 端的数据结构（如 `ncclComm`、`ncclTask`、`ncclProxyArgs`）
- 临时缓冲区（不需要 GPU 访问）
- 拓扑信息、连接信息等 CPU 端元数据

**典型用法**：
```c
// src/init.cc:1347
NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);  // 分配异步任务结构体

// src/init.cc:331
NCCLCHECK(ncclCalloc(&comm, 1));  // 分配通信器结构体
```

**类比**：就像普通的"办公桌抽屉"，只有 CPU（你）能打开，GPU（同事）看不到里面。

---

#### 2. `ncclCudaHostCalloc` - CUDA 固定主机内存（Pinned Memory）

**定义位置**：`src/include/alloc.h:22-35`

**底层实现**：
```c
cudaHostAlloc(ptr, nelem*sizeof(T), cudaHostAllocMapped);  // CUDA 固定内存 + 映射
memset(*ptr, 0, nelem*sizeof(T));  // 清零
```

**特点**：
- ✅ **CPU 可访问**：CPU 代码可以直接读写
- ✅ **GPU 可访问**：GPU Kernel 可以通过 PCIe 直接访问（**关键特性**）
- ✅ **固定内存（Pinned）**：不会被操作系统交换到磁盘，保证物理地址稳定
- ✅ **页对齐**：自动页对齐，适合 DMA 传输
- ⚡ **高性能**：CPU↔GPU 传输速度快（无需复制，直接映射）

**使用场景**：
- **CPU↔GPU 共享数据**：需要 GPU Kernel 直接读取的 Host 端数据
- **FIFO 队列**：`workFifoHeap`、`workFifoDone` 等 GPU 需要轮询的共享内存
- **中止标志**：`abortFlag` - GPU Kernel 需要实时检查是否中止
- **网络缓冲区**：Proxy 线程和 GPU Kernel 共享的缓冲区

**典型用法**：
```c
// src/init.cc:1335 - abortFlag 必须用 Pinned Memory
// GPU Kernel 在执行时会定期检查 abortFlag（见 src/collectives/device/common.h:198）
// GPU 必须能直接通过 PCIe 读取这个标志，不能等 CPU 复制（实时性要求）
NCCLCHECKGOTO(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1), res, fail);

// src/init.cc:461 - workFifoHeap：GPU Kernel 和 Proxy 共享的工作队列
NCCLCHECK(ncclCudaHostCalloc(&comm->workFifoHeap, comm->workFifoDepth));

// src/init.cc:467 - workFifoDone：GPU 需要轮询的完成标志
NCCLCHECK(ncclCudaHostCalloc(&comm->workFifoDone, MAXCHANNELS));
```

**为什么不能用 `ncclCalloc`？**
```c
// ❌ 错误示例（如果这样写会怎样？）
NCCLCHECK(ncclCalloc((uint32_t**)&comm->abortFlag, 1));  // 普通 malloc

// 问题：
// 1. GPU Kernel 无法直接访问 malloc 分配的内存（需要通过 cudaMemcpy 复制）
// 2. 如果 GPU 需要实时检查 abortFlag，延迟会很高（需要等待 CPU 复制）
// 3. 可能错过中止信号，导致 GPU Kernel 继续执行不应该执行的操作
```

**类比**：就像"共享公告板"，CPU 和 GPU 都能直接看到，不需要"传纸条"（cudaMemcpy）。

---

### 二、Device 端内存分配（GPU 内存）

#### 3. `ncclCudaMalloc` - GPU 设备内存（不初始化）

**定义位置**：`src/include/alloc.h:76-88`

**底层实现**：
```c
cudaMalloc(ptr, nelem*sizeof(T));  // 只分配，不初始化
```

**特点**：
- ✅ **GPU 可访问**：GPU Kernel 可以直接访问
- ❌ **CPU 不可直接访问**：CPU 需要通过 `cudaMemcpy` 访问
- ⚠️ **不初始化**：内存内容未定义（可能是随机值）
- ✅ **页对齐**：自动页对齐

**使用场景**：
- GPU Kernel 的临时缓冲区（不需要初始化为 0）
- 性能敏感场景（避免不必要的 `memset` 开销）

**典型用法**：
```c
// src/enqueue.cc:837 - 分配 work 队列（不需要清零，GPU 会直接写入）
NCCLCHECK(ncclCudaMalloc(&plan->workHead, nWork));
```

---

#### 4. `ncclCudaCalloc` - GPU 设备内存（同步清零）

**定义位置**：`src/include/alloc.h:91-109`

**底层实现**：
```c
cudaMalloc(ptr, nelem*sizeof(T));                    // 分配
cudaMemsetAsync(*ptr, 0, nelem*sizeof(T), stream);   // 异步清零
cudaStreamSynchronize(stream);                       // 等待清零完成
```

**特点**：
- ✅ **GPU 可访问**：GPU Kernel 可以直接访问
- ❌ **CPU 不可直接访问**：CPU 需要通过 `cudaMemcpy` 访问
- ✅ **初始化为 0**：内存内容保证为 0
- ⚠️ **同步操作**：会阻塞 CPU 线程，等待清零完成
- ✅ **页对齐**：自动页对齐

**使用场景**：
- GPU Kernel 需要初始化为 0 的数据结构
- 初始化阶段（不关心同步开销）

**典型用法**：
```c
// src/init.cc:430 - 分配设备端通信器结构（需要初始化为 0）
NCCLCHECK(ncclCudaCallocAsync(&devCommAndChans, 1, comm->deviceStream.stream));
// 注意：这里实际用的是 Async 版本，但 Calloc 版本也有类似用法
```

---

#### 5. `ncclCudaCallocAsync` - GPU 设备内存（异步清零）

**定义位置**：`src/include/alloc.h:112-125`

**底层实现**：
```c
cudaMalloc(ptr, nelem*sizeof(T));                    // 分配
cudaMemsetAsync(*ptr, 0, nelem*sizeof(T), stream);   // 异步清零（不等待）
// 不调用 cudaStreamSynchronize，立即返回
```

**特点**：
- ✅ **GPU 可访问**：GPU Kernel 可以直接访问
- ❌ **CPU 不可直接访问**：CPU 需要通过 `cudaMemcpy` 访问
- ✅ **初始化为 0**：内存内容保证为 0（在指定 stream 上异步执行）
- ✅ **异步操作**：不阻塞 CPU 线程，立即返回
- ✅ **页对齐**：自动页对齐

**使用场景**：
- GPU Kernel 需要初始化为 0 的数据结构
- **性能敏感场景**：不想阻塞 CPU 线程
- 在特定 CUDA Stream 上执行清零操作

**典型用法**：
```c
// src/init.cc:430 - 在 deviceStream 上异步分配并清零
NCCLCHECK(ncclCudaCallocAsync(&devCommAndChans, 1, comm->deviceStream.stream));

// src/channel.cc:23 - 在 deviceStream 上异步分配通道信息
NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nRanks+1, comm->deviceStream.stream));
```

**与 `ncclCudaCalloc` 的区别**：
- `ncclCudaCalloc`：同步版本，会阻塞 CPU 直到清零完成
- `ncclCudaCallocAsync`：异步版本，立即返回，清零在后台执行

---

### 三、特殊用途内存分配

#### 6. `ncclIbMalloc` - IB Verbs 内存注册专用

**定义位置**：`src/include/alloc.h:168-179`

**底层实现**：
```c
size_t page_size = sysconf(_SC_PAGESIZE);           // 获取页大小
int size_aligned = ROUNDUP(size, page_size);        // 向上对齐到页大小
posix_memalign(&p, page_size, size_aligned);        // 页对齐分配
memset(p, 0, size);                                  // 清零
```

**特点**：
- ✅ **CPU 可访问**：CPU 代码可以直接读写
- ❌ **GPU 不可访问**：GPU Kernel 无法直接访问
- ✅ **页对齐**：**必须页对齐**（`ibv_reg_mr` 的要求）
- ✅ **DONTFORK 标记**：这些页面会被标记为 DONTFORK，避免 fork 时崩溃

**使用场景**：
- **IB Verbs 内存注册**：需要调用 `ibv_reg_mr` 注册的内存
- **QP 信息交换**：连接建立时交换的 QP 信息缓冲区
- **RDMA 缓冲区**：需要注册到 IB 网卡的内存

**为什么必须页对齐？**
```c
// IB Verbs 的 ibv_reg_mr 要求内存必须页对齐
// 如果不对齐，注册会失败或性能很差
// 这些页面会被标记为 DONTFORK，避免 fork 时子进程访问已注册的内存导致崩溃
```

**典型用法**：
```c
// src/transport/net_ib.cc:702 - 分配 IB Send Comm 结构（需要注册到 IB）
NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));

// src/transport/net_ib.cc:759 - 分配 QP 信息缓冲区（用于交换连接信息）
NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(qpInfo)));
```

**类比**：就像"专用保险箱"，必须符合特定规格（页对齐），才能被 IB 网卡"登记"（注册）。

---

## 🔄 内存分配 API 选择决策树

```
需要分配内存
│
├─ CPU 端使用？
│  │
│  ├─ GPU 需要访问？
│  │  │
│  │  ├─ 是 → ncclCudaHostCalloc (Pinned Memory)
│  │  │      └─ 例如：abortFlag, workFifoHeap, workFifoDone
│  │  │
│  │  └─ 否 → 需要 IB 注册？
│  │     │
│  │     ├─ 是 → ncclIbMalloc (页对齐)
│  │     │      └─ 例如：QP 信息缓冲区
│  │     │
│  │     └─ 否 → ncclCalloc (普通 malloc)
│  │            └─ 例如：comm, job, task 等 CPU 数据结构
│  │
│  └─ GPU 端使用？
│     │
│     ├─ 需要初始化为 0？
│     │  │
│     │  ├─ 是 → 性能敏感？
│     │  │  │
│     │  │  ├─ 是 → ncclCudaCallocAsync (异步清零)
│     │  │  │      └─ 例如：devCommAndChans, channel->devPeers
│     │  │  │
│     │  │  └─ 否 → ncclCudaCalloc (同步清零)
│     │  │         └─ 例如：初始化阶段的 GPU 数据结构
│     │  │
│     │  └─ 否 → ncclCudaMalloc (只分配，不初始化)
│     │         └─ 例如：workHead, 临时缓冲区
```

---

## 📊 实际代码示例对比

### 示例 1：`abortFlag` - 为什么必须用 `ncclCudaHostCalloc`？

```c
// src/init.cc:1335
NCCLCHECKGOTO(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1), res, fail);
*comm->abortFlag = 0;

// 为什么不能用 ncclCalloc？
// ❌ ncclCalloc(&comm->abortFlag, 1);  // 错误！

// 原因：
// 1. GPU Kernel 在执行时会定期检查 abortFlag（见 src/collectives/device/common.h:198）
// 2. GPU 必须能直接通过 PCIe 读取这个标志，不能等 CPU 复制（实时性要求）
// 3. ncclCudaHostCalloc 分配的是 Pinned Memory（固定内存），GPU 可以直接访问
// 4. 如果用 malloc，GPU 无法直接访问，需要 cudaMemcpy 复制，延迟高且可能错过中止信号
```

### 示例 2：`job` - 为什么用 `ncclCalloc`？

```c
// src/init.cc:1347
NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);

// 为什么用 ncclCalloc？
// ✅ 因为 job 只在 CPU 端使用，GPU 不需要访问
// ✅ 不需要 Pinned Memory 的开销（Pinned Memory 分配较慢，占用系统资源）
```

### 示例 3：`devCommAndChans` - 为什么用 `ncclCudaCallocAsync`？

```c
// src/init.cc:430
NCCLCHECK(ncclCudaCallocAsync(&devCommAndChans, 1, comm->deviceStream.stream));

// 为什么用 ncclCudaCallocAsync？
// ✅ 这是 GPU 端的数据结构，GPU Kernel 需要访问
// ✅ 需要初始化为 0（结构体字段需要清零）
// ✅ 异步版本不阻塞 CPU 线程，性能更好
```

### 示例 4：IB QP 信息 - 为什么用 `ncclIbMalloc`？

```c
// src/transport/net_ib.cc:759
NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(qpInfo)));

// 为什么用 ncclIbMalloc？
// ✅ 这个缓冲区需要调用 ibv_reg_mr 注册到 IB 网卡
// ✅ ibv_reg_mr 要求内存必须页对齐
// ✅ posix_memalign 保证页对齐分配
```

---

## 🎯 关键区别总结

### 1. `ncclCalloc` vs `ncclCudaHostCalloc`

| 特性 | `ncclCalloc` | `ncclCudaHostCalloc` |
|------|-------------|---------------------|
| **底层实现** | `malloc` | `cudaHostAlloc` (Mapped) |
| **GPU 可访问** | ❌ 否 | ✅ **是** |
| **固定内存** | ❌ 否 | ✅ 是 |
| **性能** | 快（分配） | 较慢（分配），但 CPU↔GPU 传输快 |
| **使用场景** | CPU 端数据结构 | CPU↔GPU 共享数据 |

**选择原则**：
- GPU 需要访问 → `ncclCudaHostCalloc`
- 只在 CPU 端使用 → `ncclCalloc`

---

### 2. `ncclCudaMalloc` vs `ncclCudaCalloc` vs `ncclCudaCallocAsync`

| 特性 | `ncclCudaMalloc` | `ncclCudaCalloc` | `ncclCudaCallocAsync` |
|------|----------------|-----------------|---------------------|
| **初始化** | ❌ 不初始化 | ✅ 同步清零 | ✅ 异步清零 |
| **CPU 阻塞** | ❌ 不阻塞 | ✅ **阻塞** | ❌ 不阻塞 |
| **性能** | 最快 | 较慢（同步等待） | 快（异步） |
| **使用场景** | 临时缓冲区 | 初始化阶段 | 性能敏感场景 |

**选择原则**：
- 不需要初始化 → `ncclCudaMalloc`
- 需要初始化 + 不关心同步 → `ncclCudaCalloc`
- 需要初始化 + 性能敏感 → `ncclCudaCallocAsync`

---

### 3. `ncclCudaHostCalloc` vs `ncclIbMalloc`

| 特性 | `ncclCudaHostCalloc` | `ncclIbMalloc` |
|------|---------------------|---------------|
| **底层实现** | `cudaHostAlloc` (Mapped) | `posix_memalign` (页对齐) |
| **GPU 可访问** | ✅ **是** | ❌ 否 |
| **页对齐** | ✅ 是 | ✅ **必须页对齐** |
| **IB 注册** | 可以，但不常用 | ✅ **专门用于 IB 注册** |
| **使用场景** | CPU↔GPU 共享数据 | IB Verbs 内存注册 |

**选择原则**：
- GPU 需要访问 → `ncclCudaHostCalloc`
- 需要 IB 注册 + GPU 不需要访问 → `ncclIbMalloc`

---

## 🔗 相关文件

- **定义文件**：`src/include/alloc.h` - 所有内存分配 API 的定义
- **使用示例**：
  - `src/init.cc` - 初始化阶段的内存分配
  - `src/transport/net_ib.cc` - IB 传输层的内存分配
  - `src/proxy.cc` - Proxy 线程的内存分配
  - `src/enqueue.cc` - 排产阶段的内存分配

---

## 📝 总结

NCCL 中的内存分配 API 根据**使用场景**和**访问需求**分为 6 类：

1. **`ncclCalloc`** - CPU 端普通内存（最常用）
2. **`ncclCudaHostCalloc`** - CPU↔GPU 共享内存（GPU 需要访问时）
3. **`ncclCudaMalloc`** - GPU 端内存（不需要初始化）
4. **`ncclCudaCalloc`** - GPU 端内存（同步清零）
5. **`ncclCudaCallocAsync`** - GPU 端内存（异步清零，性能更好）
6. **`ncclIbMalloc`** - IB Verbs 注册专用（页对齐）

**核心原则**：
- **GPU 需要访问** → 用 `ncclCudaHostCalloc` 或 GPU 端 API
- **只在 CPU 端使用** → 用 `ncclCalloc`
- **需要 IB 注册** → 用 `ncclIbMalloc`
- **性能敏感** → 优先选择异步版本

---

## 版本与修订

- **v1.0** (2026-01-26)：初版，完整梳理 NCCL 中的 6 种内存分配 API
