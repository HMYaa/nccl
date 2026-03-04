/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*=======================================================================
 * 【学习导读】net_ib.cc — NCCL 的 InfiniBand/RoCE 传输层实现
 *
 * 这是 NCCL 数据面最核心的文件之一。它实现了 ncclNet_t 接口（见文件末尾的
 * ncclNetIb 函数表），使 NCCL 能通过 IB Verbs API 进行 RDMA 数据传输。
 *
 * ====== 整体架构 ======
 *
 *  上层（proxy thread）
 *       │
 *       ▼
 *  ncclNetIb 函数表（本文件导出）
 *  ┌──────────────────────────────────────────────────┐
 *  │  init        → ncclIbInit()          硬件发现     │
 *  │  listen      → ncclIbListen()        监听连接     │
 *  │  connect     → ncclIbConnect()       发起连接     │
 *  │  accept      → ncclIbAccept()        接受连接     │
 *  │  regMr       → ncclIbRegMr()         注册 MR      │
 *  │  deregMr     → ncclIbDeregMr()       注销 MR      │
 *  │  isend       → ncclIbIsend()         异步发送     │
 *  │  irecv       → ncclIbIrecv()         异步接收     │
 *  │  iflush      → ncclIbIflush()        GPU 刷新     │
 *  │  test        → ncclIbTest()          完成检测     │
 *  │  closeSend/Recv/Listen → 资源释放                  │
 *  └──────────────────────────────────────────────────┘
 *       │
 *       ▼
 *  IB Verbs API（ibv_post_send / ibv_poll_cq / ...）
 *       │
 *       ▼
 *  RDMA 网卡硬件（HCA）
 *
 * ====== 关键设计：FIFO 通知机制 ======
 *
 *  NCCL 不使用传统的 SEND/RECV 双边语义，而是用 RDMA WRITE 单边语义。
 *  但 RDMA WRITE 需要知道对端的 remote_addr 和 rkey。
 *
 *  解决方案：接收端通过 RDMA WRITE 把自己的 {addr, rkey, size} 写到
 *  发送端的 FIFO 中（ncclIbPostFifo），发送端轮询 FIFO 获取这些信息
 *  后，才执行真正的数据 RDMA WRITE（ncclIbMultiSend）。
 *
 *  数据传输流程：
 *    1. Recv 端调 ncclIbIrecv → post recv WR + RDMA WRITE 通知到 sender FIFO
 *    2. Send 端调 ncclIbIsend → 轮询 FIFO 发现对端就绪 → ncclIbMultiSend
 *    3. ncclIbMultiSend → RDMA WRITE 数据 + RDMA WRITE_WITH_IMM 通知完成
 *    4. Recv 端 poll CQ 收到 IMM 完成 → ncclIbTest 返回 done
 *
 * ====== Week 1 学习路线 ======
 *
 *  Q1-Q4:  从 ncclIbIsend/ncclIbMultiSend 开始（数据发送路径）
 *  Q5-Q7:  看 ncclIbRegMrDmaBuf/ncclIbDeregMr（MR 管理）
 *  Q8-Q10: 看 ncclIbCreateQp/ncclIbConnect/ncclIbAccept（连接建立）
 *=======================================================================*/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"

#define MAXNAMESIZE 64
static char ncclIbIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress ncclIbIfAddr;

struct ncclIbMr {
  uintptr_t addr;
  int pages;
  int refs;
  ibv_mr *mr;
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

static int ncclNIbDevs = -1;
struct alignas(64) ncclIbDev {
  pthread_mutex_t lock;
  int device;
  uint64_t guid;
  uint8_t port;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char* pciPath;
  int realPort;
  int maxQp;
  struct ncclIbMrCache mrCache;
};

#define MAX_IB_PORT 15
struct userIbDev {
  char devName[MAXNAMESIZE];
  uint16_t port_en;
};

#define MAX_IB_DEVS 16
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];
pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;
static int ncclIbRelaxedOrderingEnabled = 0;

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", 0);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 18);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbSl, "IB_SL", 0);
NCCL_PARAM(IbTc, "IB_TC", 0);
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);

pthread_t ncclIbAsyncThread;
static void* ncclIbAsyncThreadMain(void* args) {
  struct ibv_context* context = (struct ibv_context*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(context, &event)) { break; }
    char *str;
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    if (event.event_type != IBV_EVENT_COMM_EST)
      WARN("NET/IB : Got async event : %s", str);
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, int* realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    p[strlen(p)-3] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d=0; d<ncclNIbDevs; d++) {
      if (strcmp(p, ncclIbDevs[d].pciPath) == 0) (*realPort)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

/**
 * 【网卡速度计算辅助函数】
 * 
 * IB Verbs API 返回的 width 和 speed 是位掩码，需要转换为实际数值
 */

// IB Verbs 定义的链路宽度（单位：lanes）
// 对应 IBV_WIDTH_1X, IBV_WIDTH_4X, IBV_WIDTH_8X, IBV_WIDTH_12X, IBV_WIDTH_2X
static int ibvWidths[] = { 1, 4, 8, 12, 2 };
// IB Verbs 定义的速度（单位：Mbps per lane）
// 对应 SDR(2.5G), DDR(5G), QDR(10G), FDR10(10G), FDR(14G), EDR(25G), HDR(50G)
static int ibvSpeeds[] = { 2500, 5000, 10000, 10000, 14000, 25000, 50000 };

/**
 * 查找位掩码中第一个设置的位
 * @param val: 位掩码值
 * @param max: 最大位数
 * @return: 第一个设置的位的索引
 */
static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}

/**
 * 将 IB Verbs 的宽度位掩码转换为实际宽度值（lanes）
 * 例如：IBV_WIDTH_4X (位掩码) -> 4 (lanes)
 */
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}

/**
 * 将 IB Verbs 的速度位掩码转换为实际速度值（Mbps per lane）
 * 例如：IBV_PORT_SPEED_25G (位掩码) -> 25000 (Mbps per lane)
 */
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

/**
 * 【IB 网卡初始化函数】发现和初始化所有 InfiniBand/RoCE 网卡
 * 
 * 这是硬件发现的核心函数，负责：
 * 1. 加载 IB Verbs 库符号（wrap_ibv_symbols）
 * 2. 查找 IP 接口（用于 OOB 通信）
 * 3. 枚举所有 IB 网卡设备（ibv_get_device_list）
 * 4. 打开每个网卡设备（ibv_open_device）
 * 5. 查询设备属性（ibv_query_device）
 * 6. 遍历所有端口，检查端口状态和链路类型
 * 7. 计算网卡速度（speed = active_speed * active_width）
 * 8. 保存网卡信息到全局数组 ncclIbDevs
 * 
 * 调用时机：
 * - 在 ncclInit() -> ncclNetPluginInit() -> netGetState() -> ncclNetIb.init() 中被调用
 * - 只初始化一次（通过 ncclNIbDevs == -1 检查）
 * 
 * 关键点：
 * - 使用互斥锁保护初始化过程（ncclIbLock）
 * - 支持用户通过 NCCL_IB_HCA 环境变量指定网卡
 * - 只接受 ACTIVE 状态的端口
 * - 支持 InfiniBand 和 RoCE（Ethernet）两种链路类型
 * - 速度计算：如果 400G 网卡被错误识别，会在拓扑打分时降权
 */
ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction) {
  // 如果禁用了 IB，直接返回错误
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  
  // 【步骤 1】加载 IB Verbs 库符号（动态链接）
  // 如果加载失败，说明系统没有安装 IB 驱动或库
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }

  // 【步骤 2】检查是否已经初始化过（只初始化一次）
  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    // 初始化 fork 支持（多进程环境下需要）
    wrap_ibv_fork_init();
    
    // 双重检查（double-check locking pattern）
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      
      // 【步骤 3】查找 IP 接口（用于 OOB - Out-of-Band 通信，如 bootstrap）
      // OOB 通信用于进程间协调，不用于数据传输
      if (ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        return ncclInternalError;
      }

      // 【步骤 4】检测 IB 网卡
      int nIbDevs;
      struct ibv_device** devices;

      // 【步骤 5】检查用户是否通过环境变量指定了要使用的 IB 设备
      // NCCL_IB_HCA 格式：设备名:端口，例如 "mlx5_0:1,mlx5_1:1"
      // 支持前缀：
      //   ^ 表示排除（NOT）
      //   = 表示精确匹配
      char* userIbEnv = getenv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';  // 排除模式
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';  // 精确匹配模式
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      // 【步骤 6】调用 IB Verbs API 获取所有 IB 设备列表
      // 这是第一次真正调用驱动 API
      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) return ncclInternalError;

      // 【步骤 7】遍历所有 IB 设备
      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context;
        
        // 【步骤 7.1】打开设备，获取 context（设备句柄）
        // context 用于后续的查询和操作
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        
        // 【步骤 7.2】查询设备属性（设备能力、最大 QP 数等）
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
          continue;
        }
        
        // 【步骤 7.3】遍历设备的所有物理端口
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          struct ibv_port_attr portAttr;
          
          // 【步骤 7.3.1】查询端口属性（状态、速度、链路类型等）
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
            WARN("NET/IB : Unable to query port %d", port);
            continue;
          }
          
          // 【步骤 7.3.2】只接受 ACTIVE 状态的端口（端口必须已连接并激活）
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          
          // 【步骤 7.3.3】只接受 InfiniBand 或 RoCE（Ethernet）链路类型
          // 排除其他链路类型（如 IPoIB）
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // 【步骤 7.3.4】检查端口是否匹配用户指定的 HCA/端口列表
          // matchIfList 检查设备名和端口是否在用户列表中
          // searchNot: 如果为 true，则排除匹配的；如果为 false，则只接受匹配的
          if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs, searchExact) ^ searchNot)) {
            continue;
          }
          
          // 【步骤 7.3.5】记录发现的端口信息
          TRACE(NCCL_INIT|NCCL_NET,"NET/IB: [%d] %s:%d/%s ", d, devices[d]->name, port,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
          
          // 初始化设备结构体
          pthread_mutex_init(&ncclIbDevs[ncclNIbDevs].lock, NULL);
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;  // 系统镜像 GUID（唯一标识）
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;  // 链路类型（IB 或 RoCE）
          
          // 【关键】计算网卡速度：speed = active_speed * active_width
          // active_speed: 实际速度（Mbps per lane），例如 25000 表示 25Gbps per lane
          // active_width: 实际宽度（lanes），例如 4 表示 4x
          // 最终速度：25000 * 4 = 100 Gbps
          // 注意：如果 400G 网卡被错误识别（比如只识别成 100G），NCCL 会在拓扑打分时降权
          ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
          
          ncclIbDevs[ncclNIbDevs].context = context;
          ncclIbDevs[ncclNIbDevs].pdRefs = 0;  // Protection Domain 引用计数
          ncclIbDevs[ncclNIbDevs].pd = NULL;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          
          // 获取 PCI 路径（用于拓扑匹配）
          NCCLCHECK(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort));
          
          ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;  // 最大 Queue Pair 数量
          ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;  // Memory Region 缓存
          ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;

          // 为每个设备创建异步事件处理线程
          pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
          ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
          pthread_detach(ncclIbAsyncThread); // 分离线程，不需要 join
          ncclNIbDevs++;
          nPorts++;
        }
        
        // 如果设备没有任何可用端口，关闭设备
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
      }
      
      // 释放设备列表
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { return ncclInternalError; };
    }
    
    // 【步骤 8】打印发现的网卡信息
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    } else {
      char line[1024];
      line[0] = '\0';
      // 检查是否支持 RELAXED_ORDERING（PCIe 放松排序，可提升性能）
      ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
      for (int d=0; d<ncclNIbDevs; d++) {
        snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
            ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
      }
      line[1023] = '\0';
      char addrline[SOCKET_NAME_MAXLEN+1];
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
           ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
ncclResult_t ncclIbGdrSupport(int ibDev) {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    // Check for the nv_peer_mem module being loaded
    moduleLoaded = ((access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) &&
                    // Also support the new nvidia-peermem module
                    (access("/sys/kernel/mm/memory_peers/nvidia-peermem/version", F_OK) == -1)) ? 0 : 1;
  }
  if (moduleLoaded == 0) return ncclSystemError;
  return ncclSuccess;
}

// Detect whether DMA-BUF support is present in the kernel
// Returns :
// ncclSuccess : DMA-BUF support is available
// ncclSystemError : DMA-BUF is not supported by the kernel
ncclResult_t ncclIbDmaBufSupport(int dev) {
  static int dmaBufSupported = -1;
  if (dmaBufSupported == -1) {
    ncclResult_t res;
    struct ibv_pd* pd;
    struct ibv_context* ctx;
    ctx = ncclIbDevs[dev].context;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
    // Test kernel DMA-BUF support with a dummy call (fd=-1)
    (void) wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL/*offset*/, 0ULL/*len*/, 0ULL/*iova*/, -1/*fd*/, 0/*flags*/);
    // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
    dmaBufSupported = (errno != EOPNOTSUPP && errno != EPROTONOSUPPORT) ? 1 : 0;
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  }
  if (dmaBufSupported == 0) return ncclSystemError;
  return ncclSuccess;
failure:
  dmaBufSupported = 0;
  return ncclSystemError;
}

static ncclResult_t GetSocketAddr(union ncclSocketAddress* addr) {
  memcpy(addr, &ncclIbIfAddr, sizeof(*addr));
  return ncclSuccess;
}

#define NCCL_NET_IB_MAX_RECVS 8

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = ncclIbDevs[dev].devName;
  props->pciPath = ncclIbDevs[dev].pciPath;
  props->guid = ncclIbDevs[dev].guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->speed = ncclIbDevs[dev].speed;
  props->latency = 0; // Not set
  props->port = ncclIbDevs[dev].port + ncclIbDevs[dev].realPort;
  props->maxComms = ncclIbDevs[dev].maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  return ncclSuccess;
}

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

#define NCCL_IB_MAX_QPS 128

/*
 * 【Q10 相关】QP 连接信息：通过 Socket（OOB 通道）在两端之间交换
 *
 * 连接建立时，双方各自填好这个结构体，通过 TCP socket 发给对方。
 * 对方拿到后用 qpn[] 调 ibv_modify_qp(RTR) 建立 QP 连接。
 * fifoRkey/fifoAddr 是发送端的 FIFO 地址，接收端拿到后可以 RDMA WRITE 通知。
 */
struct ncclIbQpInfo {
  uint32_t lid;                     // IB 链路层 LID（仅 IB 链路使用）
  uint8_t ib_port;                  // IB 端口号
  uint8_t link_layer;               // 链路类型：IB 或 RoCE(Ethernet)
  uint32_t qpn[NCCL_IB_MAX_QPS];   // 本端 QP number 数组（对端连接时需要）

  // RoCE 专用字段（IB 链路不需要）
  uint64_t spn;                     // GID subnet prefix
  uint64_t iid;                     // GID interface id
  enum ibv_mtu mtu;                 // 协商后的 MTU

  // 发送端 FIFO 的 RDMA 信息（接收端用来写通知）
  uint32_t fifoRkey;                // FIFO 内存的 remote key
  uint64_t fifoAddr;                // FIFO 内存的远端地址
};

enum ncclIbCommState {
  ncclIbCommStateStart = 0,
  ncclIbCommStateConnect = 1,
  ncclIbCommStateAccept = 3,
  ncclIbCommStateSend = 4,
  ncclIbCommStateRecv = 5,
  ncclIbCommStateConnected = 6,
};

struct ncclIbCommStage {
  enum ncclIbCommState state;
  int offset;
  void* buffer;
  void* comm;
};

struct ncclIbHandle {
  union ncclSocketAddress connectAddr; // Filled by the target
  struct ncclIbCommStage stage; // Used by the other side when connecting
};

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3

struct ncclIbRequest {
  struct ncclIbVerbs* verbs;
  int type;
  int events;
  union ncclSocketAddress *addr;
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkey;
      int offset;
    } send;
    struct {
      int sizes[NCCL_NET_IB_MAX_RECVS];
    } recv;
  };
};

struct ncclIbVerbs {
  int dev;
  struct ibv_pd* pd; // duplicate of ncclIbDevs[dev].pd
  struct ibv_cq* cq;
  uint64_t pad[1];
  struct ncclIbRequest reqs[MAX_REQUESTS];
};

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage stage;
};

/*
 * 【Q4 相关】SendFifo 元素：接收端通过 RDMA WRITE 写到发送端的通知条目
 *
 * 这是 NCCL FIFO 通知机制的核心数据结构。每当接收端调用 ncclIbIrecv 时，
 * 它会把自己的缓冲区信息（addr/rkey/size）填入此结构，然后 RDMA WRITE 到
 * 发送端的 FIFO 中。发送端在 ncclIbIsend 中轮询这个 FIFO，看到 idx 匹配
 * 后就知道对端已就绪，才执行真正的数据 RDMA WRITE。
 *
 * 注意：此结构必须是 32 字节对齐（见下方 static_assert），
 * 因为开启 IB Relaxed Ordering 时，非对齐写入可能导致条目被拆分写入。
 */
struct ncclIbSendFifo {
  uint64_t addr;        // 接收端缓冲区的远端地址（发送端 RDMA WRITE 的目标）
  int      size;        // 接收端期望的数据大小
  uint32_t rkey;        // 接收端缓冲区的 remote key
  uint32_t nreqs;       // 本轮 multi-recv 的请求数量
  uint32_t tag;         // 消息标签（用于匹配 send/recv 对）
  uint64_t idx;         // 单调递增索引，发送端用来判断此条目是否有效
};

/*
 * 【发送端】Send 侧的连接上下文
 *
 * fifo[][]   — 本端 FIFO 缓冲区，接收端通过 RDMA WRITE 把通知写到这里。
 *              发送端在 ncclIbIsend 中轮询 fifo[slot][0].idx 判断接收端是否就绪。
 * fifoMr     — fifo 注册的 MR（rkey 在连接建立时发给对端）
 * qps[]      — 可能有多个 QP（NCCL_IB_QPS_PER_CONNECTION 控制），
 *              大数据会按 QP 数量做 round-robin 切分
 * wrs/sges   — 预分配的 WR/SGE 数组，避免热路径上的动态分配
 * sock       — TCP socket，用于 OOB 连接建立（交换 QP 信息）
 */
struct ncclIbSendComm {
  struct ncclIbVerbs verbs;
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoHead;
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS+1];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ncclSocket sock;

  int ready;
  struct ibv_qp* qps[NCCL_IB_MAX_QPS];
  int nqps;
  struct ibv_mr* fifoMr;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((offsetof(struct ncclIbSendComm, fifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");

struct ncclIbGpuFlush {
  int enabled;
  int hostMem;
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ibv_qp* qp;
};

struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t rkey;
  uint32_t flags;
  struct ibv_mr* mr;
  struct ibv_sge sge;
};

/*
 * 【接收端】Recv 侧的连接上下文
 *
 * remFifo    — 远端（发送端）FIFO 的本地镜像 + RDMA 信息。
 *              接收端每次 Irecv 时，把 {addr,rkey,size} 填入 remFifo.elems，
 *              然后 RDMA WRITE 到发送端的 fifo 中。
 * gpuFlush   — GDR 场景下用 RDMA READ 刷新 GPU 缓存的机制。
 *              因为 RDMA WRITE 到 GPU 显存后，GPU 不一定立即看到新数据，
 *              需要一次 RDMA READ 来保证数据可见性（PCIe ordering）。
 */
struct ncclIbRecvComm {
  struct ncclIbVerbs verbs;
  struct ncclIbRemFifo remFifo;
  struct ncclSocket sock;
  int ready;
  struct ibv_qp* qps[NCCL_IB_MAX_QPS];
  int nqps;
  struct ncclIbGpuFlush gpuFlush;
};
static_assert((offsetof(struct ncclIbRecvComm, remFifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

ncclResult_t ncclIbInitVerbs(int dev, struct ibv_context* ctx, struct ncclIbVerbs* verbs) {
  verbs->dev = dev;

  pthread_mutex_lock(&ncclIbDevs[dev].lock);
  if (0 == ncclIbDevs[dev].pdRefs++) {
    ncclResult_t res;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&ncclIbDevs[dev].pd, ctx), res, failure);
    if (0) {
    failure:
      pthread_mutex_unlock(&ncclIbDevs[dev].lock);
      return res;
    }
  }
  verbs->pd = ncclIbDevs[dev].pd;
  pthread_mutex_unlock(&ncclIbDevs[dev].lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for the Recv).
  NCCLCHECK(wrap_ibv_create_cq(&verbs->cq, ctx, 2*MAX_REQUESTS*ncclParamIbQpsPerConn(), NULL, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclIbDestroyVerbs(struct ncclIbVerbs* verbs) {
  ncclResult_t res;
  NCCLCHECK(wrap_ibv_destroy_cq(verbs->cq));

  pthread_mutex_lock(&ncclIbDevs[verbs->dev].lock);
  if (0 == --ncclIbDevs[verbs->dev].pdRefs) {
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(ncclIbDevs[verbs->dev].pd), res, returning);
  }
  res = ncclSuccess;
returning:
  pthread_mutex_unlock(&ncclIbDevs[verbs->dev].lock);
  return res;
}

/*
 * 【Q8 答案】创建 QP 并转到 INIT 状态
 *
 * QP 类型：IBV_QPT_RC（Reliable Connection）——和你写 RDMA 应用一样。
 * RC 提供可靠有序传输，支持 RDMA WRITE/READ，适合大块数据传输。
 * NCCL 没有用 UD（Unreliable Datagram），因为 UD 不支持 RDMA 操作。
 *
 * 【Q9 答案 — 第一步】QP 状态机：RESET → INIT
 * ibv_create_qp 后 QP 处于 RESET 状态，
 * 这里立即 modify 到 INIT，指定端口和访问权限。
 * INIT 状态下 QP 可以 post recv WR，但还不能发送。
 *
 * 注意 max_send_wr = 2*MAX_REQUESTS：
 * 因为每次发送可能包含两个 WR（数据 RDMA_WRITE + 通知 RDMA_WRITE_WITH_IMM）
 */
ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbVerbs* verbs, int access_flags, struct ibv_qp** qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = verbs->cq;
  qpInitAttr.recv_cq = verbs->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;       // 【Q8】RC 类型
  qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
  NCCLCHECK(wrap_ibv_create_qp(qp, verbs->pd, &qpInitAttr));

  // 【Q9 第一步】RESET → INIT
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;  // 通常包含 REMOTE_WRITE（允许对端 RDMA WRITE）
  NCCLCHECK(wrap_ibv_modify_qp(*qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

/*
 * 【Q9 第二步】INIT → RTR（Ready To Receive）
 *
 * 需要对端的 QPN（dest_qp_num）和寻址信息（LID 或 GID）。
 * 这些信息通过 OOB socket 交换的 ncclIbQpInfo 获得。
 *
 * IB 链路：用 LID 寻址（ah_attr.dlid）
 * RoCE 链路：用 GID 寻址（ah_attr.grh.dgid），需要设 is_global=1
 *
 * RTR 状态下 QP 可以接收数据，但还不能发送。
 */
ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, uint32_t qpn, struct ncclIbQpInfo* info) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = qpn;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = ncclParamIbGidIndex();
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = ncclParamIbTc();
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
  }
  qpAttr.ah_attr.sl = ncclParamIbSl();
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return ncclSuccess;
}

/*
 * 【Q9 第三步】RTR → RTS（Ready To Send）
 *
 * 设置超时和重试参数后，QP 进入 RTS——此时 QP 全功能可用。
 * timeout: 重传超时（指数值，18 ≈ 1秒级别）
 * retry_cnt: 重传次数（7 是默认值）
 * rnr_retry: RNR（Receiver Not Ready）重试次数（7 = 无限重试）
 *
 * 完整状态转换链：RESET → INIT → RTR → RTS
 * 其中 INIT → RTR 需要对端信息，所以必须在 OOB 交换之后。
 */
ncclResult_t ncclIbRtsQp(struct ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();
  qpAttr.retry_cnt = ncclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return ncclSuccess;
}

ncclResult_t ncclIbListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclIbListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  memset(handle, 0, sizeof(struct ncclIbHandle));
  comm->dev = dev;
  comm->sock.asyncFlag = 1; /* nonblocking socket is required by network communication. */
  NCCLCHECK(GetSocketAddr(&comm->sock.addr));
  NCCLCHECK(ncclSocketListen(&comm->sock));
  memcpy(&handle->connectAddr, &comm->sock.addr, sizeof(union ncclSocketAddress));
  *listenComm = comm;
  return ncclSuccess;
}

/*
 * 【Q10 答案】发送端连接建立（Connect 侧）
 *
 * 这是非阻塞的状态机实现，可能被多次调用直到完成。
 * 状态转换：Start → Connect(TCP) → Send(QP info) → Connected
 *
 * 连接流程：
 *  1. 通过 TCP socket 连接到接收端的 listen 地址
 *  2. 创建 QP（INIT 状态）+ 注册 FIFO 的 MR
 *  3. 把本端的 {QPN, LID/GID, MTU, fifoRkey, fifoAddr} 发给对端
 *  4. 等待对端回复其 QP 信息（在 ncclSendCheck 中完成 RTR→RTS）
 *
 * 关键点：
 * - 连接建立是通过 TCP socket 做 OOB 交换（类似你用 CM 做连接管理）
 * - NCCL 不使用 rdma_cm，而是自己实现了更轻量的连接管理
 * - fifo MR 在这里注册，rkey 发给对端供其 RDMA WRITE 通知
 */
ncclResult_t ncclIbConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  enum ncclSocketState conState;
  struct ncclIbCommStage* stage = &handle->stage;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;
  *sendComm = NULL;

  if (stage->state == ncclIbCommStateConnect) goto ib_connect_check;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));
  NCCLCHECK(ncclSocketInit(&comm->sock, &handle->connectAddr, NULL, 1));
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;
  NCCLCHECK(ncclSocketConnect(&comm->sock));

ib_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECK(ncclGetSocketState(&comm->sock, &conState));
  if (conState == ncclSocketConnecting) {
    /* expect user to call again */
    return ncclSuccess;
  } else if (conState == ncclSocketError) {
    return ncclRemoteError;
  }

  // IB Setup
  struct ibv_context* ctx;
  ctx = ncclIbDevs[dev].context;
  NCCLCHECK(ncclIbInitVerbs(dev, ctx, &comm->verbs));
  uint8_t ib_port;
  ib_port = ncclIbDevs[dev].port;
  comm->nqps = ncclParamIbQpsPerConn();
  for (int q=0; q<comm->nqps; q++) {
    NCCLCHECK(ncclIbCreateQp(ib_port, &comm->verbs, IBV_ACCESS_REMOTE_WRITE, comm->qps+q));
  }

  // Send my QP Info to receiver through the socket. Hope this won't block.
  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  struct ncclIbQpInfo qpInfo;
  qpInfo.ib_port = ib_port;
  for (int q=0; q<comm->nqps; q++) qpInfo.qpn[q] = comm->qps[q]->qp_num;
  qpInfo.mtu = portAttr.active_mtu;

  // Prepare my fifo
  NCCLCHECK(wrap_ibv_reg_mr(&comm->fifoMr, comm->verbs.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
  qpInfo.fifoRkey = comm->fifoMr->rkey;
  qpInfo.fifoAddr = (uint64_t)comm->fifo;

  // RoCE support
  qpInfo.lid = portAttr.lid;
  qpInfo.link_layer = portAttr.link_layer;
  if (qpInfo.link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
    for (int q=0; q<comm->nqps; q++)
      INFO(NCCL_NET,"NET/IB: Dev %d Port %d qpn %d mtu %d LID %d", dev, ib_port, qpInfo.qpn[q], qpInfo.mtu, qpInfo.lid);
  } else { // RoCE
    union ibv_gid gid;
    NCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, ncclParamIbGidIndex(), &gid));
    qpInfo.spn = gid.global.subnet_prefix;
    qpInfo.iid = gid.global.interface_id;
    for (int q=0; q<comm->nqps; q++)
      INFO(NCCL_NET,"NET/IB: Dev %d Port %d qpn %d mtu %d GID %ld (%lX/%lX)", dev, ib_port, qpInfo.qpn[q], qpInfo.mtu, ncclParamIbGidIndex(), qpInfo.spn, qpInfo.iid);
  }

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(qpInfo)));
  memcpy(stage->buffer, &qpInfo, sizeof(qpInfo));

ib_send:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->sock, stage->buffer, sizeof(qpInfo), &stage->offset));
  if (stage->offset != sizeof(qpInfo))
    return ncclSuccess;

  free(stage->buffer);
  stage->state = ncclIbCommStateConnected;
  *sendComm = comm;
  return ncclSuccess;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

/*
 * 【Q10 答案续】接收端连接建立（Accept 侧）
 *
 * 非阻塞状态机：Start → Accept(TCP) → Recv(对端QP info) → Send(本端QP info)
 *
 * 接收到对端 QP 信息后：
 *  1. 创建本端 QP
 *  2. 用对端的 QPN 做 RTR → RTS 状态转换（此时连接建立完成）
 *  3. 保存对端的 fifoRkey/fifoAddr（后续 ncclIbPostFifo 用于 RDMA WRITE 通知）
 *  4. 如果支持 GDR，还会创建一个额外的 gpuFlush QP（用于 RDMA READ 刷新 GPU 缓存）
 *  5. 把本端的 QP 信息发回给对端
 */
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm) {
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = &lComm->stage;
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)stage->comm;
  *recvComm = NULL;

  if (stage->state == ncclIbCommStateAccept) goto ib_accept;
  if (stage->state == ncclIbCommStateRecv) goto ib_recv;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Listencomm in unknown state %d\n", stage->state);
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)));
  stage->comm = rComm;
  stage->state = ncclIbCommStateAccept;
  NCCLCHECK(ncclSocketInit(&rComm->sock, NULL, lComm->sock.abortFlag, 1));

ib_accept:
  NCCLCHECK(ncclSocketAccept(&rComm->sock, &lComm->sock));
  if (rComm->sock.fd == -1)
    return ncclSuccess;

  struct ncclIbQpInfo remQpInfo;
  stage->state = ncclIbCommStateRecv;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remQpInfo)));
ib_recv:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->sock, stage->buffer, sizeof(remQpInfo), &stage->offset));
  if (stage->offset != sizeof(remQpInfo))
    return ncclSuccess;

  /* copy back the received info */
  memcpy(&remQpInfo, stage->buffer, sizeof(struct ncclIbQpInfo));

  // IB setup
  struct ibv_context* ctx;
  uint8_t ib_port;
  ctx = ncclIbDevs[lComm->dev].context;
  ib_port = ncclIbDevs[lComm->dev].port;
  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  union ibv_gid gid;
  NCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, ncclParamIbGidIndex(), &gid));

  // QP Creation
  NCCLCHECK(ncclIbInitVerbs(lComm->dev, ctx, &rComm->verbs));
  rComm->nqps = ncclParamIbQpsPerConn();
  for (int q=0; q<rComm->nqps; q++) {
    NCCLCHECK(ncclIbCreateQp(ib_port, &rComm->verbs, IBV_ACCESS_REMOTE_WRITE, rComm->qps+q));
  }

  // Adjust the MTU
  remQpInfo.mtu = (enum ibv_mtu)std::min(remQpInfo.mtu, portAttr.active_mtu);

  // Setup QP
  for (int q=0; q<rComm->nqps; q++) {
    struct ibv_qp* qp = rComm->qps[q];
    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn[q], &remQpInfo));
    NCCLCHECK(ncclIbRtsQp(qp));
  }

  // Retain remote fifo info and prepare my RDMA ops
  rComm->remFifo.rkey = remQpInfo.fifoRkey;
  rComm->remFifo.addr = remQpInfo.fifoAddr;
  NCCLCHECK(wrap_ibv_reg_mr(&rComm->remFifo.mr, rComm->verbs.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
  rComm->remFifo.sge.lkey = rComm->remFifo.mr->lkey;
  if (ncclParamIbUseInline()) rComm->remFifo.flags = IBV_SEND_INLINE;

  // Allocate Flush dummy buffer for GPU Direct RDMA
  rComm->gpuFlush.enabled = (ncclIbGdrSupport(lComm->dev) == 0) && (ncclParamIbGdrFlushDisable() == 0) ? 1 : 0;
  if (rComm->gpuFlush.enabled) {
    NCCLCHECK(wrap_ibv_reg_mr(&rComm->gpuFlush.hostMr, rComm->verbs.pd, &rComm->gpuFlush.hostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE));
    rComm->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlush.hostMem;
    rComm->gpuFlush.sge.length = 1;
    rComm->gpuFlush.sge.lkey = rComm->gpuFlush.hostMr->lkey;
    NCCLCHECK(ncclIbCreateQp(ib_port, &rComm->verbs, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rComm->gpuFlush.qp));
    struct ncclIbQpInfo localQpInfo;
    localQpInfo.lid=portAttr.lid;
    localQpInfo.link_layer=portAttr.link_layer;
    localQpInfo.ib_port=ib_port;
    localQpInfo.spn=gid.global.subnet_prefix;
    localQpInfo.iid=gid.global.interface_id;
    localQpInfo.mtu=portAttr.active_mtu;
    NCCLCHECK(ncclIbRtrQp(rComm->gpuFlush.qp, rComm->gpuFlush.qp->qp_num, &localQpInfo));
    NCCLCHECK(ncclIbRtsQp(rComm->gpuFlush.qp));
  }

  // Fill Handle
  struct ncclIbQpInfo qpInfo;
  qpInfo.lid=portAttr.lid;
  qpInfo.link_layer=portAttr.link_layer;
  qpInfo.ib_port=ib_port;
  for (int q=0; q<rComm->nqps; q++) qpInfo.qpn[q]=rComm->qps[q]->qp_num;
  qpInfo.spn=gid.global.subnet_prefix;
  qpInfo.iid=gid.global.interface_id;
  qpInfo.mtu=remQpInfo.mtu;

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer) free(stage->buffer);
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbQpInfo)));
  memcpy(stage->buffer, &qpInfo, sizeof(struct ncclIbQpInfo));
ib_send:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->sock, stage->buffer, sizeof(struct ncclIbQpInfo), &stage->offset));
  if (stage->offset < sizeof(struct ncclIbQpInfo)) return ncclSuccess;

  free(stage->buffer);
  *recvComm = rComm;

  /* reset lComm stage */
  stage->state = ncclIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  stage->buffer = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbGetRequest(struct ncclIbVerbs* verbs, struct ncclIbRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = verbs->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->verbs = verbs;
      r->events = 1;
      r->addr = NULL;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

ncclResult_t ncclSendCheck(struct ncclIbSendComm* comm) {
  struct ncclIbQpInfo remQpInfo;

  // Do not block on this receive, return if not ready.
  int bytes = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &remQpInfo, sizeof(remQpInfo), &bytes));
  if (bytes == 0) return ncclSuccess; // Try again later
  NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, &comm->sock, &remQpInfo, sizeof(remQpInfo), &bytes));

  for (int q=0; q<comm->nqps; q++) {
    struct ibv_qp* qp = comm->qps[q];
    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn[q], &remQpInfo));
    NCCLCHECK(ncclIbRtsQp(qp));
  }
  comm->ready = 1;
  // Block until this is done. It *should* not block indefinitely.
  NCCLCHECK(ncclSocketSend(&comm->sock, &comm->ready, sizeof(int)));

  return ncclSuccess;
}

ncclResult_t ncclRecvCheck(struct ncclIbRecvComm* comm) {
  // Do not block on this receive, return if not ready.
  int bytes = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &comm->ready, sizeof(int), &bytes));
  if (bytes == 0) return ncclSuccess; // Try again later
  NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, &comm->sock, &comm->ready, sizeof(int), &bytes));
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* size);

/*
 * 【Q5/Q6 答案】MR 注册（带 DMA-BUF 支持和缓存）
 *
 * Q5: ibv_reg_mr 在哪被调用？addr 是 GPU 还是 CPU 地址？
 *   → 都有可能。data 参数来自上层，可以是：
 *     - CPU 地址：普通 ibv_reg_mr
 *     - GPU 地址 + GDR：通过 nvidia-peermem 内核模块，ibv_reg_mr 直接注册 GPU 显存
 *     - GPU 地址 + DMA-BUF：通过 ibv_reg_dmabuf_mr（新接口，需要内核支持）
 *
 * Q6: MR Cache 用什么数据结构？Key 是什么？
 *   → 线性数组（ncclIbMrCache.slots），以 (addr, pages) 二元组做 key。
 *     查找时线性遍历，命中则增加引用计数。
 *     缓存按需增长（初始 capacity=0，首次分配 32 个槽位，之后翻倍）。
 *
 * 为什么需要 MR 缓存？
 *   ibv_reg_mr 非常昂贵（需要 pin 物理页面，可能触发内核页表操作），
 *   同一块 buffer 被反复发送时，缓存避免了重复注册。
 *
 * RELAXED_ORDERING：
 *   PCIe 放松排序，允许网卡乱序读取内存，可提升 DMA 吞吐。
 *   需要 IBVERBS_1.8 API（wrap_ibv_reg_mr_iova2）。
 */
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  static_assert(offsetof(struct ncclIbSendComm, verbs) == offsetof(struct ncclIbRecvComm, verbs), "Send and recv comms must have verbs at the same offset");
  assert(size > 0);

  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);

  struct ncclIbVerbs* verbs = (struct ncclIbVerbs*)comm;
  struct ncclIbMrCache* cache = &ncclIbDevs[verbs->dev].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  ncclResult_t res;
  pthread_mutex_lock(&ncclIbDevs[verbs->dev].lock);
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population) { // didn't find in cache
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECKGOTO(ncclRealloc(&cache->slots, cache->population, cache->capacity), res, returning);
      }
      // Deregister / register
      struct ibv_mr* mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
      if (ncclIbRelaxedOrderingEnabled) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        NCCLCHECKGOTO(wrap_ibv_reg_dmabuf_mr(&mr, verbs->pd, offset, pages*pageSize, addr, fd, flags), res, returning);
      } else {
        if (ncclIbRelaxedOrderingEnabled) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECKGOTO(wrap_ibv_reg_mr_iova2(&mr, verbs->pd, (void*)addr, pages*pageSize, addr, flags), res, returning);
        }
        else {
          NCCLCHECKGOTO(wrap_ibv_reg_mr(&mr, verbs->pd, (void*)addr, pages*pageSize, flags), res, returning);
        }
      }
      TRACE(NCCL_INIT,"regAddr %llx size %lld rkey %x fd %d", (unsigned long long)addr, (long long)pages*pageSize, mr->rkey, fd);
      cache->population += 1;
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      *mhandle = (void*)mr;
      res = ncclSuccess;
      goto returning;
    }
    else if (cache->slots[slot].addr == addr && cache->slots[slot].pages == pages) {
      cache->slots[slot].refs += 1;
      *mhandle = (void*)cache->slots[slot].mr;
      res = ncclSuccess;
      goto returning;
    }
  }
returning:
  pthread_mutex_unlock(&ncclIbDevs[verbs->dev].lock);
  return res;
}

ncclResult_t ncclIbRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return ncclIbRegMrDmaBuf(comm, data, (size_t)size, type, 0ULL, -1, mhandle);
}

/*
 * 【Q7 答案】MR 注销与缓存淘汰
 *
 * Q7: MR 什么时候被释放？有淘汰机制吗？
 *   → 引用计数机制：每次 RegMr 命中缓存则 refs++，DeregMr 则 refs--。
 *     refs 降到 0 时才真正调 ibv_dereg_mr 释放。
 *   → 释放后用 memmove 把最后一个槽位搬到空位（保持数组紧凑）。
 *   → 没有 LRU 或主动淘汰：只在 refs=0 时释放。
 *
 * 思考：如果 PyTorch 释放了 tensor 但 MR Cache 还有旧条目？
 *   → 只要上层正确调用了 DeregMr（通过 proxy 的 deregBuff），
 *     refs 会降到 0 并释放。但如果上层漏调，MR 会泄漏。
 */
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  struct ncclIbVerbs* verbs = (struct ncclIbVerbs*)comm;
  struct ncclIbMrCache* cache = &ncclIbDevs[verbs->dev].mrCache;
  ncclResult_t res;
  pthread_mutex_lock(&ncclIbDevs[verbs->dev].lock);
  for (int i=0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECKGOTO(wrap_ibv_dereg_mr((struct ibv_mr*)mhandle), res, returning);
      }
      res = ncclSuccess;
      goto returning;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  res = ncclInternalError;
returning:
  pthread_mutex_unlock(&ncclIbDevs[verbs->dev].lock);
  return res;
}

/*
 * 【Q1/Q2/Q3 答案】真正执行 RDMA WRITE 的函数
 *
 * Q1: opcode 是什么？
 *   → IBV_WR_RDMA_WRITE（数据）+ IBV_WR_RDMA_WRITE_WITH_IMM（完成通知）
 *   → 为什么不用 SEND？因为 RDMA WRITE 是单边操作，不需要对端 CPU 参与，
 *     写入直接落到对端内存。SEND 是双边操作，对端需要提前 post recv + 拷贝。
 *
 * Q2: sg_list 地址是 GPU 还是 Host？
 *   → sge.addr = reqs[r]->send.data，来自上层传入的 buffer 地址。
 *     如果是 GDR 场景，这就是 GPU 显存地址（通过 nvidia-peermem pin 过）。
 *     sge.lkey 来自 MR 缓存中注册的 lkey。
 *
 * Q3: 每次发多大？谁决定切分？
 *   → size 来自 ncclIbIsend 的参数（上层 proxy 传入）。
 *     如果有多个 QP（nqps > 1），数据按 QP 数量切分，每个 QP 发 chunkSize。
 *     chunkSize 按 128B 对齐（LL/LL128 协议要求）。
 *
 * 发送策略：
 *   - 如果数据量 > AR_THRESHOLD 或 multi-recv，先发 RDMA_WRITE（数据），
 *     再发 0 字节 RDMA_WRITE_WITH_IMM（通知接收端完成）。
 *     分两步是为了支持自适应路由（Adaptive Routing）。
 *   - 小数据直接一个 RDMA_WRITE_WITH_IMM 搞定。
 */
ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  uint64_t wr_id = 0ULL;

  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;   // 【数据面-第一次读】本地 buffer（可为 GDR 地址）
    sge->lkey=reqs[r]->send.lkey;

    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->wr.rdma.rkey = slots[r].rkey;
    wr->next = wr+1;
    wr_id += (reqs[r] - comm->verbs.reqs) << (r*8);
  }

  // Write size as immediate data. In the case of multi-send, only write
  // 0 or 1 as size to indicate whether there was data sent or received.
  uint32_t immData = 0;
  if (nreqs == 1) {
    immData = reqs[0]->send.size;
  } else {
    if (nreqs > 32) {
      WARN("Cannot store sizes of %d requests in a 32-bits field", nreqs);
      return ncclInternalError;
    }
    for (int r=0; r<nreqs; r++) {
      immData |= (reqs[r]->send.size ? 1 : 0) << r;
    }
  }

  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || reqs[0]->send.size > ncclParamIbArThreshold()) {
    // When using adaptive routing, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128 protocols still work
  const int align = 128;
  for (int q=0; q<comm->nqps; q++) {
    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, comm->nqps), align) * align;
      int length = std::min(reqs[r]->send.size-reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges+r;
        comm->wrs[r].num_sge = 1;
      }
    }
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->qps[q], comm->wrs, &bad_wr));  // 【数据面-第一次读】ibv_post_send：把 WR 提交到 QP 的 SQ，HCA 执行 RDMA Write

    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, comm->nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }
  }

  return ncclSuccess;
}

/*
 * 【Q4 答案 + 数据面入口】异步发送：Isend
 *
 * Q4: remote_addr 和 rkey 从哪来？
 *   → 来自 comm->fifo[slot]（即 ncclIbSendFifo）。
 *     接收端在 ncclIbIrecv 中通过 RDMA WRITE 把 {addr, rkey, size} 写到
 *     发送端的 fifo。发送端在这里轮询 fifo[slot][0].idx，
 *     匹配到当前 idx 说明对端已就绪，从 slots[r].addr / slots[r].rkey 取值。
 *
 * 整体流程：
 *   1. 检查连接是否 ready（首次进入时走 ncclSendCheck 完成 RTR→RTS）
 *   2. 轮询 FIFO：if (slots[0].idx != idx) return（对端未就绪，下次再来）
 *   3. __sync_synchronize() 保证 idx 读取和后续 addr/rkey 读取的顺序
 *   4. 构建 request，记录 data/size/lkey
 *   5. 调 ncclIbMultiSend 执行真正的 RDMA WRITE
 *   6. 清理 FIFO slot，推进 fifoHead
 *
 * 非阻塞设计：
 *   如果 FIFO 中还没有对端的通知（idx 不匹配），直接返回 *request=NULL，
 *   上层 proxy 会在下一次 progress 中再调。绝不阻塞等待。
 */
ncclResult_t ncclIbIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->ready == 0) NCCLCHECK(ncclSendCheck(comm));
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }

  struct ibv_mr* mr = (struct ibv_mr*)mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = (comm->fifoHead)%MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  int idx = comm->fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  __sync_synchronize(); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    // Sanity checks to catch user collective call count/size mismatches
    if (size > slots[r].size) {
      char line[SOCKET_NAME_MAXLEN+1];
      WARN("NET/IB : req %d/%d tag %x peer %s collective mismatch error, local size %d remote size %d",
           r, nreqs, tag, ncclSocketToString(&comm->sock.addr, line), size, slots[r].size);
      return ncclInvalidUsage;
    } // plus any potential programming errors
    else if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkey == 0) {
     char line[SOCKET_NAME_MAXLEN+1];
     WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %d addr %lx rkey %x",
          r, nreqs, tag, ncclSocketToString(&comm->sock.addr, line), slots[r].size, slots[r].addr, slots[r].rkey);
      return ncclInternalError;
    }
    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->verbs, &req));
    req->type = NCCL_NET_IB_REQ_SEND;
    req->addr = &comm->sock.addr;
    req->verbs = &comm->verbs;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.lkey = mr->lkey;
    req->send.offset = 0;
    req->addr = &comm->sock.addr;
    req->events = comm->nqps;
    *request = reqs[r] = req;

    // If this is a multi-recv, send only when all requests have matched.
    for (int r=0; r<nreqs; r++) {
      if (reqs[r] == NULL) return ncclSuccess;
    }

    TIME_START(0);
    NCCLCHECK(ncclIbMultiSend(comm, slot));  // 【数据面-第一次读】此处内部调 wrap_ibv_post_send，把 RDMA Write 提交到 QP

    // Clear slots[0]->nreqs, as well as other fields to help debugging and sanity checks
    memset((void*)slots, 0, sizeof(struct ncclIbSendFifo));
    memset(reqs, 0, NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbRequest*));
    comm->fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

/*
 * 【FIFO 通知机制】接收端 RDMA WRITE 通知到发送端 FIFO
 *
 * 接收端把自己的 {addr, rkey, size, tag, idx} 填入 localElem，
 * 然后 RDMA WRITE 到发送端的 fifo 地址（remFifo.addr + offset）。
 * 发送端在 ncclIbIsend 中轮询这个 FIFO 的 idx 字段。
 *
 * 关键细节：
 *   - opcode = IBV_WR_RDMA_WRITE（不带 IMM，不生成对端 CQE）
 *   - 可能使用 IBV_SEND_INLINE（数据量小时内联到 WR 中，省一次 DMA）
 *   - 每 MAX_REQUESTS 轮加一次 IBV_SEND_SIGNALED，防止 SQ 积满
 *     （Unsignaled WR 不生成 CQE，但仍占 SQ 槽位，必须偶尔 signal 一次来清理）
 */
ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data, int* sizes, int* tags, void** mhandles, struct ncclIbRequest* req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remFifo.fifoTail%MAX_REQUESTS;
  struct ncclIbSendFifo* localElem = comm->remFifo.elems[slot];

  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ibv_mr* mr = (struct ibv_mr*)mhandles[i];
    localElem[i].rkey = mr->rkey;
    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->remFifo.fifoTail+1;
  }

  wr.wr.rdma.remote_addr = comm->remFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);
  wr.wr.rdma.rkey = comm->remFifo.rkey;
  comm->remFifo.sge.addr = (uint64_t)localElem;
  comm->remFifo.sge.length = n*sizeof(struct ncclIbSendFifo);
  wr.sg_list = &comm->remFifo.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remFifo.flags; // IBV_SEND_INLINE

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  if (slot == 0) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->verbs.reqs;
    req->events++;
  }

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->qps[0], &wr, &bad_wr));  // 【数据面-第一次读】recv 侧：RDMA Write 把 recv 信息写到对端 FIFO，通知 sender 可发送
  comm->remFifo.fifoTail++;

  return ncclSuccess;
}

/*
 * 【接收路径】异步接收：Irecv
 *
 * 接收端的核心逻辑，分两步：
 *
 * 第一步：在 RQ 上 post recv WR
 *   这里的 recv WR 不携带 buffer（sg_list=NULL, num_sge=0），
 *   因为真正的数据是通过 RDMA WRITE 直接写入接收缓冲区的。
 *   post recv 只是为了接收 RDMA_WRITE_WITH_IMM 的完成通知
 *   （IMM 数据包含传输的数据大小）。
 *
 * 第二步：ncclIbPostFifo 通知发送端
 *   把 {addr, rkey, size, tag, idx} RDMA WRITE 到发送端的 FIFO 中，
 *   告诉发送端 "我的接收缓冲区已准备好，请往这里写数据"。
 *
 * 这种 "先通知再写" 的设计是 NCCL 的精髓：
 *   传统 RDMA：连接建立时交换 buffer 信息，之后固定使用
 *   NCCL：每次传输动态交换 buffer 信息，支持不同大小的 buffer
 */
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->ready == 0) NCCLCHECK(ncclRecvCheck(comm));
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->verbs, &req));
  req->type = NCCL_NET_IB_REQ_RECV;
  req->addr = &comm->sock.addr;
  req->nreqs = n;
  for (int i=0; i<n; i++) req->recv.sizes[i] = 0;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->verbs.reqs;

  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  for (int q=0; q<comm->nqps; q++) {
    struct ibv_qp* qp = comm->qps[q];
    struct ibv_recv_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_recv(qp, &wr, &bad_wr));  // 【数据面-第一次读】ibv_post_recv：在 RQ 上挂 recv WR，等对端 RDMA_WRITE_WITH_IMM
  }
  TIME_STOP(1);
  req->events = comm->nqps;

  *request = req;

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->gpuFlush.enabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->verbs, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->addr = &comm->sock.addr;
  struct ibv_mr* mr = (struct ibv_mr*)mhandles[last];

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->verbs.reqs;

  wr.wr.rdma.remote_addr = (uint64_t)data[last];
  wr.wr.rdma.rkey = mr->rkey;
  wr.sg_list = &comm->gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  TIME_START(4);
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->gpuFlush.qp, &wr, &bad_wr));
  TIME_STOP(4);

  *request = req;
  return ncclSuccess;
}

/*
 * 【完成检测】轮询 CQ 检查操作是否完成
 *
 * 被上层 proxy 反复调用，直到 *done=1。
 *
 * 工作原理：
 *   - 每个 request 有 events 计数（初始值 = nqps，多 QP 时 > 1）
 *   - ibv_poll_cq 取出完成的 WC（Work Completion）
 *   - 每个 WC 对应一个完成事件，events--
 *   - 所有 events 降到 0 → 操作完成
 *
 * 对于 RECV 类型的请求：
 *   WC 的 opcode 是 IBV_WC_RECV_RDMA_WITH_IMM（对端 WRITE_WITH_IMM 触发）
 *   imm_data 携带发送的数据大小
 *
 * 对于 SEND 类型的请求：
 *   WC 的 wr_id 编码了多个 request 的索引（每 8 位一个）
 *   需要逐个递减每个 sub-request 的 events
 */
ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  while (1) {
    if (r->events == 0) {
      *done = 1;
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i=0; i<r->nreqs; i++) sizes[i] = r->recv.sizes[i];
      }
      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    int wrDone = 0;
    struct ibv_wc wcs[4];
    TIME_START(3);
    NCCLCHECK(wrap_ibv_poll_cq(r->verbs->cq, 4, wcs, &wrDone));
    if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
    if (wrDone == 0) return ncclSuccess;

    for (int w=0; w<wrDone; w++) {
      struct ibv_wc *wc = wcs+w;
      if (wc->status != IBV_WC_SUCCESS) {
        char line[SOCKET_NAME_MAXLEN+1];
        WARN("NET/IB : Got completion from peer %s with error %d, opcode %d, len %d, vendor err %d",
             ncclSocketToString(r->addr, line), wc->status, wc->opcode, wc->byte_len, wc->vendor_err);
        return ncclRemoteError;
      }

      struct ncclIbRequest* req = r->verbs->reqs+(wc->wr_id & 0xff);
      if (req->type == NCCL_NET_IB_REQ_SEND) {
        for (int i=0; i<req->nreqs; i++) {
          struct ncclIbRequest* sendReq = r->verbs->reqs+((wc->wr_id >> (i*8)) & 0xff);
          if ((sendReq->events <= 0)) return ncclInternalError;
          sendReq->events--;
        }
      } else {
        if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          if (req->type != NCCL_NET_IB_REQ_RECV) return ncclInternalError;
          if (req->nreqs > 1) {
            // In the case of a multi recv, we only set sizes to 0 or 1.
            for (int i=0; i<req->nreqs; i++) {
              req->recv.sizes[i] = (wc->imm_data >> i) & 0x1;
            }
          } else {
            req->recv.sizes[0] += wc->imm_data;
          }
        }
        req->events--;
      }
    }
  }
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    close(comm->sock.fd);
    for (int q=0; q<comm->nqps; q++)
      if (comm->qps[q] != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->qps[q]));
    if (comm->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->fifoMr));
    NCCLCHECK(ncclIbDestroyVerbs(&comm->verbs));
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    close(comm->sock.fd);
    for (int q=0; q<comm->nqps; q++)
      if (comm->qps[q] != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->qps[q]));
    if (comm->gpuFlush.enabled) {
      if (comm->gpuFlush.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->gpuFlush.qp));
      if (comm->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->gpuFlush.hostMr));
    }
    if (comm->remFifo.mr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remFifo.mr));
    NCCLCHECK(ncclIbDestroyVerbs(&comm->verbs));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    close(comm->sock.fd);
    free(comm);
  }
  return ncclSuccess;
}

/*
 * 【导出函数表】ncclNet_t 接口实现
 *
 * 这是 NCCL 网络插件的标准接口。NCCL 核心通过这个函数指针表
 * 调用具体的传输实现。除了 IB，NCCL 还有 Socket 传输（net_socket.cc）。
 *
 * 函数调用顺序（典型生命周期）：
 *   init → devices → getProperties → listen/connect/accept
 *   → regMr → isend/irecv → test → deregMr → close
 */
ncclNet_t ncclNetIb = {
  "IB",
  ncclIbInit,
  ncclIbDevices,
  ncclIbGetProperties,
  ncclIbListen,
  ncclIbConnect,
  ncclIbAccept,
  ncclIbRegMr,
  ncclIbRegMrDmaBuf,
  ncclIbDeregMr,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbIflush,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen
};

