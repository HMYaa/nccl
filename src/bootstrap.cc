/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "bootstrap.h"

#include <sys/types.h>
#include <unistd.h>

#include "core.h"
#include "nccl.h"
#include "net.h"
#include "proxy.h"
#include "utils.h"

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
static union ncclSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetInitDone == 0) {
      char* env = getenv("NCCL_COMM_ID");
      if (env) {
        union ncclSocketAddress remoteAddr;
        if (ncclGetSocketAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN(
              "Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or "
              "<hostname>:<port>");
          return ncclInvalidArgument;
        }
        if (ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr,
                                         MAX_IF_NAME_SIZE, 1) <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        int nIfs = ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInternalError;
        }
      }
      char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
      sprintf(line, " %s:", bootstrapNetIfName);
      ncclSocketToString(&bootstrapNetIfAddr, line + strlen(line));
      INFO(NCCL_INIT, "Bootstrap : Using%s", line);
      bootstrapNetInitDone = 1;
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// Additional sync functions
static ncclResult_t bootstrapNetSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}
static ncclResult_t bootstrapNetRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  NCCLCHECK(ncclSocketRecv(sock, data, std::min(recvSize, size)));
  return ncclSuccess;
}

struct extInfo {
  int rank;
  int nranks;
  union ncclSocketAddress extAddressListenRoot;
  union ncclSocketAddress extAddressListen;
};

#include <sys/resource.h>

static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

static void* bootstrapRoot(void* args) {
  struct ncclSocket* listenSock = (struct ncclSocket*)args;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  struct extInfo info;
  union ncclSocketAddress* rankAddresses = NULL;
  union ncclSocketAddress* rankAddressesRoot =
      NULL;  // for initial rank <-> root information exchange
  union ncclSocketAddress* zero = NULL;
  NCCLCHECKGOTO(ncclCalloc(&zero, 1), res, out);
  setFilesLimit();

  TRACE(NCCL_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    struct ncclSocket sock;
    /* bootstrap root thread always uses blocking ncclSocketAccept. */
    NCCLCHECKGOTO(ncclSocketInit(&sock, NULL, NULL, 0), res, out);
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);
    NCCLCHECKGOTO(bootstrapNetRecv(&sock, &info, sizeof(info)), res, out);
    close(sock.fd);

    if (c == 0) {
      nranks = info.nranks;
      NCCLCHECKGOTO(ncclCalloc(&rankAddresses, nranks), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks, info.nranks);
      goto out;
    }

    if (memcmp(zero, &rankAddressesRoot[info.rank], sizeof(union ncclSocketAddress)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    // Save the connection handle for that rank
    memcpy(rankAddressesRoot + info.rank, &info.extAddressListenRoot,
           sizeof(union ncclSocketAddress));
    memcpy(rankAddresses + info.rank, &info.extAddressListen, sizeof(union ncclSocketAddress));

    ++c;
    TRACE(NCCL_INIT, "Received connect from rank %d total %d/%d", info.rank, c, nranks);
  } while (c < nranks);
  TRACE(NCCL_INIT, "COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r = 0; r < nranks; ++r) {
    int next = (r + 1) % nranks;
    struct ncclSocket sock;
    sock.abortFlag = NULL;
    sock.asyncFlag = 0;
    memcpy(&sock.addr, rankAddressesRoot + r, sizeof(union ncclSocketAddress));
    NCCLCHECKGOTO(ncclSocketConnect(&sock), res, out);
    NCCLCHECKGOTO(bootstrapNetSend(&sock, rankAddresses + next, sizeof(union ncclSocketAddress)),
                  res, out);
    close(sock.fd);
  }
  TRACE(NCCL_INIT, "SENT OUT ALL %d HANDLES", nranks);

out:
  close(listenSock->fd);
  free(listenSock);
  if (rankAddresses) free(rankAddresses);
  if (rankAddressesRoot) free(rankAddressesRoot);
  if (zero) free(zero);

  TRACE(NCCL_INIT, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(ncclUniqueId* id, bool idFromEnv) {
  struct ncclSocket* listenSock;
  NCCLCHECK(ncclCalloc(&listenSock, 1));
  memcpy(&listenSock->addr, id, sizeof(union ncclSocketAddress));
  NCCLCHECK(ncclSocketListen(listenSock));
  memcpy(id, &listenSock->addr, sizeof(union ncclSocketAddress));
  pthread_t thread;
  pthread_create(&thread, NULL, bootstrapRoot, (void*)listenSock);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  pthread_detach(thread);  // will not be pthread_join()'d
  return ncclSuccess;
}

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* id) {
  static_assert(sizeof(union ncclSocketAddress) < sizeof(ncclUniqueId),
                "NetId does not fit inside ncclUniqueId");
  memset(id, 0, sizeof(ncclUniqueId));
  union ncclSocketAddress* connectAddr = (union ncclSocketAddress*)id;

  char* env = getenv("NCCL_COMM_ID");
  if (env) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (ncclGetSocketAddrFromString(connectAddr, env) != ncclSuccess) {
      WARN(
          "Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or "
          "<hostname>:<port>");
      return ncclInvalidArgument;
    }
  } else {
    memcpy(id, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    NCCLCHECK(bootstrapCreateRoot(id, false));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

// 【控制面/建链视角说明】
// 这个结构是 NCCL 在“Bootstrap 阶段”的控制面状态机。
// 典型生命周期（和 initTransportsRank 中的调用对应）：
//   - initTransportsRank 调用 bootstrapInit(commId, comm)
//   - bootstrapInit 为每个 rank 分配一个 bootstrapState，建立以下几类 Socket：
//       * listenSock       : 本 rank 对外暴露的“控制端点”（别的 rank/Proxy 可以来连我）
//       * ringSendSocket   : AllGather 环中 “我 -> 右邻居” 的 Socket
//       * ringRecvSocket   : AllGather 环中 “左邻居 -> 我” 的 Socket
//   - 后续 bootstrapAllGather / bootstrapSend / bootstrapRecv 都只在这一小撮 Socket
//   上收发“控制消息”，
//     比如：peer 的监听地址、Proxy 的监听地址、拓扑/图信息等。
//   - 真正的数据面 RDMA/IB 通道是在后面的 transport/net_ib.cc 里用这些地址再建立 QP。
//
// 从 RDMA 驱动/控制面角度看：bootstrapState 就是“控制链路的连接表 + 未预期连接缓存”。
struct bootstrapState {
  struct ncclSocket listenSock;
  struct ncclSocket ringRecvSocket;
  struct ncclSocket ringSendSocket;
  union ncclSocketAddress* peerCommAddresses;
  union ncclSocketAddress* peerProxyAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  volatile uint32_t* abortFlag;
};

ncclResult_t bootstrapInit(ncclUniqueId* id, struct ncclComm* comm) {
  // 【整体作用】
  // 为当前 rank 搭一套“控制面 Bootstrap 网络”：
  //   1. 每个 rank 启一个监听 Socket（listenSock），把自己的监听地址发给 root；
  //   2. root 把“下一跳 rank”的地址发回来，形成一个逻辑 ring（ringSendSocket / ringRecvSocket）；
  //   3. 基于这个 ring，所有 rank 可以做 ring AllGather，后续 initTransportsRank 里就用它来：
  //        - AllGather peer 的通信地址（peerCommAddresses）
  //        - AllGather Proxy 的服务地址（peerProxyAddresses）
  //   4. 最后调用 ncclProxyInit，把 Proxy 线程的“服务入口”也挂到这套控制面上。
  //
  // 注意：这里所有 Socket 通信只在“控制/建链阶段”传很少量的元数据，不走真正的数据面大流量。
  int rank = comm->rank;
  int nranks = comm->nRanks;
  struct bootstrapState* state;
  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->abortFlag = comm->abortFlag;
  comm->bootstrap = state;

  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  struct extInfo info = {0};
  info.rank = rank;
  info.nranks = nranks;
  struct ncclSocket sock, listenSockRoot;

  // ========== 1. 初始化若干控制面 Socket 句柄 ==========
  // sock           : 用来连 root，把自己的 extInfo 发上去，再从 root 那里收“下一跳”的地址
  // listenSockRoot : 本 rank 临时起一个“给 root 回连”的监听口（extAddressListenRoot）
  // state->listenSock : 本 rank 的长期监听口，给其他 rank/Proxy 用（extAddressListen）
  // ringSendSocket/ringRecvSocket : 稍后用于 ring AllGather 的“右发左收”两端
  NCCLCHECK(ncclSocketInit(&sock, (union ncclSocketAddress*)id, comm->abortFlag, 0));
  NCCLCHECK(ncclSocketInit(&listenSockRoot, &bootstrapNetIfAddr, comm->abortFlag, 0));
  NCCLCHECK(ncclSocketInit(&state->listenSock, &bootstrapNetIfAddr, comm->abortFlag, 0));
  NCCLCHECK(ncclSocketInit(&state->ringSendSocket, NULL, comm->abortFlag, 0));
  NCCLCHECK(ncclSocketInit(&state->ringRecvSocket, NULL, comm->abortFlag, 0));
  // Create socket for other ranks to contact me
  // 【控制面-对外入口】其它 rank 以后要直连我（比如 bootstrapSend/Recv），都用这个地址
  NCCLCHECK(ncclSocketListen(&state->listenSock));
  memcpy(&info.extAddressListen, &state->listenSock.addr, sizeof(union ncclSocketAddress));

  // Create socket for root to contact me
  // 【只用于建 ring】root 线程稍后会回连到这个 listenSockRoot，把“下一跳”的地址告诉我
  NCCLCHECK(ncclSocketListen(&listenSockRoot));
  memcpy(&info.extAddressListenRoot, &listenSockRoot.addr, sizeof(union ncclSocketAddress));

  // stagger connection times to avoid an overload of the root
  if (nranks > 128) {
    long msec = rank;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(NCCL_INIT, "rank %d delaying connection to root by %ld msec", rank, msec);
    (void)nanosleep(&tv, NULL);
  }

  // send info on my listening socket to root
  // 【步骤 A】连上 root，报告自己的 rank / nranks 以及两个监听地址：
  //   - extAddressListen      : 给其他 rank 用的“长期服务口”
  //   - extAddressListenRoot  : 给 root 建 ring 时临时用的“回连口”
  NCCLCHECK(ncclSocketConnect(&sock));
  NCCLCHECK(bootstrapNetSend(&sock, &info, sizeof(info)));
  close(sock.fd);

  // get info on my "next" rank in the bootstrap ring from root
  // 【步骤 B】等待 root 回连到 listenSockRoot，root 会告诉我：
  //   - 在 ring AllGather 里，“我右边那个 rank”的地址（state->ringSendSocket.addr）
  //   之后：
  //     - ringSendSocket  -> 连接到“右邻居”，用于向右发送
  //     - ringRecvSocket  -> 接受来自“左邻居”的连接，用于从左接收
  NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
  NCCLCHECK(bootstrapNetRecv(&sock, &state->ringSendSocket.addr, sizeof(union ncclSocketAddress)));
  close(sock.fd);
  close(listenSockRoot.fd);

  NCCLCHECK(ncclSocketConnect(&state->ringSendSocket));
  // Accept the connect request from the previous rank in the AllGather ring
  NCCLCHECK(ncclSocketAccept(&state->ringRecvSocket, &state->listenSock));

  // AllGather all listen handlers
  // 【步骤 C】基于刚刚搭好的 ring 执行第一次 AllGather：
  //   - 收集所有 rank 的“长期服务口”地址（peerCommAddresses）
  //   - 结果：state->peerCommAddresses[r] 保存 rank r 的 listenSock.addr
  NCCLCHECK(ncclCalloc(&state->peerCommAddresses, nranks));
  memcpy(state->peerCommAddresses + rank, &state->listenSock.addr, sizeof(union ncclSocketAddress));
  NCCLCHECK(bootstrapAllGather(state, state->peerCommAddresses, sizeof(union ncclSocketAddress)));

  // Create the service proxy
  // 【步骤 D】为 Proxy 服务也建一套“控制入口”：
  //   1. 每个 rank 再起一个 proxySocket 监听口
  //   2. 再做一次 ring AllGather，收集所有 rank 的 proxySocket 地址（peerProxyAddresses）
  //   3. 调用 ncclProxyInit，把“Proxy 线程的服务端点”注册进来
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  struct ncclSocket* proxySocket;
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECK(ncclSocketInit(proxySocket, &bootstrapNetIfAddr, NULL, 0));
  NCCLCHECK(ncclSocketListen(proxySocket));
  memcpy(state->peerProxyAddresses + rank, &proxySocket->addr, sizeof(union ncclSocketAddress));
  NCCLCHECK(bootstrapAllGather(state, state->peerProxyAddresses, sizeof(union ncclSocketAddress)));
  NCCLCHECK(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses));

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  char* data = (char*)allData;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_INIT, "rank %d nranks %d size %d", rank, nranks, size);

  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i = 0; i < nranks - 1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right
    NCCLCHECK(bootstrapNetSend(&state->ringSendSocket, data + sslice * size, size));
    // Recv slice from the left
    NCCLCHECK(bootstrapNetRecv(&state->ringRecvSocket, data + rslice * size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  struct ncclSocket sock;

  NCCLCHECK(ncclSocketInit(&sock, state->peerCommAddresses + peer, state->abortFlag, 1));
  NCCLCHECK(ncclSocketConnect(&sock));
  NCCLCHECK(bootstrapNetSend(&sock, &state->rank, sizeof(int)));
  NCCLCHECK(bootstrapNetSend(&sock, &tag, sizeof(int)));
  NCCLCHECK(bootstrapNetSend(&sock, data, size));
  close(sock.fd);
  return ncclSuccess;
}

ncclResult_t bootstrapBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  if (nranks == 1) return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d tag %x - ENTER", rank, nranks, tag);

  /* Simple intra process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming,
   * 17(1):1-17, 1988"
   */
  int data[1];
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks[dst], tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks[src], tag, data, sizeof(data)));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d tag %x - DONE", rank, nranks, tag);
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks,
                                         void* allData, int size) {
  if (nranks == 1) return ncclSuccess;
  char* data = (char*)allData;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  for (int i = 1; i < nranks; i++) {
    int src = (rank - i + nranks) % nranks;
    int dst = (rank + i) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks[dst], /*tag=*/i, data + rank * size, size));
    NCCLCHECK(bootstrapRecv(commState, ranks[src], /*tag=*/i, data + src * size, size));
  }

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag,
                               struct ncclSocket* sock) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}

ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag,
                               struct ncclSocket* sock) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));
      free(elem);
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  sock->fd = -1;
  return ncclSuccess;
}

// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  struct ncclSocket sock;

  // Search unexpected connections first
  NCCLCHECK(unexpectedDequeue(state, peer, tag, &sock));
  if (sock.fd != -1) {
    NCCLCHECK(bootstrapNetRecv(&sock, ((char*)data), size));
    close(sock.fd);
    return ncclSuccess;
  }

  // Then look for new connections
  NCCLCHECK(ncclSocketInit(&sock, NULL, state->listenSock.abortFlag, 0));
  while (1) {
    NCCLCHECK(ncclSocketAccept(&sock, &state->listenSock));
    int newPeer, newTag;
    NCCLCHECK(bootstrapNetRecv(&sock, &newPeer, sizeof(int)));
    NCCLCHECK(bootstrapNetRecv(&sock, &newTag, sizeof(int)));
    if (newPeer == peer && newTag == tag) {
      NCCLCHECK(bootstrapNetRecv(&sock, ((char*)data), size));
      close(sock.fd);
      return ncclSuccess;
    }
    // Unexpected connection. Save for later.
    NCCLCHECK(unexpectedEnqueue(state, newPeer, newTag, &sock));
  }
}

ncclResult_t bootstrapClose(void* commState) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (state->unexpectedConnections != NULL) {
    WARN("Unexpected connections are not empty");
    return ncclInternalError;
  }
  if (state->listenSock.fd >= 0) close(state->listenSock.fd);
  if (state->ringSendSocket.fd >= 0) close(state->ringSendSocket.fd);
  if (state->ringRecvSocket.fd >= 0) close(state->ringRecvSocket.fd);

  free(state->peerCommAddresses);
  free(state);

  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  struct bootstrapState* state = (struct bootstrapState*)commState;
  if (commState == NULL) return ncclSuccess;
  if (state->listenSock.fd) close(state->listenSock.fd);
  if (state->ringSendSocket.fd) close(state->ringSendSocket.fd);
  if (state->ringRecvSocket.fd) close(state->ringRecvSocket.fd);
  free(state->peerCommAddresses);
  free(state->peerProxyAddresses);
  free(state);
  return ncclSuccess;
}
