/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_BOOTSTRAP_H_
#define NCCL_BOOTSTRAP_H_

#include "nccl.h"
#include "comm.h"

/*
 * Bootstrap 带外（OOB）控制面：在 transport / collective 数据面建立之前，
 * 负责 rank 会合与初始化元数据交换（peerInfo、拓扑信息等）。
 * 默认走 TCP socket（NCCL_OOB_NET_ENABLE=0），不是 IB/RoCE 上的 collective 数据通道。
 *
 * 下列 API 中的 commState 即 comm->bootstrap（struct bootstrapState*）。
 */

/* 嵌入 ncclUniqueId 的句柄：root 地址 + 会话 magic。 */
struct ncclBootstrapHandle {
  uint64_t magic;                  /* bootstrap 会话标识 */
  union ncclSocketAddress addr;    /* root 监听地址 */
  int nRanks;                      /* 已有 rank 数（Grow 场景）；否则可忽略 */
};
static_assert(sizeof(struct ncclBootstrapHandle) <= sizeof(ncclUniqueId),
              "Bootstrap handle is too large to fit inside NCCL unique ID");

/* === 进程级准备 === */

/* 进程内初始化 OOB 网络栈（socket 或 bootstrap net），通常调用一次。 */
ncclResult_t bootstrapNetInit();

/* === 会话 / uniqueId === */

/* 生成新 communicator 用的 bootstrap 句柄；comm 可为 NULL。 */
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm);

/* 根据句柄启动 bootstrap root；idFromEnv 表示地址来自 NCCL_COMM_ID 等环境变量。 */
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv);

/* Grow 时由父 comm 边界 rank 广播 grow handle（配合 ncclCommGrow）。 */
ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot);

/* === 挂接 communicator === */

/*
 * 加入 bootstrap 会话，为本 comm 建立控制面 ring。
 * handle：来自 uniqueId 的一个或多个 ncclBootstrapHandle；Grow 时 parent 非空。
 * 成功后会设置 comm->bootstrap 与 comm->magic。
 *
 * 大白话：所有 rank 到签到台上报自己的电话号码，签到台告诉每个人：你的下一个联系人是谁
 */
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent);

/*
 * 子 comm（Split/Shrink）：借助父 comm 控制面建立子群 ring。
 * parentRanks[i] 为子 rank i 在父 comm 中的 rank 编号。
 */
ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks);

/* === 元数据集体操作（仅控制面） === */

/* 每 rank 贡献 size 字节，结果写入 allData（长度 nranks * size）。 */
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size);

/* 控制面点对点发送/接收；tag 区分消息类型。 */
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size);
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size);

/* 全员 barrier / 单 root 广播（初始化元数据，非 GPU collective）。 */
ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag);
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size);

/* 同上，但仅在本机 ranks[] 子集内执行。 */
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag);
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size);
ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                         int size);

/* === 收尾 === */

/* 正常关闭 bootstrap 状态并释放资源。 */
ncclResult_t bootstrapClose(void* commState);

/* 异常路径中止 bootstrap（如 comm abort）。 */
ncclResult_t bootstrapAbort(void* commState);
#endif
