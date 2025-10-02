/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GPUNETIO_VERBS_COMMON_H_
#define GPUNETIO_VERBS_COMMON_H_

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include <doca_error.h>
#include "doca_internal.hpp"
#include <doca_verbs.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_high_level.h>
#include <doca_gpunetio_verbs_def.h>

#define NUM_QP 1
#define CUDA_THREADS_BW 512  // post list
#define CUDA_THREADS_LAT 1   // post list
#define NUM_ITERS 2048
#define NUM_MSG_SIZE 10

#define VERBS_TEST_QUEUE_SIZE (2048)
// Should be sizeof(doca gpu verbs structs)
#define VERBS_TEST_SIN_PORT (5000)
#define VERBS_TEST_HOP_LIMIT (255)
#define VERBS_TEST_LOCAL_BUF_VALUE (0xA)
#define VERBS_CUDA_BLOCK 2

#define DEFAULT_GID_INDEX (0)
#define MAX_PCI_ADDRESS_LEN 32U
#define MAX_IP_ADDRESS_LEN 128
#define SYSTEM_PAGE_SIZE 4096 /* Page size sysconf(_SC_PAGESIZE)*/

#define ROUND_UP(unaligned_mapping_size, align_val) \
    ((unaligned_mapping_size) + (align_val) - 1) & (~((align_val) - 1))

#define ALIGN_SIZE(size, align) size = ((size + (align) - 1) / (align)) * (align);

struct verbs_config {
    std::string nic_device_name; /* SF DOCA device name */
    std::string gpu_pcie_addr;   /* PF DOCA device name */
    bool is_server;              /* Sample is acting as client or server */
    uint32_t gid_index;          /* GID index */
    std::string server_ip_addr;  /* DOCA device name */
    uint32_t num_iters;          /* total number orations per cuda kernel */
    uint32_t cuda_threads;       /* cuda threads per cuda block */
    enum doca_gpu_dev_verbs_nic_handler nic_handler;
    uint8_t exec_scope;
};

struct verbs_resources {
    struct verbs_config *cfg;                 /* RDMA Verbs test configuration parameters */
    struct doca_gpu *gpu_dev;                 /* DOCA GPU device to use */
    uint8_t *data_buf[NUM_MSG_SIZE];          /* The local data buffer */
    uint64_t *flag_buf[NUM_MSG_SIZE];         /* The local data buffer */
    uint64_t remote_data_buf[NUM_MSG_SIZE];   /* The remote buffer */
    uint64_t remote_flag_buf[NUM_MSG_SIZE];   /* The remote buffer */
    struct ibv_context *verbs_context;        /* DOCA Verbs Context */
    struct ibv_pd *verbs_pd;                  /* local protection domain */
    struct doca_verbs_ah_attr *verbs_ah_attr; /* DOCA Verbs address handle */
    struct doca_gpu_verbs_qp_hl *qp;          /* DOCA GPUNetIO high-level Verbs QP */
    int conn_socket;                          /* Connection socket fd */
    uint32_t local_qp_number;                 /* Local QP number */
    uint32_t remote_qp_number;                /* Remote QP number */
    uint32_t remote_data_mkey[NUM_MSG_SIZE];  /* remote MKEY */
    uint32_t remote_flag_mkey[NUM_MSG_SIZE];  /* remote MKEY */
    struct ibv_mr *data_mr[NUM_MSG_SIZE];     /* local memory region */
    struct ibv_mr *flag_mr[NUM_MSG_SIZE];     /* local memory region */
    struct doca_verbs_gid gid;                /* local gid address */
    struct doca_verbs_gid remote_gid;         /* remote gid address */
    int lid;                                  /* IB: local ID */
    int dlid;                                 /* IB: destination ID */
    uint32_t num_iters;                       /* total number of iterations per cuda kernel */
    uint32_t cuda_threads;                    /* threads */
    enum doca_gpu_dev_verbs_nic_handler nic_handler; /* enable CPU proxy */
    enum doca_gpu_dev_verbs_exec_scope scope;

    /* write_lat test */
    struct ibv_mr *local_poll_mr[NUM_MSG_SIZE]; /* local memory region */
    struct ibv_mr *local_post_mr[NUM_MSG_SIZE]; /* local memory region */

    uint8_t *local_poll_buf[NUM_MSG_SIZE]; /* The local data buffer */
    uint8_t *local_post_buf[NUM_MSG_SIZE]; /* The local data buffer */
};

struct cpu_proxy_args {
    struct doca_gpu_verbs_qp *qp_cpu;
    uint64_t *exit_flag;
};

/*
 * Target side of the RDMA Verbs sample
 *
 * @cfg [in]: Configuration parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t verbs_server(struct verbs_config *cfg);

/*
 * Initiator side of the RDMA Verbs sample
 *
 * @cfg [in]: Configuration parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t verbs_client(struct verbs_config *cfg);

/*
 * OOB connection to exchange RDMA info - server side
 *
 * @oob_sock_fd [out]: Socket FD
 * @oob_client_sock [out]: Client socket FD
 * @return: positive integer on success and -1 otherwise
 */
int oob_verbs_connection_server_setup(int *oob_sock_fd, int *oob_client_sock);

/*
 * OOB connection to exchange RDMA info - server side closure
 *
 * @oob_sock_fd [in]: Socket FD
 * @oob_client_sock [in]: Client socket FD
 */
void oob_verbs_connection_server_close(int oob_sock_fd, int oob_client_sock);

/*
 * OOB connection to exchange RDMA info - client side
 *
 * @server_ip [in]: Server IP address to connect
 * @oob_sock_fd [out]: Socket FD
 * @return: positive integer on success and -1 otherwise
 */
int oob_verbs_connection_client_setup(const char *server_ip, int *oob_sock_fd);

/*
 * OOB connection to exchange RDMA info - client side closure
 *
 * @oob_sock_fd [in]: Socket FD
 */
void oob_verbs_connection_client_close(int oob_sock_fd);

/*
 * Create and initialize DOCA RDMA Verbs resources
 *
 * @cfg [in]: Configuration parameters
 * @resources [in/out]: DOCA RDMA resources to create
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_verbs_resources(struct verbs_config *cfg, struct verbs_resources *resources);

/*
 * Destroy DOCA RDMA resources
 *
 * @resources [in]: DOCA RDMA resources to destroy
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t destroy_verbs_resources(struct verbs_resources *resources);

/*
 * Connect a DOCA RDMA Verbs QP to a remote one
 *
 * @resources [in]: DOCA RDMA Verbs resources with the QP
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t connect_verbs_qp(struct verbs_resources *resources);

doca_error_t export_datapath_attr_in_gpu(struct verbs_resources *resources);

/*
 * Server side of the RDMA write
 *
 * @cfg [in]: Configuration parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t rdma_write_server(struct verbs_config *cfg);

/*
 * Client side of the RDMA write
 *
 * @cfg [in]: Configuration parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t rdma_write_client(struct verbs_config *cfg);

/*
 * CPU proxy progresses the QP
 *
 * @args [in]: thread args
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
void *progress_cpu_proxy(void *args_);

/*
 * Get system page size
 *
 * @return: system page size in bytes (default is 4096)
 */
size_t get_page_size(void);

#if __cplusplus
extern "C" {
#endif

doca_error_t gpunetio_verbs_write_lat(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                      uint32_t num_iters, uint32_t cuda_blocks,
                                      uint32_t cuda_threads, uint32_t size, uint8_t *local_poll_buf,
                                      uint32_t local_poll_mkey, uint8_t *local_post_buf,
                                      uint32_t local_post_mkey, uint8_t *dst_buf, uint32_t dst_mkey,
                                      bool is_client);

/*
 * Launch a CUDA kernel with to measure One-Sided Put Shared QP bandwidth
 *
 * @stream [in]: CUDA Stream to launch the kernel
 * @qp [in]: Verbs GPU object
 * @num_iters [in]: Total number of iterations
 * @cuda_blocks [in]: Number of CUDA blocks to launch the kernel
 * @cuda_threads [in]: Number of CUDA threads to launch the kernel
 * @data_size [in]: Data buffer size (number of bytes)
 * @src_buf [in]: Source GPU data buffer address
 * @src_buf_mkey [in]: Source GPU data buffer memory key
 * @dst_buf [in]: Destination GPU data buffer address
 * @dst_buf_mkey [in]: Destination GPU data buffer memory key
 * @scope [in]: Each put is called per CUDA thread or per CUDA warp
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpunetio_verbs_put_bw(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                   uint32_t num_iters, uint32_t cuda_blocks, uint32_t cuda_threads,
                                   uint32_t data_size, uint8_t *src_buf, uint32_t src_buf_mkey,
                                   uint8_t *dst_buf, uint32_t dst_buf_mkey,
                                   enum doca_gpu_dev_verbs_exec_scope scope);

#if __cplusplus
}
#endif

#endif /* GPUNETIO_VERBS_COMMON_H_ */
