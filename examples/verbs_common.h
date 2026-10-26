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
#include <doca_gpunetio_config.h>

#define NUM_QP 1
#define CUDA_THREADS_BW 512  // post list
#define CUDA_THREADS_LAT 1   // post list
#define NUM_ITERS 2048
#define NUM_MSG_SIZE 16
#define NUM_MSG_SIZE_LAT 10

#define VERBS_TEST_QUEUE_SIZE (2048)
// Should be sizeof(doca gpu verbs structs)
#define VERBS_TEST_SIN_PORT (5000)
#define VERBS_TEST_HOP_LIMIT (255)
#define VERBS_TEST_LOCAL_BUF_VALUE (0xA)
#define VERBS_CUDA_BLOCK 2
#define VERBS_OOB_PORT 2000 /* TCP port of the OOB connection used to exchange RDMA info */

#define DEFAULT_GID_INDEX (0)
#define MAX_PCI_ADDRESS_LEN 32U
#define MAX_IP_ADDRESS_LEN 128
#define SYSTEM_PAGE_SIZE 4096 /* Page size sysconf(_SC_PAGESIZE)*/

#define ROUND_UP(unaligned_mapping_size, align_val) \
    ((unaligned_mapping_size) + (align_val) - 1) & (~((align_val) - 1))

#define ALIGN_SIZE(size, align) size = ((size + (align) - 1) / (align)) * (align);

#define BW_FORMAT_FACTOR 1000000000  // GBPS

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
    enum doca_gpu_verbs_send_dbr_mode_ext
        send_dbr_mode_ext;   /* Enable send dbr mode ext to avoid DBREC update on sending */
    bool enable_data_direct; /* Register the GPU buffers with Data Direct */
};

struct verbs_resources {
    struct verbs_config *cfg;               /* RDMA Verbs test configuration parameters */
    doca_gpu_t *gpu_dev;                    /* DOCA GPU device to use */
    doca_dev_t *net_dev;                    /* DOCA network device to use */
    uint8_t *data_buf[NUM_MSG_SIZE];        /* The local data buffer */
    uint64_t *flag_buf[NUM_MSG_SIZE];       /* The local data buffer */
    uint64_t remote_data_buf[NUM_MSG_SIZE]; /* The remote buffer */
    uint64_t remote_flag_buf[NUM_MSG_SIZE]; /* The remote buffer */
    struct ibv_context *verbs_context;      /* DOCA Verbs Context */
    struct ibv_pd *verbs_pd;                /* local protection domain */
    doca_verbs_ah_attr_t *verbs_ah_attr;    /* DOCA Verbs address handle */
    struct doca_gpu_verbs_qp_hl *qp;        /* DOCA GPUNetIO high-level Verbs QP */
    bool qp_group;
    int conn_socket;                         /* Connection socket fd */
    uint32_t local_qp_number;                /* Local QP number */
    uint32_t remote_qp_number;               /* Remote QP number */
    uint32_t remote_data_mkey[NUM_MSG_SIZE]; /* remote MKEY */
    uint32_t remote_flag_mkey[NUM_MSG_SIZE]; /* remote MKEY */
    struct ibv_mr *data_mr[NUM_MSG_SIZE];    /* local memory region */
    struct ibv_mr *flag_mr[NUM_MSG_SIZE];    /* local memory region */
    struct doca_verbs_gid gid;               /* local gid address */
    struct doca_verbs_gid remote_gid;        /* remote gid address */
    int lid;                                 /* IB: local ID */
    int dlid;                                /* IB: destination ID */
    uint32_t num_iters;                      /* total number of iterations per cuda kernel */
    uint32_t cuda_threads;                   /* threads */
    enum doca_gpu_dev_verbs_nic_handler nic_handler; /* enable CPU proxy */
    enum doca_gpu_dev_verbs_exec_scope scope;
    uint8_t cq_collapsed; /* enable/disable cq collapsed */

    /* write_lat test */
    struct ibv_mr *local_poll_mr[NUM_MSG_SIZE]; /* local memory region */
    struct ibv_mr *local_post_mr[NUM_MSG_SIZE]; /* local memory region */

    uint8_t *local_poll_buf[NUM_MSG_SIZE]; /* The local data buffer */
    uint8_t *local_post_buf[NUM_MSG_SIZE]; /* The local data buffer */

    /* put_signal tests: signal values before each fetch and add */
    uint64_t *prev_flag_buf[NUM_MSG_SIZE];
    struct ibv_mr *prev_flag_mr[NUM_MSG_SIZE];

    /* get_bw test: flag the Dump WQE reads */
    uint64_t *dump_flag_buf;
    struct ibv_mr *dump_flag_mr;

    enum doca_gpu_verbs_send_dbr_mode_ext
        send_dbr_mode_ext; /* Enable send dbr mode ext to avoid DBREC update on sending */

    bool enable_umem_cpu;     /* Enable creation of QP/CQ UMEM on CPU pinned memory */
    bool enable_comp_channel; /* Enable creation of CQ comp channel for error checking */
    doca_verbs_comp_channel_t *comp_channel;
};

struct cpu_proxy_args {
    struct doca_gpu_verbs_qp *qp_cpu;
    struct doca_gpu_verbs_qp *qp_cpu_companion;

    uint64_t *exit_flag;
};

/* Options both peers must agree on, exchanged over the OOB socket in network byte order */
struct verbs_oob_config {
    char sample_name[64]; /* Sample binary name, catches a client connected to a different sample */
    uint32_t num_iters;   /* -i */
    uint32_t cuda_threads; /* CUDA threads of the sample kernel */
};

/*
 * Check the options that depend on each other or on the GPU, once all of them have been parsed
 *
 * @cfg [in]: Configuration parameters
 * @cuda_blocks [in]: Number of CUDA blocks the sample splits cfg->cuda_threads into
 * @max_iters [in]: Maximum number of iterations the sample supports, 0 for no limit
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t verbs_validate_config(const struct verbs_config *cfg, uint32_t cuda_blocks,
                                   uint32_t max_iters);

/*
 * Send a whole buffer over the OOB socket
 *
 * @sock_fd [in]: Socket FD
 * @buf [in]: Data to send
 * @len [in]: Number of bytes to send
 * @return: 0 on success and -1 otherwise
 */
int verbs_oob_send(int sock_fd, const void *buf, size_t len);

/*
 * Receive a whole buffer from the OOB socket
 *
 * @sock_fd [in]: Socket FD
 * @buf [out]: Received data
 * @len [in]: Number of bytes to receive
 * @return: 0 on success, -1 on error or if the peer closed the connection before sending len bytes
 */
int verbs_oob_recv(int sock_fd, void *buf, size_t len);

/*
 * Exchange the sample name and the options both peers must agree on, and fail on any mismatch
 *
 * @sock_fd [in]: Connected OOB socket FD
 * @sample_name [in]: Name of the running sample
 * @cfg [in]: Local configuration parameters
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t verbs_oob_check_config(int sock_fd, const char *sample_name,
                                    const struct verbs_config *cfg);

/*
 * Tell the peer that all the benchmark runs have completed successfully
 *
 * @sock_fd [in]: Connected OOB socket FD
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t verbs_oob_send_done(int sock_fd);

/*
 * Block until the peer reports that all the benchmark runs have completed successfully
 *
 * @sock_fd [in]: Connected OOB socket FD
 * @force_quit [in]: Flag set by the signal handler
 * @return: DOCA_SUCCESS if the peer reported the completion, DOCA_ERROR if it closed or reset the
 * connection first, or if force_quit was set
 */
doca_error_t verbs_oob_wait_done(int sock_fd, volatile bool *force_quit);

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
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t oob_verbs_connection_server_setup(int *oob_sock_fd, int *oob_client_sock);

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
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t oob_verbs_connection_client_setup(const char *server_ip, int *oob_sock_fd);

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

/*
 * CPU proxy progresses the QP
 *
 * @args_ [in]: thread args (struct cpu_proxy_args)
 * @return: NULL, pthread start routine signature
 */
void *progress_cpu_proxy(void *args_);

/*
 * Get system page size
 *
 * @return: system page size in bytes (default is 4096)
 */
size_t get_page_size(void);

/*
 * Register a GPU buffer: with Data Direct if enabled, otherwise through dmabuf, falling back to the
 * legacy nvidia-peermem method
 *
 * @resources [in]: DOCA Verbs resources
 * @buf [in]: GPU buffer
 * @size [in]: Buffer size
 * @access_flags [in]: Access flags of the memory region
 * @mr [out]: Memory region
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t verbs_reg_gpu_mr(struct verbs_resources *resources, void *buf, size_t size,
                              int access_flags, struct ibv_mr **mr);

/*
 * Copy to host memory a GPU buffer the NIC wrote into, once the NIC writes into GPU memory have
 * been flushed
 *
 * @host_buf [out]: Host buffer of at least size bytes
 * @gpu_buf [in]: GPU buffer
 * @size [in]: Number of bytes to copy
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t verbs_copy_gpu_to_host(void *host_buf, const void *gpu_buf, size_t size);

#if __cplusplus
extern "C" {
#endif

/*
 * Launch a CUDA kernel with to measure RDMA Write bandwidth
 *
 * @stream [in]: CUDA Stream to launch the kernel
 * @qp [in]: Verbs GPU object
 * @num_iters [in]: Total number of iterations
 * @cuda_blocks [in]: Number of CUDA blocks to launch the kernel. Must be 1, DOCA_ERROR_BAD_STATE
 * otherwise
 * @cuda_threads [in]: Number of CUDA threads to launch the kernel
 * @data_size [in]: Data buffer size (number of bytes)
 * @src_buf [in]: Source GPU data buffer address
 * @src_buf_mkey [in]: Source GPU data buffer memory key
 * @dst_buf [in]: Destination GPU data buffer address
 * @dst_buf_mkey [in]: Destination GPU data buffer memory key
 * @nic_handler [in]: Type of NIC handler of the QP
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpunetio_verbs_write_bw(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                     uint32_t num_iters, uint32_t cuda_blocks,
                                     uint32_t cuda_threads, uint32_t data_size, uint8_t *src_buf,
                                     uint32_t src_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey,
                                     enum doca_gpu_dev_verbs_nic_handler nic_handler);

/*
 * Launch a CUDA kernel with to measure RDMA Write latency
 *
 * @stream [in]: CUDA Stream to launch the kernel
 * @qp [in]: Verbs GPU object
 * @start_iters [in]: Number of iterations already run on the same buffers by previous launches
 * (e.g. warmup)
 * @num_iters [in]: Total number of iterations
 * @cuda_blocks [in]: Number of CUDA blocks to launch the kernel. Must be 1, DOCA_ERROR_BAD_STATE
 * otherwise
 * @cuda_threads [in]: Number of CUDA threads to launch the kernel. Must be 1,
 * DOCA_ERROR_BAD_STATE otherwise
 * @data_size [in]: Data buffer size (number of bytes)
 * @local_poll_buf [in]: Flag address to poll waiting for remote peer notification
 * @local_poll_mkey [in]: Flag mkey to poll waiting for remote peer notification
 * @local_post_buf [in]: Flag address to post notification to remote peer
 * @local_post_mkey [in]: Flag mkey to post notification to remote peer
 * @dst_buf [in]: Destination GPU data buffer address
 * @dst_buf_mkey [in]: Destination GPU data buffer memory key
 * @nic_handler [in]: Type of NIC handler
 * @is_client [in]: This kernel should act like the client (true) or server (false)
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpunetio_verbs_write_lat(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp, uint32_t start_iters, uint32_t num_iters,
    uint32_t cuda_blocks, uint32_t cuda_threads, uint32_t data_size, uint8_t *local_poll_buf,
    uint32_t local_poll_mkey, uint8_t *local_post_buf, uint32_t local_post_mkey, uint8_t *dst_buf,
    uint32_t dst_buf_mkey, enum doca_gpu_dev_verbs_nic_handler nic_handler, bool is_client);

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
 * @scope [in]: Each put is called per CUDA thread or per CUDA warp; any other scope returns
 * DOCA_ERROR_INVALID_VALUE
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpunetio_verbs_put_bw(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                   uint32_t num_iters, uint32_t cuda_blocks, uint32_t cuda_threads,
                                   uint32_t data_size, uint8_t *src_buf, uint32_t src_buf_mkey,
                                   uint8_t *dst_buf, uint32_t dst_buf_mkey,
                                   enum doca_gpu_dev_verbs_exec_scope scope);

/*
 * Launch a CUDA kernel to measure One-Sided Get Shared QP bandwidth
 *
 * @stream [in]: CUDA Stream to launch the kernel
 * @qp [in]: Verbs GPU object
 * @num_iters [in]: Total number of iterations
 * @cuda_blocks [in]: Number of CUDA blocks to launch the kernel
 * @cuda_threads [in]: Number of CUDA threads to launch the kernel
 * @data_size [in]: Data buffer size (number of bytes)
 * @local_buf [in]: Local GPU data buffer address, where the data is read into
 * @local_buf_mkey [in]: Local GPU data buffer memory key
 * @remote_buf [in]: Remote data buffer address, where the data is read from
 * @remote_buf_mkey [in]: Remote data buffer memory key
 * @dump_flag [in]: GPU flag the Dump WQE reads, before Hopper
 * @dump_flag_mkey [in]: GPU flag memory key
 * @scope [in]: Each get is called per CUDA thread or per CUDA warp; any other scope returns
 * DOCA_ERROR_INVALID_VALUE
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpunetio_verbs_get_bw(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                   uint32_t num_iters, uint32_t cuda_blocks, uint32_t cuda_threads,
                                   uint32_t data_size, uint8_t *local_buf, uint32_t local_buf_mkey,
                                   uint8_t *remote_buf, uint32_t remote_buf_mkey,
                                   uint64_t *dump_flag, uint32_t dump_flag_mkey,
                                   enum doca_gpu_dev_verbs_exec_scope scope);

/*
 * Launch a CUDA kernel to measure One-Sided Put with Signal Shared QP bandwidth. Each CUDA thread
 * (lane 0 for its warp in warp scope) increments its own remote signal.
 *
 * @stream [in]: CUDA Stream to launch the kernel
 * @qp [in]: Verbs GPU object
 * @num_iters [in]: Total number of iterations
 * @cuda_blocks [in]: Number of CUDA blocks to launch the kernel
 * @cuda_threads [in]: Number of CUDA threads to launch the kernel
 * @data_size [in]: Data buffer size (number of bytes)
 * @src_buf [in]: Source GPU data buffer address
 * @src_buf_mkey [in]: Source GPU data buffer memory key
 * @prev_flag_buf [in]: GPU buffer of the signal values before each fetch and add, one per thread
 * @prev_flag_buf_mkey [in]: GPU buffer memory key
 * @dst_buf [in]: Destination data buffer address
 * @dst_buf_mkey [in]: Destination data buffer memory key
 * @dst_flag [in]: Destination signals address, one per thread
 * @dst_flag_mkey [in]: Destination signals memory key
 * @scope [in]: Each put with signal is called per CUDA thread or per CUDA warp; any other scope
 * returns DOCA_ERROR_INVALID_VALUE
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpunetio_verbs_put_signal_bw(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t cuda_blocks,
    uint32_t cuda_threads, uint32_t data_size, uint8_t *src_buf, uint32_t src_buf_mkey,
    uint64_t *prev_flag_buf, uint32_t prev_flag_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey,
    uint64_t *dst_flag, uint32_t dst_flag_mkey, enum doca_gpu_dev_verbs_exec_scope scope);

/*
 * Launch a CUDA kernel to measure One-Sided Put with Signal latency
 *
 * @stream [in]: CUDA Stream to launch the kernel
 * @qp [in]: Verbs GPU object
 * @start_iters [in]: Number of iterations already run on the same signals by previous launches
 * (e.g. warmup)
 * @num_iters [in]: Total number of iterations
 * @cuda_blocks [in]: Number of CUDA blocks to launch the kernel. Must be 1, DOCA_ERROR_BAD_STATE
 * otherwise
 * @cuda_threads [in]: Number of CUDA threads to launch the kernel. Must be 1,
 * DOCA_ERROR_BAD_STATE otherwise
 * @data_size [in]: Data buffer size (number of bytes)
 * @src_buf [in]: Source GPU data buffer address
 * @src_buf_mkey [in]: Source GPU data buffer memory key
 * @local_flag [in]: GPU signal the remote peer increments, to poll
 * @prev_flag_buf [in]: GPU buffer of the remote signal value before each fetch and add
 * @prev_flag_buf_mkey [in]: GPU buffer memory key
 * @dst_buf [in]: Destination GPU data buffer address
 * @dst_buf_mkey [in]: Destination GPU data buffer memory key
 * @dst_flag [in]: Remote signal address
 * @dst_flag_mkey [in]: Remote signal memory key
 * @is_client [in]: This kernel should act like the client (true) or server (false)
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpunetio_verbs_put_signal_lat(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp, uint32_t start_iters, uint32_t num_iters,
    uint32_t cuda_blocks, uint32_t cuda_threads, uint32_t data_size, uint8_t *src_buf,
    uint32_t src_buf_mkey, uint64_t *local_flag, uint64_t *prev_flag_buf,
    uint32_t prev_flag_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey, uint64_t *dst_flag,
    uint32_t dst_flag_mkey, bool is_client);

#if __cplusplus
}
#endif

#endif /* GPUNETIO_VERBS_COMMON_H_ */
