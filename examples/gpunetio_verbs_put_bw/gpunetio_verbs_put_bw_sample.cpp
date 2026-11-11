/*
 * Copyright (c) 2025 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "verbs_common.h"
#include <signal.h>

#define SAMPLE_NAME "gpunetio_verbs_put_bw"

#define RESULT_LINE \
    "------------------------------------------------------------------------------------\n"
#define RESULT_FMT_G \
    " #bytes     #iterations    BW average[Gbps]   MsgRate[Mpps]    CUDA Kernel[ms]"
#define REPORT_FMT_EXT " %-7u    	%-7u           %-7.6lf            %-7.6lf            %-7.6f"

cudaStream_t cstream = NULL;
int message_size[NUM_MSG_SIZE] = {1,    64,    128,   256,   512,    1024,   2048,   4096,
                                  8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
volatile bool server_force_quit = false;

/*
 * Server validates data from client at the end of the test
 */
static doca_error_t server_validate_test(struct verbs_resources *resources) {
    const size_t max_size = (size_t)resources->cuda_threads * message_size[NUM_MSG_SIZE - 1];
    doca_error_t status = DOCA_SUCCESS;
    uint8_t *host_copy = NULL;
    const uint8_t *data;

    /* With Data Direct the buffers are in GPU memory: validate a host copy of each */
    if (resources->cfg->enable_data_direct) {
        host_copy = (uint8_t *)malloc(max_size);
        if (host_copy == NULL) {
            DOCA_LOG(LOG_ERR, "Failed to allocate a host buffer of size = %zu for the validation",
                     max_size);
            return DOCA_ERROR_NO_MEMORY;
        }
    }

    for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
        data = resources->data_buf[idx];
        if (host_copy != NULL) {
            status = verbs_copy_gpu_to_host(host_copy, data,
                                            (size_t)resources->cuda_threads * message_size[idx]);
            if (status != DOCA_SUCCESS) goto exit;
            data = host_copy;
        }

        for (int pos = 0; pos < (int)resources->cuda_threads * message_size[idx]; pos++) {
            if (data[pos] != (idx + 1)) {
                DOCA_LOG(LOG_ERR, "Validation error: buffer %d pos %d has invalid data %d", idx,
                         pos, data[pos]);
                status = DOCA_ERROR_BAD_STATE;
                goto exit;
            }
        }
    }

    DOCA_LOG(LOG_WARNING, "Validation successful! Data received correctly from client");

exit:
    free(host_copy);
    return status;
}

/*
 * Signal handler to quit application gracefully
 *
 * @signum [in]: signal received
 */
static void signal_handler(int signum) {
    /* Only async-signal-safe operations here: the main loop reports the signal */
    if (signum == SIGINT || signum == SIGTERM) server_force_quit = true;
}

static doca_error_t destroy_local_memory_objects(struct verbs_resources *resources) {
    const bool gpu_buffers = !resources->cfg->is_server || resources->cfg->enable_data_direct;
    doca_error_t result = DOCA_SUCCESS;
    int ret = 0;

    /* Clear each pointer so that the error path of create_local_memory_object() and the caller
     * can both call it */
    for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
        if (resources->data_mr[idx]) {
            ret = ibv_dereg_mr(resources->data_mr[idx]);
            if (ret != 0) {
                DOCA_LOG(LOG_ERR, "ibv_dereg_mr failed with error=%d", ret);
                if (result == DOCA_SUCCESS) result = DOCA_ERROR_DRIVER;
            }
            resources->data_mr[idx] = NULL;
        }

        if (resources->data_buf[idx]) {
            if (!gpu_buffers)
                free(resources->data_buf[idx]);
            else
                doca_gpu_mem_free(resources->gpu_dev, (void *)resources->data_buf[idx]);
            resources->data_buf[idx] = NULL;
        }
    }

    return result;
}

static doca_error_t create_local_memory_object(struct verbs_resources *resources) {
    const bool data_direct = resources->cfg->enable_data_direct;
    /* Data Direct registers GPU memory only, so with it the server buffers are in GPU memory too */
    const bool gpu_buffers = !resources->cfg->is_server || data_direct;
    const int access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_RELAXED_ORDERING;
    doca_error_t status = DOCA_SUCCESS;
    size_t host_page_size = get_page_size();
    size_t size;
    cudaError_t cuda_ret;

    for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
        resources->data_mr[idx] = NULL;
        size = (size_t)(resources->cuda_threads * message_size[idx]);
        ALIGN_SIZE(size, host_page_size);

        if (!gpu_buffers) {
            resources->data_buf[idx] = (uint8_t *)calloc(size, 1);
            if (resources->data_buf[idx] == NULL) {
                status = DOCA_ERROR_NO_MEMORY;
                DOCA_LOG(LOG_ERR, "Failed to allocate CPU memory buffer of size = %zd", size);
                goto exit_error;
            }
            memset(resources->data_buf[idx], 0, size);

            resources->data_mr[idx] =
                ibv_reg_mr(resources->verbs_pd, resources->data_buf[idx], size, access_flags);
            if (resources->data_mr[idx] == NULL) {
                status = DOCA_ERROR_DRIVER;
                DOCA_LOG(LOG_ERR, "Failed to create data mr: %s", strerror(errno));
                goto exit_error;
            }
        } else {
            status =
                doca_gpu_mem_alloc(resources->gpu_dev, size, host_page_size, DOCA_GPU_MEM_TYPE_GPU,
                                   (void **)&(resources->data_buf[idx]), NULL);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to allocate GPU memory buffer %d of size = %zd (%d x %d)",
                         idx, size, message_size[idx], resources->cuda_threads);
                goto exit_error;
            }

            cuda_ret =
                cudaMemset(resources->data_buf[idx], resources->cfg->is_server ? 0 : idx + 1, size);
            if (cuda_ret != cudaSuccess) {
                status = DOCA_ERROR_DRIVER;
                DOCA_LOG(LOG_ERR, "Failed to initialize GPU memory buffer %d: %s", idx,
                         cudaGetErrorString(cuda_ret));
                goto exit_error;
            }

            status = verbs_reg_gpu_mr(resources, resources->data_buf[idx], size, access_flags,
                                      &resources->data_mr[idx]);
            if (status != DOCA_SUCCESS) goto exit_error;
        }
    }

    /* cudaMemset() is asynchronous: the buffers must be initialized before the NIC accesses them */
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) {
        status = DOCA_ERROR_DRIVER;
        DOCA_LOG(LOG_ERR, "Failed to initialize the GPU memory buffers: %s",
                 cudaGetErrorString(cuda_ret));
        goto exit_error;
    }

    return DOCA_SUCCESS;

exit_error:
    destroy_local_memory_objects(resources);
    return status;
}

static doca_error_t exchange_params_with_remote_peer(struct verbs_resources *resources) {
    if (resources->cfg->is_server) {
        // Server sends local info
        for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
            uint64_t local_addr = (uint64_t)resources->data_buf[idx];
            if (verbs_oob_send(resources->conn_socket, &local_addr, sizeof(uint64_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local buffer address");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (verbs_oob_send(resources->conn_socket, &resources->data_mr[idx]->rkey,
                               sizeof(uint32_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local MKEY");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }
    } else {
        // Client waits for server info
        for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
            if (verbs_oob_recv(resources->conn_socket, &resources->remote_data_buf[idx],
                               sizeof(uint64_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote buffer address ");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (verbs_oob_recv(resources->conn_socket, &resources->remote_data_mkey[idx],
                               sizeof(uint32_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote MKEY, err = %d", errno);
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }
    }

    if (verbs_oob_send(resources->conn_socket, &resources->local_qp_number, sizeof(uint32_t)) !=
        0) {
        DOCA_LOG(LOG_ERR, "Failed to send local QP number");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (verbs_oob_recv(resources->conn_socket, &resources->remote_qp_number, sizeof(uint32_t)) !=
        0) {
        DOCA_LOG(LOG_ERR, "Failed to receive remote QP number, err = %d", errno);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (verbs_oob_send(resources->conn_socket, &resources->gid.raw, sizeof(resources->gid.raw)) !=
        0) {
        DOCA_LOG(LOG_ERR, "Failed to send local GID address");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (verbs_oob_recv(resources->conn_socket, &resources->remote_gid.raw,
                       sizeof(resources->gid.raw)) != 0) {
        DOCA_LOG(LOG_ERR, "Failed to receive remote GID address, err = %d", errno);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (verbs_oob_send(resources->conn_socket, &resources->lid, sizeof(uint32_t)) != 0) {
        DOCA_LOG(LOG_ERR, "Failed to send local LID");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (verbs_oob_recv(resources->conn_socket, &resources->dlid, sizeof(uint32_t)) != 0) {
        DOCA_LOG(LOG_ERR, "Failed to receive remote LID, err = %d", errno);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    return DOCA_SUCCESS;
}

doca_error_t verbs_server(struct verbs_config *cfg) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct verbs_resources resources = {0};
    int server_sock_fd = -1;

    resources.conn_socket = -1;
    resources.num_iters = cfg->num_iters;
    resources.cuda_threads = cfg->cuda_threads;
    resources.enable_umem_cpu = false;

    status = create_verbs_resources(cfg, &resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create local rdma resources: %d", status);
        return status;
    }

    status = create_local_memory_object(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create local memory resources: %d", status);
        goto server_cleanup;
    }

    status = oob_verbs_connection_server_setup(&server_sock_fd, &resources.conn_socket);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to setup OOB connection with remote peer: %d", status);
        goto server_cleanup;
    }

    status = verbs_oob_check_config(resources.conn_socket, SAMPLE_NAME, cfg);
    if (status != DOCA_SUCCESS) goto server_cleanup;

    status = exchange_params_with_remote_peer(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to exchange params with remote peer: %d", status);
        goto server_cleanup;
    }

    status = connect_verbs_qp(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to connect DOCA verbs QP: %d", status);
        goto server_cleanup;
    }

    /* Gracefully terminate app if ctrlc */
    server_force_quit = false;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /*
     * The client reports when all its runs have completed: closing the connection isn't enough, as
     * a client that fails after a warmup closes it too, with the expected data already in place
     */
    DOCA_LOG(LOG_INFO, "Waiting for the client to complete the benchmark");
    status = verbs_oob_wait_done(resources.conn_socket, &server_force_quit);
    if (status != DOCA_SUCCESS) goto server_cleanup;

    status = server_validate_test(&resources);
    if (status != DOCA_SUCCESS) DOCA_LOG(LOG_ERR, "Server data validation failed");

server_cleanup:
    tmp_status = destroy_local_memory_objects(&resources);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy local memory resources: %d", tmp_status);
        if (status == DOCA_SUCCESS) status = tmp_status;
    }

    tmp_status = destroy_verbs_resources(&resources);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy local rdma resources: %d", tmp_status);
        if (status == DOCA_SUCCESS) status = tmp_status;
    }

    oob_verbs_connection_server_close(server_sock_fd, resources.conn_socket);

    return status;
}

doca_error_t verbs_client(struct verbs_config *cfg) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct verbs_resources resources = {0};
    cudaError_t cuda_ret;
    CUresult cu_result;
    CUevent e_start = NULL, e_end = NULL;
    float et_ms = 0.0f;
    const unsigned long num_messages = cfg->num_iters * NUM_QP;
    pthread_t thread_id;
    struct cpu_proxy_args args = {};
    bool cpu_proxy_thread = false;
    struct doca_gpu_dev_verbs_qp *qp_gpu;
    int ret;

    resources.conn_socket = -1;
    resources.num_iters = cfg->num_iters;
    resources.cuda_threads = cfg->cuda_threads;
    resources.nic_handler = cfg->nic_handler;
    resources.scope = (enum doca_gpu_dev_verbs_exec_scope)cfg->exec_scope;
    resources.enable_umem_cpu = false;

    status = create_verbs_resources(cfg, &resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create DOCA verbs resources: %d", status);
        return status;
    }

    status = create_local_memory_object(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create local memory resources: %d", status);
        goto client_cleanup;
    }

    status = oob_verbs_connection_client_setup(cfg->server_ip_addr.c_str(), &resources.conn_socket);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to setup OOB connection with remote peer: %d", status);
        goto client_cleanup;
    }

    status = verbs_oob_check_config(resources.conn_socket, SAMPLE_NAME, cfg);
    if (status != DOCA_SUCCESS) goto client_cleanup;

    status = exchange_params_with_remote_peer(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to exchange params with remote peer: %d", status);
        goto client_cleanup;
    }

    status = connect_verbs_qp(&resources);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to connect DOCA verbs QP: %d", status);
        goto client_cleanup;
    }

    cuda_ret = cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking);
    if (cuda_ret != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Function cudaStreamCreateWithFlags error %d", cuda_ret);
        status = DOCA_ERROR_DRIVER;
        goto client_cleanup;
    }

    cu_result = cuEventCreate(&e_start, CU_EVENT_BLOCKING_SYNC);
    if (cu_result) {
        DOCA_LOG(LOG_ERR, "Function cuEventCreate for e_start %d", cu_result);
        status = DOCA_ERROR_DRIVER;
        goto destroy_events;
    }

    cu_result = cuEventCreate(&e_end, CU_EVENT_BLOCKING_SYNC);
    if (cu_result) {
        DOCA_LOG(LOG_ERR, "Function cuEventCreate for e_end %d", cu_result);
        status = DOCA_ERROR_DRIVER;
        goto destroy_events;
    }

    if (resources.qp == NULL) {
        DOCA_LOG(LOG_ERR, "resources.qp is NULL");
        status = DOCA_ERROR_DRIVER;
        goto destroy_events;
    }

    DOCA_LOG(LOG_INFO,
             "Launching gpunetio_verbs_put_bw kernel with %d CUDA Blocks, %d CUDA threads each, %d "
             "total number of iterations, %d iterations per cuda thread, %d nic handler, %s scope",
             VERBS_CUDA_BLOCK, resources.cuda_threads / VERBS_CUDA_BLOCK, resources.num_iters,
             resources.num_iters / resources.cuda_threads, resources.nic_handler,
             (resources.scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD) ? "THREAD" : "WARP");

    status = doca_gpu_verbs_get_qp_dev(resources.qp->qp_gverbs, &qp_gpu);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "doca_gpu_verbs_get_qp_dev failed: %d", status);
        goto destroy_events;
    }

    /* A CPU thread rings the NIC doorbell for the GPU: the CPU proxy NIC handler was requested, or
     * AUTO resolved to it */
    if (resources.qp->qp_gverbs->cpu_proxy) {
        args.qp_cpu = resources.qp->qp_gverbs;
        args.qp_cpu_companion = NULL;
        args.exit_flag = (uint64_t *)calloc(1, sizeof(uint64_t));
        if (args.exit_flag == NULL) {
            DOCA_LOG(LOG_ERR, "Failed to allocate CPU proxy exit flag");
            status = DOCA_ERROR_NO_MEMORY;
            goto destroy_events;
        }

        ret = pthread_create(&thread_id, NULL, progress_cpu_proxy, (void *)&args);
        if (ret != 0) {
            DOCA_LOG(LOG_ERR, "Failed to create CPU proxy thread: %s", strerror(ret));
            free(args.exit_flag);
            status = DOCA_ERROR_OPERATING_SYSTEM;
            goto destroy_events;
        }
        cpu_proxy_thread = true;
    }

    printf(RESULT_LINE);
    printf(RESULT_FMT_G);
    printf("\n");
    printf(RESULT_LINE);

    for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
        /* Warmup per size*/
        status = gpunetio_verbs_put_bw(
            cstream, qp_gpu, resources.num_iters, VERBS_CUDA_BLOCK,
            resources.cuda_threads / VERBS_CUDA_BLOCK, message_size[idx], resources.data_buf[idx],
            htobe32(resources.data_mr[idx]->lkey), (uint8_t *)(resources.remote_data_buf[idx]),
            htobe32(resources.remote_data_mkey[idx]), resources.scope);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function gpunetio_verbs_put_bw failed: %d", status);
            goto stop_thread;
        }

        cuda_ret = cudaStreamSynchronize(cstream);
        if (cuda_ret != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Warmup kernel failed: %s", cudaGetErrorString(cuda_ret));
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        cu_result = cuEventRecord(e_start, cstream);
        if (cu_result) {
            DOCA_LOG(LOG_ERR, "Error in cuEventRecord for e_start");
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        status = gpunetio_verbs_put_bw(
            cstream, qp_gpu, resources.num_iters, VERBS_CUDA_BLOCK,
            resources.cuda_threads / VERBS_CUDA_BLOCK, message_size[idx], resources.data_buf[idx],
            htobe32(resources.data_mr[idx]->lkey), (uint8_t *)(resources.remote_data_buf[idx]),
            htobe32(resources.remote_data_mkey[idx]), resources.scope);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function gpunetio_verbs_put_bw failed: %d", status);
            goto stop_thread;
        }

        cu_result = cuEventRecord(e_end, cstream);
        if (cu_result) {
            DOCA_LOG(LOG_ERR, "Error in cuEventRecord for e_end");
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        cu_result = cuEventSynchronize(e_end);
        if (cu_result) {
            DOCA_LOG(LOG_ERR, "Error in cuEventSynchronize for e_end");
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        cu_result = cuEventElapsedTime(&et_ms, e_start, e_end);
        if (cu_result) {
            DOCA_LOG(LOG_ERR, "Error in cuEventElapsedTime");
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        double bw = (double)((double)((message_size[idx] * num_messages) / et_ms * 1000.0f) *
                             ((double)8.0) / BW_FORMAT_FACTOR);
        double msgrate = (double)(num_messages / et_ms * 1000.0f / 1000000.0f);

        printf(REPORT_FMT_EXT, message_size[idx], resources.num_iters, bw, msgrate, (double)et_ms);

        printf("\n");
    }

    /* Every run has completed: the server can validate the data */
    status = verbs_oob_send_done(resources.conn_socket);

stop_thread:
    if (cpu_proxy_thread) {
        WRITE_ONCE(*args.exit_flag, 1);
        pthread_join(thread_id, NULL);
        free(args.exit_flag);
    }

destroy_events:
    if (cstream != NULL) cudaStreamDestroy(cstream);
    if (e_start != NULL) cuEventDestroy(e_start);
    if (e_end != NULL) cuEventDestroy(e_end);

client_cleanup:
    oob_verbs_connection_client_close(resources.conn_socket);

    tmp_status = destroy_local_memory_objects(&resources);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy local memory resources: %d", tmp_status);
        if (status == DOCA_SUCCESS) status = tmp_status;
    }

    tmp_status = destroy_verbs_resources(&resources);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy local rdma resources: %d", tmp_status);
        if (status == DOCA_SUCCESS) status = tmp_status;
    }

    return status;
}
