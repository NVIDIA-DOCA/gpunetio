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

#include "verbs_common.h"

#define RESULT_LINE                                                                                \
    "--------------------------------------------------------------------------------------------" \
    "--\n"
#define RESULT_FMT_LAT \
    " #bytes 	#iterations    Unidirectional[usec]    Bidirectional[usec]    CUDA Kernel[usec]"
#define REPORT_FMT_LAT \
    " %-7lu 		%-7d          %-7.2f       		%-7.2f    	 	%-7.2f"

cudaStream_t cstream;
int message_size[NUM_MSG_SIZE] = {1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

static doca_error_t destroy_local_memory_objects(struct verbs_resources *resources) {
    int ret = 0;

    for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
        if (resources->local_poll_mr[idx]) {
            ret = ibv_dereg_mr(resources->local_poll_mr[idx]);
            if (ret != 0) {
                DOCA_LOG(LOG_ERR, "ibv_dereg_mr failed with error=%d", ret);
                return DOCA_ERROR_DRIVER;
            }
        }

        if (resources->local_post_mr[idx]) {
            ret = ibv_dereg_mr(resources->local_post_mr[idx]);
            if (ret != 0) {
                DOCA_LOG(LOG_ERR, "ibv_dereg_mr failed with error=%d", ret);
                return DOCA_ERROR_DRIVER;
            }
        }

        if (resources->local_poll_buf[idx])
            doca_gpu_mem_free(resources->gpu_dev, (void *)resources->local_poll_buf[idx]);

        if (resources->local_post_buf[idx])
            doca_gpu_mem_free(resources->gpu_dev, (void *)resources->local_post_buf[idx]);
    }

    return DOCA_SUCCESS;
}

static doca_error_t create_local_memory_object(struct verbs_resources *resources) {
    doca_error_t status = DOCA_SUCCESS;
    size_t host_page_size = get_page_size();
    size_t size_data;
    int dmabuf_fd;

    for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
        resources->data_mr[idx] = NULL;
        size_data = (size_t)(resources->cuda_threads * message_size[idx]);
        ALIGN_SIZE(size_data, host_page_size);

        status =
            doca_gpu_mem_alloc(resources->gpu_dev, size_data, host_page_size, DOCA_GPU_MEM_TYPE_GPU,
                               (void **)&(resources->local_poll_buf[idx]), NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to allocate GPU memory buffer %d of size = %zd (%d x %d)",
                     idx, size_data, message_size[idx], resources->cuda_threads);
            goto exit_error;
        }

        cudaMemset(resources->local_poll_buf[idx], 0, size_data);

        /* Try with dmabuf mapping first. If it doesn't work, fallback to legacy nvidia-peermem
         * method. */
        status = doca_gpu_dmabuf_fd(resources->gpu_dev, resources->local_poll_buf[idx], size_data,
                                    &dmabuf_fd);
        if (status == DOCA_SUCCESS) {
            resources->local_poll_mr[idx] =
                ibv_reg_dmabuf_mr(resources->verbs_pd, 0, size_data,
                                  (uint64_t)resources->local_poll_buf[idx], dmabuf_fd,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        }

        if (resources->local_poll_mr[idx] == NULL) {
            resources->local_poll_mr[idx] =
                ibv_reg_mr(resources->verbs_pd, resources->local_poll_buf[idx], size_data,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
            if (resources->local_poll_mr[idx] == NULL) {
                DOCA_LOG(LOG_ERR, "Failed to create data mr: %d", status);
                goto exit_error;
            }
        }

        status =
            doca_gpu_mem_alloc(resources->gpu_dev, size_data, host_page_size, DOCA_GPU_MEM_TYPE_GPU,
                               (void **)&(resources->local_post_buf[idx]), NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to allocate GPU memory buffer %d of size = %zd (%d x %d)",
                     idx, size_data, message_size[idx], resources->cuda_threads);
            goto exit_error;
        }

        cudaMemset(resources->local_post_buf[idx], 0, size_data);

        /* Try with dmabuf mapping first. If it doesn't work, fallback to legacy nvidia-peermem
         * method. */
        status = doca_gpu_dmabuf_fd(resources->gpu_dev, resources->local_post_buf[idx], size_data,
                                    &dmabuf_fd);
        if (status == DOCA_SUCCESS) {
            resources->local_post_mr[idx] =
                ibv_reg_dmabuf_mr(resources->verbs_pd, 0, size_data,
                                  (uint64_t)resources->local_post_buf[idx], dmabuf_fd,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        }

        if (resources->local_post_mr[idx] == NULL) {
            resources->local_post_mr[idx] =
                ibv_reg_mr(resources->verbs_pd, resources->local_post_buf[idx], size_data,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
            if (resources->local_post_mr[idx] == NULL) {
                DOCA_LOG(LOG_ERR, "Failed to create data mr: %d", status);
                goto exit_error;
            }
        }
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
            uint64_t local_addr = (uint64_t)resources->local_poll_buf[idx];
            if (send(resources->conn_socket, &local_addr, sizeof(uint64_t), 0) < 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local buffer address");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (send(resources->conn_socket, &resources->local_poll_mr[idx]->rkey, sizeof(uint32_t),
                     0) < 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local MKEY");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }

        for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
            if (recv(resources->conn_socket, &resources->remote_data_buf[idx], sizeof(uint64_t),
                     0) < 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote buffer address ");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (recv(resources->conn_socket, &resources->remote_data_mkey[idx], sizeof(uint32_t),
                     0) < 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote MKEY, err = %d", errno);
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }

    } else {
        // Client waits for server info
        for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
            if (recv(resources->conn_socket, &resources->remote_data_buf[idx], sizeof(uint64_t),
                     0) < 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote buffer address ");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (recv(resources->conn_socket, &resources->remote_data_mkey[idx], sizeof(uint32_t),
                     0) < 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote MKEY, err = %d", errno);
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }

        for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
            uint64_t local_addr = (uint64_t)resources->local_poll_buf[idx];
            if (send(resources->conn_socket, &local_addr, sizeof(uint64_t), 0) < 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local buffer address");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (send(resources->conn_socket, &resources->local_poll_mr[idx]->rkey, sizeof(uint32_t),
                     0) < 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local MKEY");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }
    }

    if (send(resources->conn_socket, &resources->local_qp_number, sizeof(uint32_t), 0) < 0) {
        DOCA_LOG(LOG_ERR, "Failed to send local QP number");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (recv(resources->conn_socket, &resources->remote_qp_number, sizeof(uint32_t), 0) < 0) {
        DOCA_LOG(LOG_ERR, "Failed to receive remote QP number, err = %d", errno);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (send(resources->conn_socket, &resources->gid.raw, sizeof(resources->gid.raw), 0) < 0) {
        DOCA_LOG(LOG_ERR, "Failed to send local GID address");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (recv(resources->conn_socket, &resources->remote_gid.raw, sizeof(resources->gid.raw), 0) <
        0) {
        DOCA_LOG(LOG_ERR, "Failed to receive remote GID address, err = %d", errno);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (send(resources->conn_socket, &resources->lid, sizeof(uint32_t), 0) < 0) {
        DOCA_LOG(LOG_ERR, "Failed to send local GID address");
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    if (recv(resources->conn_socket, &resources->dlid, sizeof(uint32_t), 0) < 0) {
        DOCA_LOG(LOG_ERR, "Failed to receive remote GID address, err = %d", errno);
        return DOCA_ERROR_CONNECTION_ABORTED;
    }

    return DOCA_SUCCESS;
}

doca_error_t verbs_server(struct verbs_config *cfg) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct verbs_resources resources = {0};
    int server_sock_fd = -1;
    cudaError_t cuda_ret;
    CUresult cu_result;
    CUevent e_start, e_end;
    float et_ms = 0.0f;
    pthread_t thread_id;
    struct cpu_proxy_args args;
    struct doca_gpu_dev_verbs_qp *qp_gpu;
    int ret;

    resources.conn_socket = -1;
    resources.num_iters = cfg->num_iters;
    resources.cuda_threads = cfg->cuda_threads;
    resources.nic_handler = cfg->nic_handler;

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

    ret = oob_verbs_connection_server_setup(&server_sock_fd, &resources.conn_socket);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to setup OOB connection with remote peer: %d", ret);
        goto server_cleanup;
    }

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

    cuda_ret = cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking);
    if (cuda_ret != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Function cudaStreamCreateWithFlags error %d", cuda_ret);
        status = DOCA_ERROR_DRIVER;
        goto destroy_events;
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
        cuEventDestroy(e_start);
        status = DOCA_ERROR_DRIVER;
        goto destroy_events;
    }

    DOCA_LOG(LOG_INFO,
             "Launching gpunetio_verbs_write_lat server kernel with %d CUDA Blocks, %d CUDA "
             "threads, %d total number of iterations, %d iterations per cuda thread, %d cpu proxy",
             NUM_QP, resources.cuda_threads, resources.num_iters,
             resources.num_iters / resources.cuda_threads,  // check this is ok
             resources.nic_handler);

    if (resources.nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        args.qp_cpu = resources.qp->qp_gverbs;
        args.exit_flag = (uint64_t *)calloc(1, sizeof(uint64_t));
        *(args.exit_flag) = 0;

        int result = pthread_create(&thread_id, NULL, progress_cpu_proxy, (void *)&args);
        if (result != 0) {
            perror("Failed to create thread");
            goto destroy_events;
        }
    }

    printf(RESULT_LINE);
    printf(RESULT_FMT_LAT);
    printf("\n");
    printf(RESULT_LINE);

    status = doca_gpu_verbs_get_qp_dev(resources.qp->qp_gverbs, &qp_gpu);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "doca_gpu_verbs_get_qp_dev failed");
        goto stop_thread;
    }

    for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
        /* Warmup per size*/
        status = gpunetio_verbs_write_lat(
            cstream, qp_gpu, resources.num_iters, NUM_QP, resources.cuda_threads, message_size[idx],
            resources.local_poll_buf[idx], htobe32(resources.local_poll_mr[idx]->lkey),
            resources.local_post_buf[idx], htobe32(resources.local_post_mr[idx]->lkey),
            (uint8_t *)(resources.remote_data_buf[idx]), htobe32(resources.remote_data_mkey[idx]),
            false);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function kernel_write_client failed: %d", status);
            goto stop_thread;
        }

        cu_result = cuEventRecord(e_start, cstream);
        if (cu_result) {
            DOCA_LOG(LOG_ERR, "Error in cuEventRecord for e_start");
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        status = gpunetio_verbs_write_lat(
            cstream, qp_gpu, resources.num_iters, NUM_QP, resources.cuda_threads, message_size[idx],
            resources.local_poll_buf[idx], htobe32(resources.local_poll_mr[idx]->lkey),
            resources.local_post_buf[idx], htobe32(resources.local_post_mr[idx]->lkey),
            (uint8_t *)(resources.remote_data_buf[idx]), htobe32(resources.remote_data_mkey[idx]),
            false);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function kernel_write_client failed: %d", status);
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

        double lat_half = ((double)(et_ms * 1000.0f)) / ((double)(resources.num_iters * 2.0f));
        double lat_full = ((double)(et_ms * 1000.0f)) / ((double)(resources.num_iters));

        printf(REPORT_FMT_LAT, (unsigned long)message_size[idx], resources.num_iters, lat_half,
               lat_full, ((double)(et_ms * 1000.0f)));

        printf("\n");
    }

stop_thread:
    if (resources.nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        *((volatile uint64_t *)args.exit_flag) = 1;
        pthread_join(thread_id, NULL);
    }

destroy_events:
    cuEventDestroy(e_start);
    cuEventDestroy(e_end);
    cudaStreamDestroy(cstream);

server_cleanup:
    oob_verbs_connection_client_close(resources.conn_socket);

    tmp_status = destroy_local_memory_objects(&resources);
    if (tmp_status != DOCA_SUCCESS)
        DOCA_LOG(LOG_ERR, "Failed to destroy local resources: %d", tmp_status);

    tmp_status = destroy_verbs_resources(&resources);
    if (tmp_status != DOCA_SUCCESS)
        DOCA_LOG(LOG_ERR, "Failed to destroy local memory resources: %d", tmp_status);

    return status;
}

doca_error_t verbs_client(struct verbs_config *cfg) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct verbs_resources resources = {0};
    cudaError_t cuda_ret;
    CUresult cu_result;
    CUevent e_start, e_end;
    float et_ms = 0.0f;
    pthread_t thread_id;
    struct cpu_proxy_args args;
    struct doca_gpu_dev_verbs_qp *qp_gpu;
    int ret;

    resources.conn_socket = -1;
    resources.num_iters = cfg->num_iters;
    resources.cuda_threads = cfg->cuda_threads;
    resources.nic_handler = cfg->nic_handler;

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

    ret = oob_verbs_connection_client_setup(cfg->server_ip_addr.c_str(), &resources.conn_socket);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to setup OOB connection with remote peer: %d", ret);
        goto client_cleanup;
    }

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
        goto destroy_events;
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
        cuEventDestroy(e_start);
        status = DOCA_ERROR_DRIVER;
        goto destroy_events;
    }

    DOCA_LOG(LOG_INFO,
             "Launching gpunetio_verbs_write_lat client kernel with %d CUDA Blocks, %d CUDA "
             "threads, %d total number of iterations, %d iterations per cuda thread, %d cpu proxy",
             NUM_QP, resources.cuda_threads, resources.num_iters,
             resources.num_iters / resources.cuda_threads,  // check this is ok
             resources.nic_handler);

    if (resources.nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        args.qp_cpu = resources.qp->qp_gverbs;
        args.exit_flag = (uint64_t *)calloc(1, sizeof(uint64_t));
        *(args.exit_flag) = 0;

        int result = pthread_create(&thread_id, NULL, progress_cpu_proxy, (void *)&args);
        if (result != 0) {
            perror("Failed to create thread");
            goto destroy_events;
        }
    }

    printf(RESULT_LINE);
    printf(RESULT_FMT_LAT);
    printf("\n");
    printf(RESULT_LINE);

    status = doca_gpu_verbs_get_qp_dev(resources.qp->qp_gverbs, &qp_gpu);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "doca_gpu_verbs_get_qp_dev failed");
        goto stop_thread;
    }

    for (int idx = 0; idx < NUM_MSG_SIZE; idx++) {
        /* Warmup per size*/
        status = gpunetio_verbs_write_lat(
            cstream, qp_gpu, resources.num_iters, NUM_QP, resources.cuda_threads, message_size[idx],
            resources.local_poll_buf[idx], htobe32(resources.local_poll_mr[idx]->lkey),
            resources.local_post_buf[idx], htobe32(resources.local_post_mr[idx]->lkey),
            (uint8_t *)(resources.remote_data_buf[idx]), htobe32(resources.remote_data_mkey[idx]),
            true);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function kernel_write_client failed: %d", status);
            goto stop_thread;
        }

        cu_result = cuEventRecord(e_start, cstream);
        if (cu_result) {
            DOCA_LOG(LOG_ERR, "Error in cuEventRecord for e_start");
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        status = gpunetio_verbs_write_lat(
            cstream, qp_gpu, resources.num_iters, NUM_QP, resources.cuda_threads, message_size[idx],
            resources.local_poll_buf[idx], htobe32(resources.local_poll_mr[idx]->lkey),
            resources.local_post_buf[idx], htobe32(resources.local_post_mr[idx]->lkey),
            (uint8_t *)(resources.remote_data_buf[idx]), htobe32(resources.remote_data_mkey[idx]),
            true);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function kernel_write_client failed: %d", status);
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

        double lat_half = ((double)(et_ms * 1000.0f)) / ((double)(resources.num_iters * 2.0f));
        double lat_full = ((double)(et_ms * 1000.0f)) / ((double)(resources.num_iters));

        printf(REPORT_FMT_LAT, (unsigned long)message_size[idx], resources.num_iters, lat_half,
               lat_full, ((double)(et_ms * 1000.0f)));

        printf("\n");
    }

stop_thread:
    if (resources.nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY) {
        *((volatile uint64_t *)args.exit_flag) = 1;
        pthread_join(thread_id, NULL);
    }

destroy_events:
    cuEventDestroy(e_start);
    cuEventDestroy(e_end);
    cudaStreamDestroy(cstream);

client_cleanup:
    oob_verbs_connection_client_close(resources.conn_socket);

    tmp_status = destroy_local_memory_objects(&resources);
    if (tmp_status != DOCA_SUCCESS)
        DOCA_LOG(LOG_ERR, "Failed to destroy local resources: %d", tmp_status);

    tmp_status = destroy_verbs_resources(&resources);
    if (tmp_status != DOCA_SUCCESS)
        DOCA_LOG(LOG_ERR, "Failed to destroy local memory resources: %d", tmp_status);

    return status;
}
