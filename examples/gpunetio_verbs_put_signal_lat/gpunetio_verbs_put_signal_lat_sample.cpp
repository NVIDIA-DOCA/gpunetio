/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define SAMPLE_NAME "gpunetio_verbs_put_signal_lat"

#define RESULT_LINE                                                                                \
    "--------------------------------------------------------------------------------------------" \
    "--\n"
#define RESULT_FMT_LAT \
    " #bytes 	#iterations    Unidirectional[usec]    Bidirectional[usec]    CUDA Kernel[usec]"
#define REPORT_FMT_LAT \
    " %-7lu 		%-7d          %-7.2f       		%-7.2f    	 	%-7.2f"

cudaStream_t cstream = NULL;
int message_size[NUM_MSG_SIZE_LAT] = {1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

static doca_error_t destroy_local_memory_objects(struct verbs_resources *resources) {
    doca_error_t result = DOCA_SUCCESS;
    int ret = 0;

    /* Clear each pointer so that the error path of create_local_memory_object() and the caller
     * can both call it */
    for (int idx = 0; idx < NUM_MSG_SIZE_LAT; idx++) {
        if (resources->data_mr[idx]) {
            ret = ibv_dereg_mr(resources->data_mr[idx]);
            if (ret != 0) {
                DOCA_LOG(LOG_ERR, "ibv_dereg_mr failed with error=%d", ret);
                if (result == DOCA_SUCCESS) result = DOCA_ERROR_DRIVER;
            }
            resources->data_mr[idx] = NULL;
        }

        if (resources->flag_mr[idx]) {
            ret = ibv_dereg_mr(resources->flag_mr[idx]);
            if (ret != 0) {
                DOCA_LOG(LOG_ERR, "ibv_dereg_mr failed with error=%d", ret);
                if (result == DOCA_SUCCESS) result = DOCA_ERROR_DRIVER;
            }
            resources->flag_mr[idx] = NULL;
        }

        if (resources->prev_flag_mr[idx]) {
            ret = ibv_dereg_mr(resources->prev_flag_mr[idx]);
            if (ret != 0) {
                DOCA_LOG(LOG_ERR, "ibv_dereg_mr failed with error=%d", ret);
                if (result == DOCA_SUCCESS) result = DOCA_ERROR_DRIVER;
            }
            resources->prev_flag_mr[idx] = NULL;
        }

        if (resources->data_buf[idx]) {
            doca_gpu_mem_free(resources->gpu_dev, (void *)resources->data_buf[idx]);
            resources->data_buf[idx] = NULL;
        }

        if (resources->flag_buf[idx]) {
            doca_gpu_mem_free(resources->gpu_dev, (void *)resources->flag_buf[idx]);
            resources->flag_buf[idx] = NULL;
        }

        if (resources->prev_flag_buf[idx]) {
            doca_gpu_mem_free(resources->gpu_dev, (void *)resources->prev_flag_buf[idx]);
            resources->prev_flag_buf[idx] = NULL;
        }
    }

    return result;
}

/*
 * Allocate a zeroed buffer of 8 bytes signals in GPU memory and register it
 *
 * @resources [in]: DOCA Verbs resources
 * @size [in]: Buffer size
 * @access_flags [in]: Access flags of the memory region
 * @flag_buf [out]: GPU buffer, set as soon as it is allocated so that the caller can free it
 * @flag_mr [out]: Memory region
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t create_gpu_flags(struct verbs_resources *resources, size_t size,
                                     int access_flags, uint64_t **flag_buf,
                                     struct ibv_mr **flag_mr) {
    doca_error_t status;
    cudaError_t cuda_ret;

    status = doca_gpu_mem_alloc(resources->gpu_dev, size, get_page_size(), DOCA_GPU_MEM_TYPE_GPU,
                                (void **)flag_buf, NULL);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to allocate GPU memory buffer of size = %zd", size);
        return status;
    }

    cuda_ret = cudaMemset(*flag_buf, 0, size);
    if (cuda_ret != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to initialize GPU memory buffer: %s",
                 cudaGetErrorString(cuda_ret));
        return DOCA_ERROR_DRIVER;
    }

    return verbs_reg_gpu_mr(resources, *flag_buf, size, access_flags, flag_mr);
}

static doca_error_t create_local_memory_object(struct verbs_resources *resources) {
    doca_error_t status = DOCA_SUCCESS;
    size_t host_page_size = get_page_size();
    size_t size_data, size_flag;
    cudaError_t cuda_ret;

    for (int idx = 0; idx < NUM_MSG_SIZE_LAT; idx++) {
        resources->data_mr[idx] = NULL;
        size_data = (size_t)(resources->cuda_threads * message_size[idx]);
        ALIGN_SIZE(size_data, host_page_size);
        size_flag = (size_t)resources->cuda_threads * sizeof(uint64_t);
        ALIGN_SIZE(size_flag, host_page_size);

        /* Source of the local puts and destination of the remote ones: both peers send the same
         * data */
        status =
            doca_gpu_mem_alloc(resources->gpu_dev, size_data, host_page_size, DOCA_GPU_MEM_TYPE_GPU,
                               (void **)&(resources->data_buf[idx]), NULL);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to allocate GPU memory buffer %d of size = %zd (%d x %d)",
                     idx, size_data, message_size[idx], resources->cuda_threads);
            goto exit_error;
        }

        cuda_ret = cudaMemset(resources->data_buf[idx], idx + 1, size_data);
        if (cuda_ret != cudaSuccess) {
            status = DOCA_ERROR_DRIVER;
            DOCA_LOG(LOG_ERR, "Failed to initialize GPU memory buffer %d: %s", idx,
                     cudaGetErrorString(cuda_ret));
            goto exit_error;
        }

        status = verbs_reg_gpu_mr(
            resources, resources->data_buf[idx], size_data,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_RELAXED_ORDERING,
            &resources->data_mr[idx]);
        if (status != DOCA_SUCCESS) goto exit_error;

        /* The signal the remote peer increments. No relaxed ordering: the NIC write of a signal
         * must not pass the data writes before it */
        status = create_gpu_flags(resources, size_flag,
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC,
                                  &resources->flag_buf[idx], &resources->flag_mr[idx]);
        if (status != DOCA_SUCCESS) goto exit_error;

        /* The remote signal value before each fetch and add */
        status = create_gpu_flags(resources, size_flag, IBV_ACCESS_LOCAL_WRITE,
                                  &resources->prev_flag_buf[idx], &resources->prev_flag_mr[idx]);
        if (status != DOCA_SUCCESS) goto exit_error;
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
        for (int idx = 0; idx < NUM_MSG_SIZE_LAT; idx++) {
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

            local_addr = (uint64_t)resources->flag_buf[idx];
            if (verbs_oob_send(resources->conn_socket, &local_addr, sizeof(uint64_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local flag address");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (verbs_oob_send(resources->conn_socket, &resources->flag_mr[idx]->rkey,
                               sizeof(uint32_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local flag MKEY");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }

        for (int idx = 0; idx < NUM_MSG_SIZE_LAT; idx++) {
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

            if (verbs_oob_recv(resources->conn_socket, &resources->remote_flag_buf[idx],
                               sizeof(uint64_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote flag address");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (verbs_oob_recv(resources->conn_socket, &resources->remote_flag_mkey[idx],
                               sizeof(uint32_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote flag MKEY, err = %d", errno);
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }

    } else {
        // Client waits for server info
        for (int idx = 0; idx < NUM_MSG_SIZE_LAT; idx++) {
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

            if (verbs_oob_recv(resources->conn_socket, &resources->remote_flag_buf[idx],
                               sizeof(uint64_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote flag address");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (verbs_oob_recv(resources->conn_socket, &resources->remote_flag_mkey[idx],
                               sizeof(uint32_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to receive remote flag MKEY, err = %d", errno);
                return DOCA_ERROR_CONNECTION_ABORTED;
            }
        }

        for (int idx = 0; idx < NUM_MSG_SIZE_LAT; idx++) {
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

            local_addr = (uint64_t)resources->flag_buf[idx];
            if (verbs_oob_send(resources->conn_socket, &local_addr, sizeof(uint64_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local flag address");
                return DOCA_ERROR_CONNECTION_ABORTED;
            }

            if (verbs_oob_send(resources->conn_socket, &resources->flag_mr[idx]->rkey,
                               sizeof(uint32_t)) != 0) {
                DOCA_LOG(LOG_ERR, "Failed to send local flag MKEY");
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
    cudaError_t cuda_ret;
    CUresult cu_result;
    CUevent e_start = NULL, e_end = NULL;
    float et_ms = 0.0f;
    pthread_t thread_id;
    struct cpu_proxy_args args = {};
    bool cpu_proxy_thread = false;
    struct doca_gpu_dev_verbs_qp *qp_gpu;
    int ret;

    resources.conn_socket = -1;
    resources.num_iters = cfg->num_iters;
    resources.cuda_threads = cfg->cuda_threads;
    resources.nic_handler = cfg->nic_handler;
    resources.qp_group = false;

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
        status = DOCA_ERROR_DRIVER;
        goto destroy_events;
    }

    DOCA_LOG(
        LOG_INFO,
        "Launching gpunetio_verbs_put_signal_lat server kernel with %d CUDA Blocks, %d CUDA "
        "threads, %d total number of iterations, %d iterations per cuda thread, %d NIC handler",
        NUM_QP, resources.cuda_threads, resources.num_iters,
        resources.num_iters / resources.cuda_threads, resources.nic_handler);

    status = doca_gpu_verbs_get_qp_dev(resources.qp->qp_gverbs, &qp_gpu);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "doca_gpu_verbs_get_qp_dev failed: %d", status);
        goto destroy_events;
    }

    /*
     * A CPU thread progresses the QP for the GPU: it rings the NIC doorbell with the CPU proxy NIC
     * handlers (requested, or resolved from AUTO), or updates the DBR with the software-emulated no
     * DBR mode. It starts after doca_gpu_verbs_get_qp_dev(), which sets up what the thread reads in
     * that mode.
     */
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
    printf(RESULT_FMT_LAT);
    printf("\n");
    printf(RESULT_LINE);

    for (int idx = 0; idx < NUM_MSG_SIZE_LAT; idx++) {
        /* Warmup per size*/
        status = gpunetio_verbs_put_signal_lat(
            cstream, qp_gpu, 0, resources.num_iters, NUM_QP, resources.cuda_threads,
            message_size[idx], resources.data_buf[idx], htobe32(resources.data_mr[idx]->lkey),
            resources.flag_buf[idx], resources.prev_flag_buf[idx],
            htobe32(resources.prev_flag_mr[idx]->lkey), (uint8_t *)(resources.remote_data_buf[idx]),
            htobe32(resources.remote_data_mkey[idx]), (uint64_t *)(resources.remote_flag_buf[idx]),
            htobe32(resources.remote_flag_mkey[idx]), false);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function gpunetio_verbs_put_signal_lat failed: %d", status);
            goto stop_thread;
        }

        cu_result = cuEventRecord(e_start, cstream);
        if (cu_result) {
            DOCA_LOG(LOG_ERR, "Error in cuEventRecord for e_start");
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        /* Same signals as the warmup: they keep counting from where it stopped */
        status = gpunetio_verbs_put_signal_lat(
            cstream, qp_gpu, resources.num_iters, resources.num_iters, NUM_QP,
            resources.cuda_threads, message_size[idx], resources.data_buf[idx],
            htobe32(resources.data_mr[idx]->lkey), resources.flag_buf[idx],
            resources.prev_flag_buf[idx], htobe32(resources.prev_flag_mr[idx]->lkey),
            (uint8_t *)(resources.remote_data_buf[idx]), htobe32(resources.remote_data_mkey[idx]),
            (uint64_t *)(resources.remote_flag_buf[idx]), htobe32(resources.remote_flag_mkey[idx]),
            false);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function gpunetio_verbs_put_signal_lat failed: %d", status);
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
    if (cpu_proxy_thread) {
        WRITE_ONCE(*args.exit_flag, 1);
        pthread_join(thread_id, NULL);
        free(args.exit_flag);
    }

destroy_events:
    if (e_start != NULL) cuEventDestroy(e_start);
    if (e_end != NULL) cuEventDestroy(e_end);
    if (cstream != NULL) cudaStreamDestroy(cstream);

server_cleanup:
    oob_verbs_connection_server_close(server_sock_fd, resources.conn_socket);

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

doca_error_t verbs_client(struct verbs_config *cfg) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct verbs_resources resources = {0};
    cudaError_t cuda_ret;
    CUresult cu_result;
    CUevent e_start = NULL, e_end = NULL;
    float et_ms = 0.0f;
    pthread_t thread_id;
    struct cpu_proxy_args args = {};
    bool cpu_proxy_thread = false;
    struct doca_gpu_dev_verbs_qp *qp_gpu;
    int ret;

    resources.conn_socket = -1;
    resources.num_iters = cfg->num_iters;
    resources.cuda_threads = cfg->cuda_threads;
    resources.nic_handler = cfg->nic_handler;
    resources.qp_group = false;

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
        status = DOCA_ERROR_DRIVER;
        goto destroy_events;
    }

    DOCA_LOG(
        LOG_INFO,
        "Launching gpunetio_verbs_put_signal_lat client kernel with %d CUDA Blocks, %d CUDA "
        "threads, %d total number of iterations, %d iterations per cuda thread, %d NIC handler",
        NUM_QP, resources.cuda_threads, resources.num_iters,
        resources.num_iters / resources.cuda_threads, resources.nic_handler);

    status = doca_gpu_verbs_get_qp_dev(resources.qp->qp_gverbs, &qp_gpu);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "doca_gpu_verbs_get_qp_dev failed: %d", status);
        goto destroy_events;
    }

    /*
     * A CPU thread progresses the QP for the GPU: it rings the NIC doorbell with the CPU proxy NIC
     * handlers (requested, or resolved from AUTO), or updates the DBR with the software-emulated no
     * DBR mode. It starts after doca_gpu_verbs_get_qp_dev(), which sets up what the thread reads in
     * that mode.
     */
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
    printf(RESULT_FMT_LAT);
    printf("\n");
    printf(RESULT_LINE);

    for (int idx = 0; idx < NUM_MSG_SIZE_LAT; idx++) {
        /* Warmup per size*/
        status = gpunetio_verbs_put_signal_lat(
            cstream, qp_gpu, 0, resources.num_iters, NUM_QP, resources.cuda_threads,
            message_size[idx], resources.data_buf[idx], htobe32(resources.data_mr[idx]->lkey),
            resources.flag_buf[idx], resources.prev_flag_buf[idx],
            htobe32(resources.prev_flag_mr[idx]->lkey), (uint8_t *)(resources.remote_data_buf[idx]),
            htobe32(resources.remote_data_mkey[idx]), (uint64_t *)(resources.remote_flag_buf[idx]),
            htobe32(resources.remote_flag_mkey[idx]), true);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function gpunetio_verbs_put_signal_lat failed: %d", status);
            goto stop_thread;
        }

        cu_result = cuEventRecord(e_start, cstream);
        if (cu_result) {
            DOCA_LOG(LOG_ERR, "Error in cuEventRecord for e_start");
            status = DOCA_ERROR_DRIVER;
            goto stop_thread;
        }

        /* Same signals as the warmup: they keep counting from where it stopped */
        status = gpunetio_verbs_put_signal_lat(
            cstream, qp_gpu, resources.num_iters, resources.num_iters, NUM_QP,
            resources.cuda_threads, message_size[idx], resources.data_buf[idx],
            htobe32(resources.data_mr[idx]->lkey), resources.flag_buf[idx],
            resources.prev_flag_buf[idx], htobe32(resources.prev_flag_mr[idx]->lkey),
            (uint8_t *)(resources.remote_data_buf[idx]), htobe32(resources.remote_data_mkey[idx]),
            (uint64_t *)(resources.remote_flag_buf[idx]), htobe32(resources.remote_flag_mkey[idx]),
            true);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Function gpunetio_verbs_put_signal_lat failed: %d", status);
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
    if (cpu_proxy_thread) {
        WRITE_ONCE(*args.exit_flag, 1);
        pthread_join(thread_id, NULL);
        free(args.exit_flag);
    }

destroy_events:
    if (e_start != NULL) cuEventDestroy(e_start);
    if (e_end != NULL) cuEventDestroy(e_end);
    if (cstream != NULL) cudaStreamDestroy(cstream);

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
