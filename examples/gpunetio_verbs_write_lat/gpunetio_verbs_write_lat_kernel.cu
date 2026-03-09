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

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <doca_gpunetio_dev_verbs_qp.cuh>
#include <doca_gpunetio_dev_verbs_cq.cuh>

#include "verbs_common.h"

#define ENABLE_DEBUG 0

/*
 * Ping-pong write latency test.
 * The code assumes the CUDA kernel is launched with only 1 CUDA thread.
 */
template <bool is_client, enum doca_gpu_dev_verbs_nic_handler nic_handler>
__global__ void write_lat(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t size,
                          uint8_t *local_poll_buf, uint32_t local_poll_mkey,
                          uint8_t *local_post_buf, uint32_t local_post_mkey, uint8_t *dst_buf,
                          uint32_t dst_buf_mkey) {
    uint64_t wqe_idx = 0;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    uint64_t scnt = 0;
    uint64_t rcnt = 0;
    int idx = (size * threadIdx.x) + (size - 1);
    uint64_t dst = (uint64_t)(dst_buf + (size * threadIdx.x));
    uint64_t src = (uint64_t)(local_post_buf + (size * threadIdx.x));

#ifdef DOCA_GPUNETIO_VERBS_HAS_TMA_COPY
    __shared__ struct doca_gpu_dev_verbs_wqe wqe_ptr_sh[CUDA_THREADS_LAT];
#endif

    wqe_idx = doca_gpu_dev_verbs_atomic_read<uint64_t,
                                             DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        &qp->sq_wqe_pi);

    while (scnt < num_iters || rcnt < num_iters) {
        if (rcnt < num_iters && (scnt >= 1 || is_client == true)) {
            ++rcnt;
            /*
             * To ensure correctness, client and servers polls on the latest byte of the expected
             * message. This should reflect an increased latency for bigger message size.
             */
            while (DOCA_GPUNETIO_VOLATILE(local_poll_buf[idx]) != (uint8_t)rcnt);
        }

        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();

        if (scnt < num_iters) {
            ++scnt;
            DOCA_GPUNETIO_VOLATILE(local_post_buf[idx]) = (uint8_t)scnt;

#ifdef DOCA_GPUNETIO_VERBS_HAS_TMA_COPY
            if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
                doca_gpu_dev_verbs_wqe_prepare_write(qp, &wqe_ptr_sh[threadIdx.x], wqe_idx,
                                                     MLX5_OPCODE_RDMA_WRITE,
                                                     DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                     0,  // immediate
                                                     dst, dst_buf_mkey, src, local_post_mkey, size);

                // SW-emulated reliable BF
                wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
                doca_gpu_dev_verbs_wqe_prepare_write(qp, wqe_ptr, wqe_idx, MLX5_OPCODE_RDMA_WRITE,
                                                     DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                     0,  // immediate
                                                     dst, dst_buf_mkey, src, local_post_mkey, size);
            } else {
#endif
                wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
                doca_gpu_dev_verbs_wqe_prepare_write(qp, wqe_ptr, wqe_idx, MLX5_OPCODE_RDMA_WRITE,
                                                     DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                     0,  // immediate
                                                     dst, dst_buf_mkey, src, local_post_mkey, size);

#ifdef DOCA_GPUNETIO_VERBS_HAS_TMA_COPY
            }
#endif
            /* DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU is needed to enforce an internal fence before
             * updating the BlueFlame to prevent instructions reordering */
            if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
#ifdef DOCA_GPUNETIO_VERBS_HAS_TMA_COPY
                doca_gpu_dev_verbs_submit_bf<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                             DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(
                    qp, wqe_idx + 1, &(wqe_ptr_sh[threadIdx.x]));
#else
                doca_gpu_dev_verbs_submit_bf<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                             DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(qp, wqe_idx + 1,
                                                                                 wqe_ptr);
#endif
            } else {
                doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                          DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU, nic_handler>(
                    qp, (wqe_idx + 1),
                    DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI);  // Useful only in case
                                                                            // of CPU proxy
            }

            wqe_idx++;
        }
    }

    if (doca_gpu_dev_verbs_poll_cq_collapsed_at(qp, wqe_idx - 1) != 0) {
#if ENABLE_DEBUG == 1
        printf("Error CQE!\n");
#endif
    }
}

extern "C" {

doca_error_t gpunetio_verbs_write_lat(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t cuda_blocks,
    uint32_t cuda_threads, uint32_t size, uint8_t *local_poll_buf, uint32_t local_poll_mkey,
    uint8_t *local_post_buf, uint32_t local_post_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey,
    enum doca_gpu_dev_verbs_nic_handler nic_handler, bool is_client) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    if (cuda_blocks > 1) {
        DOCA_LOG(LOG_ERR, "The kernel supports only 1 CUDA Block\n");
        return DOCA_ERROR_BAD_STATE;
    }

    if (cuda_threads > 1) {
        DOCA_LOG(LOG_ERR, "The kernel supports only 1 CUDA Thread\n");
        return DOCA_ERROR_BAD_STATE;
    }

    if (is_client) {
        if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF)
            write_lat<true, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
        else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)
            write_lat<true, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
        else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR)
            write_lat<true, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
        else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY)
            write_lat<true, DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
        else
            write_lat<true, DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
    } else {
        if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF)
            write_lat<false, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
        else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)
            write_lat<false, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
        else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR)
            write_lat<false, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
        else if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY)
            write_lat<false, DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
        else
            write_lat<false, DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>
                <<<cuda_blocks, cuda_threads, 0, stream>>>(qp, num_iters, size, local_poll_buf,
                                                           local_poll_mkey, local_post_buf,
                                                           local_post_mkey, dst_buf, dst_buf_mkey);
    }

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
}
