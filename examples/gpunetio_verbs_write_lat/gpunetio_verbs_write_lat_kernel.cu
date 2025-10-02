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

template <bool is_client>
__global__ void write_lat(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t size,
                          uint8_t *local_poll_buf, uint32_t local_poll_mkey,
                          uint8_t *local_post_buf, uint32_t local_post_mkey, uint8_t *dst_buf,
                          uint32_t dst_buf_mkey) {
    uint64_t wqe_idx = 0, cqe_idx = 0;
    struct doca_gpu_dev_verbs_cq *cq_sq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE;
    uint64_t scnt = 0;
    uint64_t rcnt = 0;

    if (threadIdx.x == (blockDim.x - 1)) cflag = DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE;

    while (scnt < num_iters || rcnt < num_iters) {
        if (rcnt < num_iters && (scnt >= 1 || is_client == true)) {
            ++rcnt;
            while (DOCA_GPUNETIO_VOLATILE(local_poll_buf[size * threadIdx.x]) != (uint8_t)rcnt);
        }
        __threadfence_block();

        if (scnt < num_iters) {
            ++scnt;
            DOCA_GPUNETIO_VOLATILE(local_post_buf[size * threadIdx.x]) = (uint8_t)scnt;

            wqe_idx = doca_gpu_dev_verbs_atomic_read<uint64_t,
                                                     DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &qp->sq_wqe_pi);
            wqe_idx += threadIdx.x;
            wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, (wqe_idx & qp->sq_wqe_mask));

            doca_gpu_dev_verbs_wqe_prepare_write(
                qp, wqe_ptr, wqe_idx, MLX5_OPCODE_RDMA_WRITE, cflag,
                0,  // immediate
                (uint64_t)(dst_buf + (size * threadIdx.x)), dst_buf_mkey,
                (uint64_t)(local_post_buf + (size * threadIdx.x)), local_post_mkey, size);

            __syncthreads();

            if (threadIdx.x == (blockDim.x - 1)) {
                doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                    qp, (wqe_idx + 1));
                // Wait for final CQE in block of iterations
                cqe_idx =
                    doca_gpu_dev_verbs_atomic_read<uint64_t,
                                                   DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                        &(cq_sq->cqe_ci)) &
                    cq_sq->cqe_mask;
                if (doca_gpu_dev_verbs_poll_cq_at(cq_sq, cqe_idx) != 0) printf("Error CQE!\n");
                /* TODO: move this instruction in doca_gpu_dev_verbs_ring_proxy() */
                if (qp->nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY)
                    doca_gpu_dev_verbs_atomic_max<
                        uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, true>(
                        &qp->sq_wqe_pi, wqe_idx + 1);
            }
            __syncthreads();
        }
    }
}

extern "C" {

doca_error_t gpunetio_verbs_write_lat(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                      uint32_t num_iters, uint32_t cuda_blocks,
                                      uint32_t cuda_threads, uint32_t size, uint8_t *local_poll_buf,
                                      uint32_t local_poll_mkey, uint8_t *local_post_buf,
                                      uint32_t local_post_mkey, uint8_t *dst_buf,
                                      uint32_t dst_buf_mkey, bool is_client) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    if (is_client) {
        write_lat<true><<<cuda_blocks, cuda_threads, 0, stream>>>(
            qp, num_iters, size, local_poll_buf, local_poll_mkey, local_post_buf, local_post_mkey,
            dst_buf, dst_buf_mkey);
    } else {
        write_lat<false><<<cuda_blocks, cuda_threads, 0, stream>>>(
            qp, num_iters, size, local_poll_buf, local_poll_mkey, local_post_buf, local_post_mkey,
            dst_buf, dst_buf_mkey);
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