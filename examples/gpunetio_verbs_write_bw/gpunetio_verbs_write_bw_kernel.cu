/*
 * Copyright (c) 2025-2026 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <doca_gpunetio_dev_verbs_qp.cuh>
#include <doca_gpunetio_dev_verbs_cq.cuh>

#include "verbs_common.h"

#define ENABLE_DEBUG 0

__global__ void write_bw(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t size,
                         uint8_t *src_buf, uint32_t src_buf_mkey, uint8_t *dst_buf,
                         uint32_t dst_buf_mkey) {
    uint64_t wqe_idx = 0;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;

    wqe_idx = (doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_wqe_pi) + threadIdx.x);

    for (uint32_t idx = threadIdx.x; idx < num_iters; idx += blockDim.x) {
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

        doca_gpu_dev_verbs_wqe_prepare_write(
            qp, wqe_ptr, wqe_idx, MLX5_OPCODE_RDMA_WRITE, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
            0, (uint64_t)(dst_buf + (size * threadIdx.x)), dst_buf_mkey,
            (uint64_t)(src_buf + (size * threadIdx.x)), src_buf_mkey, size);

        __syncthreads();

        if (threadIdx.x == (blockDim.x - 1))
            doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
                qp, (wqe_idx + 1), DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI);

        __syncthreads();

        wqe_idx += blockDim.x;
    }

    // Assumption: QP is long enough to hold all the WQEs posted in the loop.
    // Application needs to poll only the last CQE corresponding to the last posted WQE.
    if (threadIdx.x == (blockDim.x - 1)) {
        if (doca_gpu_dev_verbs_poll_cq_at(doca_gpu_dev_verbs_qp_get_cq_sq(qp),
                                          (wqe_idx - blockDim.x)) != 0) {
#if ENABLE_DEBUG == 1
            printf("Error CQE!\n");
#endif
        }
    }
    __syncthreads();
}

extern "C" {

doca_error_t gpunetio_verbs_write_bw(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                     uint32_t cuda_threads_iters, uint32_t cuda_blocks,
                                     uint32_t cuda_threads, uint32_t size, uint8_t *src_buf,
                                     uint32_t src_buf_mkey, uint8_t *dst_buf,
                                     uint32_t dst_buf_mkey) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    if (cuda_blocks > 1) {
        DOCA_LOG(LOG_ERR, "The kernel supports only 1 CUDA Block\n");
        return DOCA_ERROR_BAD_STATE;
    }

    write_bw<<<cuda_blocks, cuda_threads, 0, stream>>>(qp, cuda_threads_iters, size, src_buf,
                                                       src_buf_mkey, dst_buf, dst_buf_mkey);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
}
