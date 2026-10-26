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

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <doca_error.h>
#include <doca_gpunetio_dev_verbs_onesided.cuh>

#include "verbs_common.h"

/*
 * Each CUDA thread writes its own chunk of the destination buffer and then adds 1 to its own remote
 * signal with an atomic fetch and add, which returns the previous signal value into the thread's
 * prev_flag_buf entry. In warp scope, the warp posts the puts of its threads together with a single
 * fetch and add, on the signal and prev_flag_buf entry of its lane 0.
 */
template <enum doca_gpu_dev_verbs_exec_scope scope>
__global__ void put_signal_bw(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters,
                              uint32_t data_size, uint8_t *src_buf, uint32_t src_buf_mkey,
                              uint64_t *prev_flag_buf, uint32_t prev_flag_buf_mkey,
                              uint8_t *dst_buf, uint32_t dst_buf_mkey, uint64_t *dst_flag,
                              uint32_t dst_flag_mkey) {
    uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);

    for (uint32_t idx = tidx; idx < num_iters; idx += (blockDim.x * gridDim.x)) {
        doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
                                      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                      DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO, scope>(
            qp,
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)(dst_buf + (data_size * tidx)),
                                    .key = (uint32_t)dst_buf_mkey},
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)(src_buf + (data_size * tidx)),
                                    .key = (uint32_t)src_buf_mkey},
            data_size,
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)(&dst_flag[tidx]),
                                    .key = (uint32_t)dst_flag_mkey},
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)(&prev_flag_buf[tidx]),
                                    .key = (uint32_t)prev_flag_buf_mkey},
            (uint64_t)1);

        __syncthreads();
    }

    // Poll CQE to ensure writes are ok
	__syncthreads();
	if (threadIdx.x == 0)
		doca_gpu_dev_verbs_wait<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
					DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp);
}

extern "C" {

doca_error_t gpunetio_verbs_put_signal_bw(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t cuda_blocks,
    uint32_t cuda_threads, uint32_t data_size, uint8_t *src_buf, uint32_t src_buf_mkey,
    uint64_t *prev_flag_buf, uint32_t prev_flag_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey,
    uint64_t *dst_flag, uint32_t dst_flag_mkey, enum doca_gpu_dev_verbs_exec_scope scope) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
        put_signal_bw<DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD>
            <<<cuda_blocks, cuda_threads, 0, stream>>>(
                qp, num_iters, data_size, src_buf, src_buf_mkey, prev_flag_buf, prev_flag_buf_mkey,
                dst_buf, dst_buf_mkey, dst_flag, dst_flag_mkey);
    else if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
        put_signal_bw<DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP>
            <<<cuda_blocks, cuda_threads, 0, stream>>>(
                qp, num_iters, data_size, src_buf, src_buf_mkey, prev_flag_buf, prev_flag_buf_mkey,
                dst_buf, dst_buf_mkey, dst_flag, dst_flag_mkey);
    else {
        DOCA_LOG(LOG_ERR, "Invalid exec scope %d, must be thread or warp", scope);
        return DOCA_ERROR_INVALID_VALUE;
    }

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
}
