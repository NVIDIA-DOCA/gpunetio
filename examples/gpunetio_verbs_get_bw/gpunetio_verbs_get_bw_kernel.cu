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

template <enum doca_gpu_dev_verbs_exec_scope scope>
__global__ void get_bw(struct doca_gpu_dev_verbs_qp *qp, uint32_t num_iters, uint32_t data_size,
                       uint8_t *local_buf, uint32_t local_buf_mkey, uint8_t *remote_buf,
                       uint32_t remote_buf_mkey, uint64_t *dump_flag, uint32_t dump_flag_mkey) {
    doca_gpu_dev_verbs_ticket_t out_ticket;
    uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);

    for (uint32_t idx = tidx; idx < num_iters; idx += (blockDim.x * gridDim.x)) {
        /* The memory consistency Dump WQE, when needed, is posted once by get_wait() below */
        doca_gpu_dev_verbs_get<
            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
            scope, DOCA_GPUNETIO_VERBS_MCST_DISABLED, DOCA_GPUNETIO_VERBS_BLOCKING_MODE_DISABLED>(
            qp,
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)(remote_buf + (data_size * tidx)),
                                    .key = (uint32_t)remote_buf_mkey},
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)(local_buf + (data_size * tidx)),
                                    .key = (uint32_t)local_buf_mkey},
            data_size, doca_gpu_dev_verbs_addr{.addr = 0, .key = 0}, &out_ticket);

        __syncthreads();
    }

    /*
     * The get waits for a free send queue slot by itself, so there is no need to poll each
     * completion: the first thread of each block waits for the last posted get, which completes
     * after all the previous ones. Before Hopper, the data the NIC wrote into GPU memory may not be
     * visible yet when the completion arrives: get_wait() then posts a Dump WQE, which reads the
     * dump flag after the data, and waits for it instead.
     */
    if (threadIdx.x == 0) {
#if __CUDA_ARCH__ < 900
        doca_gpu_dev_verbs_get_wait<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                    DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                                    DOCA_GPUNETIO_VERBS_MCST_ENABLED>(
            qp,
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)dump_flag, .key = (uint32_t)dump_flag_mkey});
#else
        doca_gpu_dev_verbs_get_wait<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                    DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                                    DOCA_GPUNETIO_VERBS_MCST_DISABLED>(
            qp, doca_gpu_dev_verbs_addr{.addr = 0, .key = 0});
#endif
    }
}

extern "C" {

doca_error_t gpunetio_verbs_get_bw(cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp,
                                   uint32_t num_iters, uint32_t cuda_blocks, uint32_t cuda_threads,
                                   uint32_t data_size, uint8_t *local_buf, uint32_t local_buf_mkey,
                                   uint8_t *remote_buf, uint32_t remote_buf_mkey,
                                   uint64_t *dump_flag, uint32_t dump_flag_mkey,
                                   enum doca_gpu_dev_verbs_exec_scope scope) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD)
        get_bw<DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD><<<cuda_blocks, cuda_threads, 0, stream>>>(
            qp, num_iters, data_size, local_buf, local_buf_mkey, remote_buf, remote_buf_mkey,
            dump_flag, dump_flag_mkey);
    else if (scope == DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP)
        get_bw<DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP><<<cuda_blocks, cuda_threads, 0, stream>>>(
            qp, num_iters, data_size, local_buf, local_buf_mkey, remote_buf, remote_buf_mkey,
            dump_flag, dump_flag_mkey);
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
