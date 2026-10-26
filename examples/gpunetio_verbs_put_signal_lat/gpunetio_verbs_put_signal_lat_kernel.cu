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
 * Ping-pong put with signal latency test.
 * The code assumes the CUDA kernel is launched with only 1 CUDA thread.
 *
 * Each message writes the remote data buffer, then adds 1 to the remote signal. A side waits for
 * its local signal, which the remote peer increments, to count the messages received so far. The
 * signals keep counting across launches on the same buffers, from start_iters.
 */
template <bool is_client>
__global__ void put_signal_lat(struct doca_gpu_dev_verbs_qp *qp, uint32_t start_iters,
                               uint32_t num_iters, uint32_t data_size, uint8_t *src_buf,
                               uint32_t src_buf_mkey, uint64_t *local_flag, uint64_t *prev_flag_buf,
                               uint32_t prev_flag_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey,
                               uint64_t *dst_flag, uint32_t dst_flag_mkey) {
    for (uint32_t iter_idx = 0; iter_idx < num_iters; iter_idx++) {
        /* The server replies to each message of the client */
        if (!is_client) {
            while (DOCA_GPUNETIO_VOLATILE(*local_flag) < (uint64_t)start_iters + iter_idx + 1);
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        }

        /* Exclusive sharing mode: a single CUDA thread uses the QP */
        doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD,
                                      DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                      DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
            qp, doca_gpu_dev_verbs_addr{.addr = (uint64_t)dst_buf, .key = (uint32_t)dst_buf_mkey},
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)src_buf, .key = (uint32_t)src_buf_mkey},
            data_size,
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)dst_flag, .key = (uint32_t)dst_flag_mkey},
            doca_gpu_dev_verbs_addr{.addr = (uint64_t)prev_flag_buf,
                                    .key = (uint32_t)prev_flag_buf_mkey},
            (uint64_t)1);

        /* The client waits for the reply before sending the next message */
        if (is_client) {
            while (DOCA_GPUNETIO_VOLATILE(*local_flag) < (uint64_t)start_iters + iter_idx + 1);
            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        }
    }

    /* Wait for the last put with signal, which completes after all the previous WQEs */
    doca_gpu_dev_verbs_wait<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp);
}

extern "C" {

doca_error_t gpunetio_verbs_put_signal_lat(
    cudaStream_t stream, struct doca_gpu_dev_verbs_qp *qp, uint32_t start_iters, uint32_t num_iters,
    uint32_t cuda_blocks, uint32_t cuda_threads, uint32_t data_size, uint8_t *src_buf,
    uint32_t src_buf_mkey, uint64_t *local_flag, uint64_t *prev_flag_buf,
    uint32_t prev_flag_buf_mkey, uint8_t *dst_buf, uint32_t dst_buf_mkey, uint64_t *dst_flag,
    uint32_t dst_flag_mkey, bool is_client) {
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    if (cuda_blocks != 1) {
        DOCA_LOG(LOG_ERR, "The kernel supports only 1 CUDA Block, requested cuda_blocks=%u",
                 cuda_blocks);
        return DOCA_ERROR_BAD_STATE;
    }

    if (cuda_threads != 1) {
        DOCA_LOG(LOG_ERR, "The kernel supports only 1 CUDA Thread, requested cuda_threads=%u",
                 cuda_threads);
        return DOCA_ERROR_BAD_STATE;
    }

    if (is_client)
        put_signal_lat<true><<<cuda_blocks, cuda_threads, 0, stream>>>(
            qp, start_iters, num_iters, data_size, src_buf, src_buf_mkey, local_flag, prev_flag_buf,
            prev_flag_buf_mkey, dst_buf, dst_buf_mkey, dst_flag, dst_flag_mkey);
    else
        put_signal_lat<false><<<cuda_blocks, cuda_threads, 0, stream>>>(
            qp, start_iters, num_iters, data_size, src_buf, src_buf_mkey, local_flag, prev_flag_buf,
            prev_flag_buf_mkey, dst_buf, dst_buf_mkey, dst_flag, dst_flag_mkey);

    result = cudaGetLastError();
    if (cudaSuccess != result) {
        DOCA_LOG(LOG_ERR, "[%s:%d] cuda failed with %s", __FILE__, __LINE__,
                 cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}
}
