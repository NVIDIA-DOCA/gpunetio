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

/**
 * @file doca_sdk_wrapper.h
 * @brief Wrapper for DOCA SDK API calls and structs
 *
 * This wrapper provides an abstraction layer over DOCA SDK APIs.
 * It's enabled by default. At runtime, if the DOCA_SDK_LIB_PATH env var
 * is set, DOCA open will look for DOCA SDK to execute functions.
 * When DOCA_SDK_LIB_PATH env var is defined:
 * - All DOCA SDK API calls are wrapped using dlopen
 * - All DOCA SDK API structs are wrapped
 * - The wrapper provides a clean abstraction layer with dynamic loading
 *
 * If the env var DOCA_SDK_LIB_PATH is not set, the standalone open source implementation
 * is used. This means, DOCA SDK restricted features are not available.
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_SDK_WRAPPER_H
#define DOCA_GPUNETIO_SDK_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "host/doca_error.h"
#include "host/doca_gpunetio.h"
#include "doca_sdk_wrapper.h"

/* Wrapper function declarations */
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_create(const char *gpu_bus_id, void **gpu_dev);
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_destroy(void *gpu_dev);
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_mem_alloc(void *gpu_dev, size_t size,
                                                        size_t alignment,
                                                        enum doca_gpu_mem_type mtype,
                                                        void **memptr_gpu, void **memptr_cpu);

doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_mem_free(void *gpu, void *memptr_gpu);
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_dmabuf_fd(void *gpu, void *memptr_gpu, size_t size,
                                                        int *dmabuf_fd);
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_verbs_req_notify_cq(void *gpu_dev, void *verbs_cq);

doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_cap_is_gpu_mem_umem_supported(void *gpu_dev,
                                                                            void *net_dev);
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_cap_is_host_mem_umem_supported(void *gpu_dev,
                                                                             void *net_dev);
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_cap_is_nic_handler_gpu_sm_db_supported(void *gpu_dev,
                                                                                     void *net_dev);

/*
 * Values of enum doca_gpu_dev_verbs_nic_handler of the DOCA GPUNetIO SDK. The SDK gives one bit to
 * each handler, while this library combines flags, so the two enumerations don't share the values
 * of the handlers that set more than one flag. Translate before calling the SDK.
 */
#define DOCA_GPUNETIO_SDK_NIC_HANDLER_AUTO 0u
#define DOCA_GPUNETIO_SDK_NIC_HANDLER_CPU_PROXY (1u << 0)
#define DOCA_GPUNETIO_SDK_NIC_HANDLER_GPU_SM_DB (1u << 1)
#define DOCA_GPUNETIO_SDK_NIC_HANDLER_GPU_SM_BF (1u << 2)
#define DOCA_GPUNETIO_SDK_NIC_HANDLER_GPU_SM_NO_DBR (1u << 3)
#define DOCA_GPUNETIO_SDK_NIC_HANDLER_CPU_PROXY_NO_DBR (1u << 4)

/*
 * Check that the DOCA SDK in use has every entry point the export of a QP relies on:
 * doca_gpu_verbs_bridge_export_qp(), doca_gpu_verbs_get_qp_dev() and doca_gpu_verbs_unexport_qp().
 * Checked before building an SDK handler, so that none is built that couldn't be queried or
 * released.
 */
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_verbs_bridge_is_supported(void);

/*
 * Build an SDK GPUNetIO Verbs QP out of the queues of a QP this library owns.
 *
 * Wraps doca_gpu_verbs_bridge_export_qp(). nic_handler is a DOCA_GPUNETIO_SDK_NIC_HANDLER_* value,
 * not one of this library. qp_sdk receives the SDK struct doca_gpu_verbs_qp CPU handler.
 */
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_verbs_bridge_export_qp(
    void *gpu_dev, uint32_t sq_qpn, void *sq_wqe_addr, uint16_t sq_wqe_num, uint32_t *sq_dbrec,
    uint64_t *sq_db, size_t uar_size, uint32_t sq_cqn, void *sq_cqe_addr, uint32_t sq_cqe_num,
    uint32_t *sq_cq_dbrec, uint32_t rq_qpn, void *rq_wqe_addr, uint16_t rq_wqe_num,
    uint32_t *rq_dbrec, uint32_t rcv_wqe_size, uint32_t rq_cqn, void *rq_cqe_addr,
    uint32_t rq_cqe_num, uint32_t *rq_cq_dbrec, uint32_t nic_handler, void **qp_sdk);

/*
 * Destroy an SDK GPUNetIO Verbs QP built by doca_gpu_sdk_wrapper_verbs_bridge_export_qp().
 *
 * Wraps doca_gpu_verbs_unexport_qp() of the SDK: it frees the GPU handler and the CPU object, and
 * leaves the queues themselves alone.
 */
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_verbs_unexport_qp(void *gpu_dev, void *qp_sdk);

/*
 * Get the GPU handler of an SDK GPUNetIO Verbs QP.
 *
 * Wraps doca_gpu_verbs_get_qp_dev() of the SDK: qp_gpu receives a GPU pointer to the SDK struct
 * doca_gpu_dev_verbs_qp, whose layout is the SDK one.
 */
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_verbs_get_qp_dev(void *qp_sdk, void **qp_gpu);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_SDK_WRAPPER_H */

/** @} */
