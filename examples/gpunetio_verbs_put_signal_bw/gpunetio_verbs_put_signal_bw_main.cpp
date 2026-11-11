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

#include <stdlib.h>
#include "verbs_common.h"

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv) {
    struct verbs_config verbs_cfg = {};
    doca_error_t status;
    int exit_status = EXIT_FAILURE;
    int option;

    /* Set the default configuration values */
    verbs_cfg.is_server = true;
    verbs_cfg.gid_index = DEFAULT_GID_INDEX;
    verbs_cfg.num_iters = NUM_ITERS;
    verbs_cfg.cuda_threads = CUDA_THREADS_BW;
    verbs_cfg.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    verbs_cfg.exec_scope = DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD;
    verbs_cfg.enable_data_direct = false;

    while ((option = getopt(argc, argv, "c:d:e:g:i:l:p:z:")) != -1) {
        switch (option) {
            case 'c': {
                struct in_addr addr;
                if (inet_pton(AF_INET, optarg, &addr) != 1) {
                    DOCA_LOG(LOG_ERR, "Entered server address %s is not a valid IPv4 address",
                             optarg);
                    return 1;
                }
                verbs_cfg.server_ip_addr = optarg;
                verbs_cfg.is_server = false;
                break;
            }
            case 'd': {
                verbs_cfg.nic_device_name = optarg;
                break;
            }
            case 'e': {
                int exec_scope = std::atoi(optarg);
                if (exec_scope != DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD &&
                    exec_scope != DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP) {
                    DOCA_LOG(LOG_ERR,
                             "Execution scope must be %d (thread) or %d (warp), requested %d",
                             DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
                             DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP, exec_scope);
                    return 1;
                }
                verbs_cfg.exec_scope = (enum doca_gpu_dev_verbs_exec_scope)exec_scope;
                break;
            }
            case 'g': {
                verbs_cfg.gpu_pcie_addr = optarg;
                break;
            }
            case 'i': {
                int iters = std::atoi(optarg);
                if (iters <= 0) {
                    DOCA_LOG(LOG_ERR, "Number of iterations must be greater than 0, requested %d",
                             iters);
                    return 1;
                }
                verbs_cfg.num_iters = (uint32_t)iters;
                break;
            }
            case 'l': {
                int gid_index = std::atoi(optarg);
                if (gid_index < 0) {
                    DOCA_LOG(LOG_ERR, "GID index must be non-negative, requested %d", gid_index);
                    return 1;
                }
                verbs_cfg.gid_index = (uint32_t)gid_index;
                break;
            }
            case 'p': {
                int nic_handler = std::atoi(optarg);
                if (nic_handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF) {
                    DOCA_LOG(LOG_ERR, "NIC handler BlueFlame not supported in this example");
                    return 1;
                }
                if (nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO &&
                    nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY &&
                    nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
                    nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR &&
                    nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW) {
                    DOCA_LOG(LOG_ERR,
                             "NIC handler must be %d (AUTO), %d (CPU proxy), %d (GPU SM DB), %d "
                             "(GPU SM no DBR) or %d (CPU proxy free flow), requested %d",
                             DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                             DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY,
                             DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                             DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR,
                             DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW, nic_handler);
                    return 1;
                }
                verbs_cfg.nic_handler = (enum doca_gpu_dev_verbs_nic_handler)nic_handler;
                break;
            }
            case 'z': {
                verbs_cfg.enable_data_direct = (std::atoi(optarg) != 0);
                break;
            }
            default:
                std::cerr
                    << "Usage: " << argv[0] << "\n"
                    << " -c <server_ip> (Client only)\n"
                    << " -d <mlx5_X device name>\n"
                    << " -e <exec scope. 0: THREAD 1: WARP (default: 0)>\n"
                    << " -g <GPU PCIe address>\n"
                    << " -i <number of iterations, a multiple of " << CUDA_THREADS_BW
                    << " (default: " << NUM_ITERS << ")>\n"
                    << " -l <GID Index (default: 0)>\n"
                    << " -p <NIC handler. 0: AUTO 1: CPU PROXY 2: GPU SM_DB 10: GPU SM no DBR "
                       "17: CPU PROXY free flow (default: 0)>\n"
                    << " -z <Data Direct. 0: disabled 1: enabled (default: 0)>\n"
                    << std::endl;
                return 1;
        }
    }

    /* The kernel splits the CUDA threads over VERBS_CUDA_BLOCK CUDA blocks */
    status = verbs_validate_config(&verbs_cfg, VERBS_CUDA_BLOCK, 0);
    if (status != DOCA_SUCCESS) goto sample_exit;

    if (verbs_cfg.is_server) {
        status = verbs_server(&verbs_cfg);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "verbs_server() failed: %d", status);
            goto sample_exit;
        }
    } else {
        status = verbs_client(&verbs_cfg);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "verbs_client() failed: %d", status);
            goto sample_exit;
        }
    }

    exit_status = EXIT_SUCCESS;

sample_exit:
    if (exit_status == EXIT_SUCCESS)
        DOCA_LOG(LOG_INFO, "Sample finished successfully");
    else
        DOCA_LOG(LOG_INFO, "Sample finished with errors");
    return exit_status;
}
