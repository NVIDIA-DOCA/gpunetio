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
    struct verbs_config verbs_cfg;
    doca_error_t status;
    struct doca_log_backend *sdk_log;
    int exit_status = EXIT_FAILURE;
    int option;

    /* Set the default configuration values */
    verbs_cfg.is_server = true;
    verbs_cfg.gid_index = DEFAULT_GID_INDEX;
    verbs_cfg.num_iters = NUM_ITERS;
    /* Only 1 Thread is needed for the latency test */
    verbs_cfg.cuda_threads = CUDA_THREADS_LAT;
    verbs_cfg.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;

    while ((option = getopt(argc, argv, "c:d:g:i:p:")) != -1) {
        switch (option) {
            case 'c': {
                verbs_cfg.server_ip_addr = optarg;
                verbs_cfg.is_server = false;
                break;
            }
            case 'd': {
                verbs_cfg.nic_device_name = optarg;
                break;
            }
            case 'g': {
                verbs_cfg.gpu_pcie_addr = optarg;
                break;
            }
            case 'i': {
                verbs_cfg.num_iters = std::atoi(optarg);
                break;
            }
            case 'p': {
                verbs_cfg.nic_handler = (enum doca_gpu_dev_verbs_nic_handler)std::atoi(optarg);
                break;
            }
            default:
                std::cerr << "Usage: " << argv[0] << " -n name" << std::endl;
                return 1;
        }
    }

    if (verbs_cfg.is_server) {
        status = verbs_server(&verbs_cfg);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "verbs_server() failed");
            goto sample_exit;
        }
    } else {
        status = verbs_client(&verbs_cfg);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "verbs_client() failed");
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
