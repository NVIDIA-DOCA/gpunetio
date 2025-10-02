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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <doca_verbs_uar.hpp>

#include "verbs_common.h"

/*
 * OOB connection to exchange RDMA info - server side
 *
 * @oob_sock_fd [out]: Socket FD
 * @oob_client_sock [out]: Client socket FD
 * @return: positive integer on success and -1 otherwise
 */
int oob_verbs_connection_server_setup(int *oob_sock_fd, int *oob_client_sock) {
    struct sockaddr_in server_addr = {0}, client_addr = {0};
    unsigned int client_size = 0;
    int enable = 1;
    int oob_sock_fd_ = 0;
    int oob_client_sock_ = 0;

    /* Create socket */
    oob_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (oob_sock_fd_ < 0) {
        DOCA_LOG(LOG_ERR, "Error while creating socket %d", oob_sock_fd_);
        return -1;
    }

    DOCA_LOG(LOG_INFO, "Socket created successfully");

    if (setsockopt(oob_sock_fd_, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(enable))) {
        DOCA_LOG(LOG_ERR, "Error setting socket options");
        close(oob_sock_fd_);
        return -1;
    }

    /* Set port and IP: */
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(2000);
    server_addr.sin_addr.s_addr = INADDR_ANY; /* listen on any interface */

    /* Bind to the set port and IP: */
    if (bind(oob_sock_fd_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        DOCA_LOG(LOG_ERR, "Couldn't bind to the port");
        close(oob_sock_fd_);
        return -1;
    }
    DOCA_LOG(LOG_INFO, "Done with binding");

    /* Listen for clients: */
    if (listen(oob_sock_fd_, 1) < 0) {
        DOCA_LOG(LOG_ERR, "Error while listening");
        close(oob_sock_fd_);
        return -1;
    }
    DOCA_LOG(LOG_INFO, "Listening for incoming connections");

    /* Accept an incoming connection: */
    client_size = sizeof(client_addr);
    oob_client_sock_ = accept(oob_sock_fd_, (struct sockaddr *)&client_addr, &client_size);
    if (oob_client_sock_ < 0) {
        DOCA_LOG(LOG_ERR, "Can't accept socket connection %d", oob_client_sock_);
        close(oob_sock_fd_);
        return -1;
    }

    *(oob_sock_fd) = oob_sock_fd_;
    *(oob_client_sock) = oob_client_sock_;

    DOCA_LOG(LOG_INFO, "Client connected at IP: %s and port: %i", inet_ntoa(client_addr.sin_addr),
             ntohs(client_addr.sin_port));

    return 0;
}

/*
 * OOB connection to exchange RDMA info - server side closure
 *
 * @oob_sock_fd [in]: Socket FD
 * @oob_client_sock [in]: Client socket FD
 * @return: positive integer on success and -1 otherwise
 */
void oob_verbs_connection_server_close(int oob_sock_fd, int oob_client_sock) {
    if (oob_client_sock > 0) close(oob_client_sock);

    if (oob_sock_fd > 0) close(oob_sock_fd);
}

/*
 * OOB connection to exchange RDMA info - client side
 *
 * @server_ip [in]: Server IP address to connect
 * @oob_sock_fd [out]: Socket FD
 * @return: positive integer on success and -1 otherwise
 */
int oob_verbs_connection_client_setup(const char *server_ip, int *oob_sock_fd) {
    struct sockaddr_in server_addr = {0};
    int oob_sock_fd_;

    /* Create socket */
    oob_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (oob_sock_fd_ < 0) {
        DOCA_LOG(LOG_ERR, "Unable to create socket");
        return -1;
    }
    DOCA_LOG(LOG_INFO, "Socket created successfully");

    /* Set port and IP the same as server-side: */
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(2000);
    server_addr.sin_addr.s_addr = inet_addr(server_ip);

    /* Send connection request to server: */
    if (connect(oob_sock_fd_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        close(oob_sock_fd_);
        DOCA_LOG(LOG_ERR, "Unable to connect to server at %s", server_ip);
        return -1;
    }
    DOCA_LOG(LOG_INFO, "Connected with server successfully");

    *oob_sock_fd = oob_sock_fd_;
    return 0;
}

/*
 * OOB connection to exchange RDMA info - client side closure
 *
 * @oob_sock_fd [in]: Socket FD
 * @return: positive integer on success and -1 otherwise
 */
void oob_verbs_connection_client_close(int oob_sock_fd) {
    if (oob_sock_fd > 0) close(oob_sock_fd);
}

static struct ibv_context *open_ib_device(const char *name) {
    int nb_ibdevs = 0;
    struct ibv_device **ibdev_list = ibv_get_device_list(&nb_ibdevs);
    struct ibv_context *context;

    if ((ibdev_list == NULL) || (nb_ibdevs == 0)) {
        DOCA_LOG(LOG_ERR, "Failed to get RDMA devices list, ibdev_list null");
        return NULL;
    }

    for (uint32_t i = 0; i < nb_ibdevs; i++) {
        if (strncmp(ibv_get_device_name(ibdev_list[i]), name, strlen(name)) == 0) {
            struct ibv_device *dev_handle = ibdev_list[i];
            ibv_free_device_list(ibdev_list);

            context = ibv_open_device(dev_handle);
            return context;
        }
    }

    ibv_free_device_list(ibdev_list);

    return NULL;
}

static doca_error_t create_verbs_ah_attr(struct ibv_context *verbs_context, uint32_t gid_index,
                                         enum doca_verbs_addr_type addr_type,
                                         struct doca_verbs_gid gid,
                                         struct doca_verbs_ah_attr **verbs_ah_attr) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct doca_verbs_ah_attr *new_ah_attr = NULL;

    status = doca_verbs_ah_attr_create(verbs_context, &new_ah_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs ah attributes: %d", status);
        return status;
    }

    status = doca_verbs_ah_attr_set_addr_type(new_ah_attr, addr_type);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set address type: %d", status);
        goto destroy_verbs_ah;
    }

    status = doca_verbs_ah_attr_set_sgid_index(new_ah_attr, gid_index);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set sgid index: %d", status);
        goto destroy_verbs_ah;
    }

    status = doca_verbs_ah_attr_set_hop_limit(new_ah_attr, VERBS_TEST_HOP_LIMIT);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set hop limit: %d", status);
        goto destroy_verbs_ah;
    }

    *verbs_ah_attr = new_ah_attr;

    return DOCA_SUCCESS;

destroy_verbs_ah:
    tmp_status = doca_verbs_ah_attr_destroy(new_ah_attr);
    if (tmp_status != DOCA_SUCCESS)
        DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs AH: %d", tmp_status);

    return status;
}

doca_error_t create_verbs_resources(struct verbs_config *cfg, struct verbs_resources *resources) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    union ibv_gid rgid;
    int ret = 0;
    struct ibv_port_attr port_attr;
    struct doca_gpu_verbs_qp_init_attr_hl qp_init;
    resources->cfg = cfg;

    cudaFree(0);
    cudaSetDevice(0);

    status = doca_gpu_create(cfg->gpu_pcie_addr.c_str(), &resources->gpu_dev);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca gpu context: %d", status);
        goto destroy_resources;
    }

    resources->verbs_context = open_ib_device(cfg->nic_device_name.c_str());
    if (resources->verbs_context == NULL) {
        DOCA_LOG(LOG_ERR, "Failed to create doca network context");
        status = DOCA_ERROR_BAD_STATE;
        goto destroy_resources;
    }

    resources->verbs_pd = ibv_alloc_pd(resources->verbs_context);
    if (!resources->verbs_pd) {
        DOCA_LOG(LOG_ERR, "Failed to create doca network context");
        goto destroy_resources;
    }

    ret = ibv_query_gid(resources->verbs_pd->context, 1, cfg->gid_index, &rgid);
    if (ret) {
        DOCA_LOG(LOG_ERR, "Failed to query ibv gid attributes");
        status = DOCA_ERROR_DRIVER;
        goto destroy_resources;
    }
    memcpy(resources->gid.raw, rgid.raw, DOCA_VERBS_GID_BYTE_LENGTH);

    ret = ibv_query_port(resources->verbs_pd->context, 1, &port_attr);
    if (ret) {
        DOCA_LOG(LOG_ERR, "Failed to query ibv port attributes");
        status = DOCA_ERROR_DRIVER;
        goto destroy_resources;
    }

    if (port_attr.link_layer == 1) {
        status = create_verbs_ah_attr(resources->verbs_context, cfg->gid_index,
                                      DOCA_VERBS_ADDR_TYPE_IB_NO_GRH, resources->gid,
                                      &resources->verbs_ah_attr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create doca verbs ah attributes");
            goto destroy_resources;
        }
        resources->lid = port_attr.lid;
    } else {
        status = create_verbs_ah_attr(resources->verbs_context, cfg->gid_index,
                                      DOCA_VERBS_ADDR_TYPE_IPv4, resources->gid,
                                      &resources->verbs_ah_attr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create doca verbs ah attributes");
            goto destroy_resources;
        }
    }

    qp_init.gpu_dev = resources->gpu_dev;
    qp_init.ibpd = resources->verbs_pd;
    qp_init.sq_nwqe = VERBS_TEST_QUEUE_SIZE;
    qp_init.nic_handler = resources->nic_handler;
    qp_init.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;

    status = doca_gpu_verbs_create_qp_hl(&qp_init, &(resources->qp));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca verbs high-level qp %d", status);
        goto destroy_resources;
    }

    resources->local_qp_number = doca_verbs_qp_get_qpn(resources->qp->qp);

    return DOCA_SUCCESS;

destroy_resources:
    tmp_status = destroy_verbs_resources(resources);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy resources: %d", tmp_status);
    }
    return status;
}

doca_error_t destroy_verbs_resources(struct verbs_resources *resources) {
    doca_error_t status = DOCA_SUCCESS;

    status = doca_gpu_verbs_destroy_qp_hl(resources->qp);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs high-level qp");
        return status;
    }

    if (resources->verbs_ah_attr) {
        status = doca_verbs_ah_attr_destroy(resources->verbs_ah_attr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy doca verbs AH: %d", status);
            return status;
        }
    }

    if (resources->gpu_dev) {
        status = doca_gpu_destroy(resources->gpu_dev);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy pf doca gpu context: %d", status);
            return status;
        }
    }

    if (resources->verbs_pd) ibv_dealloc_pd(resources->verbs_pd);

    if (resources->verbs_context) ibv_close_device(resources->verbs_context);

    return DOCA_SUCCESS;
}

doca_error_t connect_verbs_qp(struct verbs_resources *resources) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    struct doca_verbs_qp_attr *verbs_qp_attr = NULL;
    int ret;
    struct ibv_port_attr port_attr;

    status = doca_verbs_ah_attr_set_gid(resources->verbs_ah_attr, resources->remote_gid);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set remote gid: %d", status);
        return status;
    }

    ret = ibv_query_port(resources->verbs_pd->context, 1, &port_attr);
    if (ret) {
        DOCA_LOG(LOG_ERR, "Failed to query ibv port attributes");
        status = DOCA_ERROR_DRIVER;
        return status;
    }

    /* IB only parameter */
    if (port_attr.link_layer == 1) {
        status = doca_verbs_ah_attr_set_dlid(resources->verbs_ah_attr, resources->dlid);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to set dlid");
            return status;
        }
    }

    status = doca_verbs_qp_attr_create(&verbs_qp_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create DOCA verbs QP attributes: %d", status);
        return status;
    }

    status = doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_VERBS_MTU_SIZE_1K_BYTES);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set path MTU: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set RQ PSN: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set SQ PSN: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_port_num(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set port number: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, 14);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set ACK timeout: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, 7);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set retry counter: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set RNR retry: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set minimum RNR timer: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set next state: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set allow remote write: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set allow remote read: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_allow_remote_atomic(verbs_qp_attr,
                                                        DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set destination QP number");
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, resources->verbs_ah_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set address handle: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, resources->remote_qp_number);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set destination QP number: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status =
        doca_verbs_qp_modify(resources->qp->qp, verbs_qp_attr,
                             DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                 DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
                                 DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set next state: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status =
        doca_verbs_qp_modify(resources->qp->qp, verbs_qp_attr,
                             DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                 DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                 DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to set next state: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_modify(resources->qp->qp, verbs_qp_attr,
                                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                      DOCA_VERBS_QP_ATTR_ACK_TIMEOUT |
                                      DOCA_VERBS_QP_ATTR_RETRY_CNT | DOCA_VERBS_QP_ATTR_RNR_RETRY);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP: %d", status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_destroy(verbs_qp_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy DOCA verbs QP attributes: %d", status);
        return status;
    }

    DOCA_LOG(LOG_INFO, "QP has been successfully connected and ready to use");

    return DOCA_SUCCESS;

destroy_verbs_qp_attr:
    tmp_status = doca_verbs_qp_attr_destroy(verbs_qp_attr);
    if (tmp_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy DOCA verbs QP attributes: %d", tmp_status);
    }

    return status;
}

void *progress_cpu_proxy(void *args_) {
    struct cpu_proxy_args *args = (struct cpu_proxy_args *)args_;

    printf("Thread CPU proxy progress is running... %ld\n",
           *((volatile uint64_t *)args->exit_flag));

    while (*((volatile uint64_t *)args->exit_flag) == 0) {
        doca_gpu_verbs_cpu_proxy_progress(args->qp_cpu);
    }

    return NULL;
}

size_t get_page_size(void) {
    long ret = sysconf(_SC_PAGESIZE);
    if (ret == -1) return 4096;  // 4KB, default Linux page size

    return (size_t)ret;
}
