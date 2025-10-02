#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

show_help()
{
    echo "Usage: ${0} [-hco]"
    echo
    echo "  -h              Show this help text."
    echo "  -c <compiler>   Specify the compiler."
    echo "  -a <arch>       Specify the architecture."
    echo "  -o <file>       Specify the doca_gpunetio_config_file.h file location."
    echo
}

compiler=nvcc
arch="sm_100"
doca_gpunetio_config_file=

OPTIND=1    # Reset in case getopts has been used previously in the shell.
while getopts "hc:a:o:" opt ; do
    case "${opt}" in
        h)
            show_help
            exit 0
            ;;
        c)
            compiler="${OPTARG}"
            ;;
        a)
            arch="${OPTARG}"
            ;;
        o)
            doca_gpunetio_config_file="${OPTARG}"
            ;;
        ?)
            show_help
            exit 0
            ;;
    esac
done

tmpfolder=$(mktemp --tmpdir -d doca_gpunetio.XXXXXXXXX)

testfile="${tmpfolder}/test-dummy.cu"

cat >${testfile} <<EOF
#include <stdio.h>
#include <stdint.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

int main()
{
    struct mlx5dv_devx_umem_in umem_in = {
        0,
    };
    umem_in.comp_mask = MLX5DV_UMEM_MASK_DMABUF;
    umem_in.dmabuf_fd = 1;
    umem_in.addr = 0;

    struct mlx5dv_devx_umem *umem = mlx5dv_devx_umem_reg_ex(NULL, &umem_in);
    return 0;
}
EOF

cd ${tmpfolder}
${compiler} -arch ${arch} -o test-dummy test-dummy.cu -libverbs -lmlx5 > /dev/null 2>&1
ret=$?

rm -rf ${tmpfolder}

if [ -n "${doca_gpunetio_config_file}" ]; then
    echo "/* DOCA_GPUNETIO_HAVE_MLX5DV_UMEM_DMABUF support */" >> ${doca_gpunetio_config_file}
    if [ "${ret}" -eq 0 ]; then
        echo "#define DOCA_GPUNETIO_HAVE_MLX5DV_UMEM_DMABUF 1" >> ${doca_gpunetio_config_file}
    else
        echo "/* #undef DOCA_GPUNETIO_HAVE_MLX5DV_UMEM_DMABUF */" >> ${doca_gpunetio_config_file}
    fi
fi

if [ "${ret}" -eq 0 ]; then
    echo "y"
else
    echo "n"
fi

