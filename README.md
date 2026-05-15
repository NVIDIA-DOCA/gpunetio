# DOCA GPUNetIO Open Source

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE.txt)
[![CUDA](https://img.shields.io/badge/CUDA-12.2%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Contributions](https://img.shields.io/badge/Contributions-Not%20Accepted-red.svg)]()

This repository provides an open-source version of the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) and [DOCA Verbs](https://docs.nvidia.com/doca/sdk/doca+verbs/index.html) libraries. The features included here are limited to enabling **GPUDirect Async Kernel-Initiated (GDAKI)** network communication technology over RDMA protocols (InfiniBand and RoCE) using a DOCA-like API in an open-source environment.

## Open vs Full

The table below highlights the key differences between this DOCA GPUNetIO open-source project and the full DOCA GPUNetIO SDK:

| Item | DOCA Full SDK | DOCA Open Source |
| ---- | ------------- | ---------------- |
| Verbs CPU control path | Closed-source shared library | Open-source C++ files |
| GPUNetIO CPU control path | Closed-source shared library | Open-source C++ files |
| GPUNetIO GPU data path for RDMA Verbs one-sided | Yes | Yes |
| GPUNetIO GPU data path for RDMA Verbs two-sided | Yes | No |
| GPUNetIO GPU data path for Ethernet | Yes | No |
| GPUNetIO GPU data path for DMA | Yes | No |

The **Full SDK** is more comprehensive and includes additional features that are not part of this open-source release.
It is important to note, however, that the CUDA header files for the GPUNetIO Verbs data path are identical between the open-source and full versions.

## Goals

The overarching goal of DOCA GPUNetIO (both Open Source and Full) is to consolidate multiple GDAKI implementations into a unified driver and library with consistent host- and device-side interfaces. This common foundation can be shared across current and future consumers of GDAKI technology such as [NVSHMEM](https://docs.nvidia.com/nvshmem/api/using.html#using-the-nvshmem-infiniband-gpudirect-async-transport), [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html) or [UCX](https://github.com/openucx/ucx).
This approach promotes knowledge sharing while reducing the engineering effort required for long-term maintenance.


## Core Features

**CPU control path:**
- Interfaces to create and manage completion queues (CQs) and queue pairs (QPs) in CPU/GPU memory.
- Support for connecting QPs over Reliable Connection (RC) transport.
- Move CQ/QP resources between CPU and GPU memory.
- Compatibility with standard `verbs` resources (MRs, PDs, context, device attributes, etc.).
- Possibility to use DOCA SDK restricted features via internal `dlopen`-based dynamic linking.

**GPU data path:**
- Device-side APIs to post direct work requests (WRs) and poll completion responses (CQEs).
- Directly ring NIC doorbells from the GPU (update registers).

For a deep dive into features, see the official [DOCA GPUNetIO documentation](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) and [DOCA Verbs documentation](https://docs.nvidia.com/doca/sdk/doca+verbs/index.html).


## Usage

To enable GDAKI technology with the DOCA API, an application must be divided into two phases.
A CPU control path phase, which initializes devices, allocates memory, and performs other setup tasks.
A GPU data path phase, where a CUDA kernel is launched and GPUNetIO CUDA functions are used within it.

### Control Path Workflow
1. Open an RDMA device context: `ibv_open_device`.
2. Allocate a PD: `ibv_alloc_pd`.
3. Register memory regions: `ibv_reg_mr`.
4. Create a GPUNetIO handler: `doca_gpu_create`.
5. Create CQ and QP using `doca_verbs_*` functions.
6. Connect QPs with remote peers using `doca_verbs_qp_modify`.
7. Export QPs and CQs to GPU memory using: `doca_gpu_verbs_export_cq` and `doca_gpu_verbs_export_qp`

### Data Path Workflow
1. Launch a GPU kernel
2. Post work requests using:
  - High-level API in CUDA header files `doca_gpunetio_dev_verbs_onesided.cuh` and `doca_gpunetio_dev_verbs_counter.cuh` starting with `doca_gpu_dev_verbs_*`
  - Low-level API (advanced users) in CUDA header files like `doca_gpunetio_dev_verbs_qp.cuh` and `doca_gpunetio_dev_verbs_cq.cuh` like `doca_gpu_dev_verbs_wqe_prepare_*`, `doca_gpu_dev_verbs_submit`
3. Poll completions with: `doca_gpu_dev_verbs_poll_cq_*`

> Mixing high- and low-level APIs is **not recommended**.

#### CPU-assisted GDAKI

Some systems do not support direct NIC doorbell ringing from GPU SMs. In this case, a CUDA kernel can post WQEs and poll CQEs in GPU memory, but it cannot update the network card registers.
In such scenarios, DOCA GPUNetIO GDAKI can still be used by enabling CPU-assisted mode: the GPU notifies a CPU thread, which rings the NIC doorbell on its behalf. This mode provides a reliable fallback (with lower performance) and requires a CPU thread to periodically call `doca_gpu_verbs_cpu_proxy_progress()`.

## Build

To build the host-side library `libdoca_gpunetio.so`:

```bash
cd doca-gpunetio
make -j
```

The output is a `lib` directory containing the shared library.

It is also possible to specify CUDA arch at build time, as well as install library and examples in a specific directory.
An example to build and install library and examples for Hopper GPU with `sm_90` on a given prefix:

```bash
make install install_examples PREFIX=/path/to/directory/install CUDA_ARCH=90
```

## Enable logs

Logs are managed by macro `DOCA_LOG`, relying on `syslog` with different log levels:

0. EMERG
1. ALERT
2. CRIT
3. ERR
4. WARNING
5. NOTICE
6. INFO
7. DEBUG

By default, the `EMERG` level (0) is set. To print the `DOCA_LOG` with higher level, please set the `DOCA_GPUNETIO_LOG`
environment variable to the right level number.

## Enable SDK mode

To access DOCA SDK restricted functionality from GPUNetIO open source, CPU functions now use internal `dlopen`-based dynamic linking.
If the environment variable `DOCA_SDK_LIB_PATH` is set to a valid [DOCA SDK](https://developer.nvidia.com/doca-downloads) library installation directory (typically `/opt/mellanox/doca/libs/x86_64-linux-gnu` for x86 systems), GPUNetIO open dynamically loads the DOCA SDK functions and uses them instead of the standalone open-source implementation.

An example command line to enable the SDK mode:

```
$ DOCA_GPUNETIO_LOG=6 DOCA_SDK_LIB_PATH=/opt/mellanox/doca/lib/x86_64-linux-gnu ./gpunetio_verbs_write_lat -d mlx5_0 -g 8a:00.0
Wed Apr  1 10:42:00 2026 [INFO] [examples/verbs_common.cpp]: 255: create_verbs_resources(): Setting GPU device 0 at 8a:00.0
Wed Apr  1 10:42:00 2026 [WARNING] [src/doca_gpunetio_sdk_wrapper.cpp]: 202: doca_gpu_sdk_wrapper_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64-linux-gnu. DOCA SDK is in use
Wed Apr  1 10:42:00 2026 [INFO] [src/doca_gpunetio.cpp]: 120: doca_gpu_create(): Use DOCA GPUNetIO SDK
Wed Apr  1 10:42:00 2026 [WARNING] [src/doca_verbs_dev_sdk_wrapper.cpp]: 164: doca_verbs_sdk_wrapper_dev_open_from_pd(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64-linux-gnu. DOCA SDK is in use
Wed Apr  1 10:42:00 2026 [INFO] [src/doca_verbs_dev.cpp]: 83: doca_verbs_dev_open(): Use DOCA Verbs Dev SDK
Wed Apr  1 10:42:00 2026 [WARNING] [src/doca_verbs_qp_sdk_wrapper.cpp]: 1430: doca_verbs_sdk_wrapper_ah_attr_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64-linux-gnu. DOCA SDK is in use
Wed Apr  1 10:42:00 2026 [INFO] [src/doca_verbs_qp.cpp]: 2902: doca_verbs_ah_attr_create(): Use DOCA Verbs AH Attr SDK
Wed Apr  1 10:42:00 2026 [WARNING] [src/doca_verbs_cq_sdk_wrapper.cpp]: 246: doca_verbs_sdk_wrapper_cq_attr_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64-linux-gnu. DOCA SDK is in use
Wed Apr  1 10:42:00 2026 [INFO] [src/doca_verbs_cq.cpp]: 327: doca_verbs_cq_attr_create(): Use DOCA Verbs CQ Attr SDK
Wed Apr  1 10:42:00 2026 [WARNING] [src/doca_verbs_umem_sdk_wrapper.cpp]: 204: doca_verbs_sdk_wrapper_umem_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64-linux-gnu. DOCA SDK is in use
Wed Apr  1 10:42:00 2026 [INFO] [src/doca_verbs_umem.cpp]: 150: doca_verbs_umem_create(): Use DOCA Verbs UMEM SDK
Wed Apr  1 10:42:00 2026 [WARNING] [src/doca_verbs_cq_sdk_wrapper.cpp]: 449: doca_verbs_sdk_wrapper_cq_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64-linux-gnu. DOCA SDK is in use
Wed Apr  1 10:42:00 2026 [INFO] [src/doca_verbs_cq.cpp]: 589: doca_verbs_cq_create(): Use DOCA Verbs CQ SDK
Wed Apr  1 10:42:00 2026 [WARNING] [src/doca_verbs_uar_sdk_wrapper.cpp]: 160: doca_verbs_sdk_wrapper_uar_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64-linux-gnu. DOCA SDK is in use
Wed Apr  1 10:42:00 2026 [INFO] [src/doca_verbs_uar.cpp]: 164: doca_verbs_uar_create(): Use DOCA Verbs UAR SDK
Wed Apr  1 10:42:00 2026 [WARNING] [src/doca_verbs_qp_sdk_wrapper.cpp]: 574: doca_verbs_sdk_wrapper_qp_init_attr_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64-linux-gnu. DOCA SDK is in use
....
```
Prints like `Use DOCA GPUNetIO SDK`, `Use DOCA Verbs Dev SDK`, etc.. indicate the path specified in `DOCA_SDK_LIB_PATH` correcly points to a DOCA SDK library installation directory.

Conversely, if something is wrong with the path specified in `DOCA_SDK_LIB_PATH`, the output changes:

```
$ DOCA_GPUNETIO_LOG=6 DOCA_SDK_LIB_PATH=/opt/mellanox/doca/lib/x86_64 ./gpunetio_verbs_write_lat -d mlx5_0 -g 8a:00.0
Wed Apr  1 10:41:52 2026 [INFO] [examples/verbs_common.cpp]: 255: create_verbs_resources(): Setting GPU device 0 at 8a:00.0
Wed Apr  1 10:41:52 2026 [ERR] [src/doca_gpunetio_sdk_wrapper.cpp]: 110: doca_gpunetio_sdk_wrapper_init(): Failed to find libdoca_common.so library /opt/mellanox/doca/lib/x86_64/libdoca_common.so (DOCA_SDK_LIB_PATH=/opt/mellanox/doca/lib/x86_64)
Wed Apr  1 10:41:52 2026 [WARNING] [src/doca_gpunetio_sdk_wrapper.cpp]: 193: doca_gpu_sdk_wrapper_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64, but DOCA SDK libraries not found. DOCA SDK is not in use
Wed Apr  1 10:41:52 2026 [INFO] [src/doca_gpunetio.cpp]: 131: doca_gpu_create(): Use DOCA GPUNetIO open
Wed Apr  1 10:41:52 2026 [ERR] [src/doca_verbs_dev_sdk_wrapper.cpp]: 87: doca_verbs_sdk_wrapper_init(): Failed to find libdoca_common.so library /opt/mellanox/doca/lib/x86_64/libdoca_common.so (DOCA_SDK_LIB_PATH=/opt/mellanox/doca/lib/x86_64)
Wed Apr  1 10:41:52 2026 [WARNING] [src/doca_verbs_dev_sdk_wrapper.cpp]: 154: doca_verbs_sdk_wrapper_dev_open_from_pd(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64, but DOCA SDK libraries not found. DOCA SDK is not in use
Wed Apr  1 10:41:52 2026 [INFO] [src/doca_verbs_dev.cpp]: 94: doca_verbs_dev_open(): Use DOCA Verbs Dev open
Wed Apr  1 10:41:52 2026 [INFO] [src/doca_verbs_dev.cpp]: 106: doca_verbs_dev_open(): doca_verbs_dev_open=0x564b462a7e00 was created
Wed Apr  1 10:41:52 2026 [ERR] [src/doca_verbs_qp_sdk_wrapper.cpp]: 295: doca_verbs_sdk_wrapper_init(): Failed to find libdoca_common.so library /opt/mellanox/doca/lib/x86_64/libdoca_common.so (DOCA_SDK_LIB_PATH=/opt/mellanox/doca/lib/x86_64)
Wed Apr  1 10:41:52 2026 [WARNING] [src/doca_verbs_qp_sdk_wrapper.cpp]: 1404: doca_verbs_sdk_wrapper_ah_attr_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64, but DOCA SDK libraries not found. DOCA SDK is not in use
Wed Apr  1 10:41:52 2026 [INFO] [src/doca_verbs_qp.cpp]: 2913: doca_verbs_ah_attr_create(): Use DOCA Verbs AH Attr open
Wed Apr  1 10:41:52 2026 [INFO] [src/doca_verbs_qp.cpp]: 2919: doca_verbs_ah_attr_create(): doca_verbs_verbs_ah_open=0x564b462a7e20 was created
Wed Apr  1 10:41:52 2026 [ERR] [src/doca_verbs_cq_sdk_wrapper.cpp]: 133: doca_verbs_sdk_wrapper_init(): Failed to find libdoca_common.so library /opt/mellanox/doca/lib/x86_64/libdoca_common.so (DOCA_SDK_LIB_PATH=/opt/mellanox/doca/lib/x86_64)
Wed Apr  1 10:41:52 2026 [WARNING] [src/doca_verbs_cq_sdk_wrapper.cpp]: 237: doca_verbs_sdk_wrapper_cq_attr_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64, but DOCA SDK libraries not found. DOCA SDK is not in use
Wed Apr  1 10:41:52 2026 [INFO] [src/doca_verbs_cq.cpp]: 338: doca_verbs_cq_attr_create(): Use DOCA Verbs CQ Attr open
Wed Apr  1 10:41:52 2026 [INFO] [src/doca_verbs_cq.cpp]: 344: doca_verbs_cq_attr_create(): doca_verbs_cq_attr_open=0x564b462a7eb0 was created
Wed Apr  1 10:41:52 2026 [ERR] [src/doca_verbs_umem_sdk_wrapper.cpp]: 106: doca_verbs_sdk_wrapper_init(): Failed to find libdoca_gpunetio.so library /opt/mellanox/doca/lib/x86_64/libdoca_gpunetio.so (DOCA_SDK_LIB_PATH=/opt/mellanox/doca/lib/x86_64)
Wed Apr  1 10:41:52 2026 [WARNING] [src/doca_verbs_umem_sdk_wrapper.cpp]: 186: doca_verbs_sdk_wrapper_umem_create(): Env var DOCA_SDK_LIB_PATH set to /opt/mellanox/doca/lib/x86_64, but DOCA SDK libraries not found. DOCA SDK is not in use
Wed Apr  1 10:41:52 2026 [INFO] [src/doca_verbs_umem.cpp]: 161: doca_verbs_umem_create(): Use DOCA Verbs UMEM open
...
```

## Examples

Two examples are included to demonstrate usage and measure performance.
Make sure to build `libdoca_gpunetio.so` **before compiling examples**.

> All examples require both a client and a server running on network-connected machines.
> GPU timers can be enabled per operation by setting `#define KERNEL_DEBUG_TIMES 1` (useful for debugging, not recommended for performance testing).

Additional samples are available in the [NVIDIA DOCA Full Samples repository](https://github.com/NVIDIA-DOCA/doca-samples).

The following command lines assume samples are running on systems where GPU is at PCIe address `8A:00.0` and NIC interface is `mlx5_0`.

### Example 1: `gpunetio_verbs_put_bw`

This example is a GDAKI perftest `ib_write_bw`-like benchmark where client launches a CUDA kernel to execute the high-level `doca_gpu_dev_verbs_put` operation.
Server doesn't launch any CUDA kernel: upon user typing ctrl+c, server validate data received from client.

**Run (server):**
```bash
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/doca-gpunetio/lib DOCA_GPUNETIO_LOG=6 ./gpunetio_verbs_put_bw -g 8A:00.0 -d mlx5_0
```

**Run (client):**
```bash
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/doca-gpunetio/lib DOCA_GPUNETIO_LOG=6 ./gpunetio_verbs_put_bw -g 8A:00.0 -d mlx5_0 -c 192.168.1.64
```

Modes:
- **CUDA Thread execution scope** (default).
- **CUDA Warp execution scope**: add `-e 1`.
- **NIC handler type**: add `-p <nic_handler value>` where 0: AUTO (default), 1: CPU Proxy, 2: GPU SM DB.

Validation success message (server):
```
Validation successful! Data received correctly from client.
```

### Example 2: `gpunetio_verbs_write_lat`

This example is a GDAKI perftest `ib_write_lat`-like benchmark where Client and server both launch CUDA kernels using low-level APIs.

**Run (server):**
```bash
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/doca-gpunetio/lib DOCA_GPUNETIO_LOG=6 ./gpunetio_verbs_write_lat -g 8A:00.0 -d mlx5_0 -p 2
```

**Run (client):**
```bash
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/doca-gpunetio/lib DOCA_GPUNETIO_LOG=6 ./gpunetio_verbs_write_lat -g 8A:00.0 -d mlx5_0 -p 2 -c <server_ip_address>
```

Modes:
- **NIC handler type**: add `-p <nic_handler value>` where 0: AUTO (default), 1: CPU Proxy, 2: GPU SM DB, 6: GPU SM BlueFlame.

### Example 3: `gpunetio_verbs_write_bw`

This example is a GDAKI perftest `ib_write_bw`-like benchmark where Client and server both launch CUDA kernels using low-level APIs.

**Run (server):**
```bash
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/doca-gpunetio/lib DOCA_GPUNETIO_LOG=6 ./gpunetio_verbs_write_bw -g 8A:00.0 -d mlx5_0
```

**Run (client):**
```bash
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/doca-gpunetio/lib DOCA_GPUNETIO_LOG=6 ./gpunetio_verbs_write_bw -g 8A:00.0 -d mlx5_0 -c <server_ip_address>
```

Modes:
- **NIC handler type**: add `-p <nic_handler value>` where 0: AUTO (default), 1: CPU Proxy, 2: GPU SM DB.

GPU SM BlueFlame is not supported in this example.

Validation success message (server):
```
Validation successful! Data received correctly from client.
```

## Acknowledgments

If you use this software in your work, please cite the official [DOCA GPUNetIO documentation](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html).

## Contributing

This project is developed internally and released as open source.
We currently do **not accept external contributions**.


## Troubleshooting & Feedback

We appreciate community discussion and feedback in support of DOCA GPUNetIO Open users and developers. We ask that users:

- Review the [DOCA SDK Programming Guide](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) for system configuration, technology explaination, API, etc...
- Ask questions on the [NVIDIA DOCA Support Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370).
- Report issues on the [GitHub Issues board](https://github.com/NVIDIA-DOCA/gpunetio/issues).

## License

See the [LICENSE.txt](LICENSE.txt) file.
