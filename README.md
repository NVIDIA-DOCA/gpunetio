# DOCA GPUNetIO Open Source

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE.txt)
[![CUDA](https://img.shields.io/badge/CUDA-12.2%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Contributions](https://img.shields.io/badge/Contributions-Not%20Accepted-red.svg)]()

This repository provides an open-source version of the [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) and [DOCA Verbs](https://docs.nvidia.com/doca/sdk/doca+verbs/index.html) libraries. The features included here are limited to enabling **GPUDirect Async Kernel-Initiated (GDAKI)** network communication technology over RDMA protocols (InfiniBand and RoCE) using a DOCA-like API in an open-source environment.

## Open vs Full

The table below highlights the key differences between this DOCA GPUNetIO open-source project and the full DOCA GPUNetIO SDK:

| Item | DOCA Full SDK | DOCA Open Source |
| ---- | ------------- | ---------------- |
| Verbs CPU control path | Close-source shared library | Open-source C++ files |
| GPUNetIO CPU control path | Close-source shared library | Open-source C++ files |
| GPUNetIO GPU data path for RDMA Verbs one-sided | Yes | Yes |
| GPUNetIO GPU data path for RDMA Verbs two-sided | Yes | No |
| GPUNetIO GPU data path for Ethernet | Yes | No |
| GPUNetIO GPU data path for DMA | Yes | No |

The **Full SDK** is more comprehensive and includes additional features that are not part of this open-source release.
It is important to note, however, that the CUDA header files for the GPUNetIO Verbs data path are identical between the open-source and full versions.

## Goals

The overarching goal of DOCA GPUNetIO (both Open Source and Full) is to consolidate multiple GDAKI implementations into a unified driver and library with consistent host- and device-side interfaces. This common foundation can be shared across current and future consumers of GDAKI technology such as [NVSHMEM](https://docs.nvidia.com/nvshmem/api/using.html#using-the-nvshmem-infiniband-gpudirect-async-transport), [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html), and GPUDirect.
This approach promotes knowledge sharing while reducing the engineering effort required for long-term maintenance.


## Core Features

**CPU control path:**
- Interfaces to create and manage completion queues (CQs) and queue pairs (QPs) in CPU/GPU memory.
- Support for connecting QPs over Reliable Connection (RC) transport.
- Move CQ/QP resources between CPU and GPU memory.
- Compatibility with standard `verbs` resources (MRs, PDs, context, device attributes, etc.).

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

This generates a `lib` directory containing the shared library.

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

**Build:**
```bash
cd doca-gpunetio/examples/gpunetio_verbs_put_bw
make -j
```

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
- **CPU proxy mode**: add `-p 1`.

Validation success message (server):
```
Validation successful! Data received correctly from client.
```


### Example 2: `gpunetio_verbs_write_lat`

This example is a GDAKI perftest `ib_write_lat`-like benchmark where Client and server both launch CUDA kernels using low-level APIs.

**Build:**
```bash
cd doca-gpunetio/examples/gpunetio_verbs_write_lat
make -j
```

**Run (server):**
```bash
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/doca-gpunetio/lib DOCA_GPUNETIO_LOG=6 ./gpunetio_verbs_write_lat -g 8A:00.0 -d mlx5_0
```

**Run (client):**
```bash
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/doca-gpunetio/lib DOCA_GPUNETIO_LOG=6 ./gpunetio_verbs_write_lat -g 8A:00.0 -d mlx5_0 -c <server_ip_address>
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
