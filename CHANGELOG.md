# Changelog

## [5.0.0]

### Added

- Added GPU/NIC capability discovery APIs for GPU-memory UMEM, host-memory UMEM, and GPU SM doorbell support: `doca_gpu_nic_cap_is_gpu_mem_umem_supported`, `doca_gpu_nic_cap_is_host_mem_umem_supported`, and `doca_gpu_nic_cap_is_nic_handler_gpu_sm_db_supported`, with open-source and DOCA SDK implementations.
- Added `DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_PREFER_UAR_SHARING` to prefer shared noncached UARs when creating high-level QPs and QP groups.
- Added open-source QP data-placement ordering semantic support for `IBTA`, `OOO_RW`, and `OOO_ALL`, with device capability validation.
- Added `doca_verbs_query_global_traffic_class` to query a port's global RoCE traffic class from sysfs using a `doca_dev` in open-source or DOCA SDK mode.
- Added receive-queue foundations for RDMA Write with Immediate, including the high-level `rq_nwqe` attribute, host receive CQs, receive doorbell initialization, and receive-state reset. Receive queues are supported by every high-level QP creation API.
- Added receive-queue support to `doca_gpu_verbs_create_qp_list_hl` and `doca_gpu_verbs_create_qp_group_list_hl`. With `rq_nwqe > 0`, every list QP, or every group's main QP, gets a receive queue and a host receive CQ. The receive CQs share two host-memory blocks, exposed as the new `rq_cq_umem` and `rq_cq_dbr_umem` members appended to `struct doca_gpu_verbs_qp_list_hl` and `struct doca_gpu_verbs_qp_group_list_hl`. With a GPU NIC handler, the shared doorbell-record block becomes GDRCopy-mapped GPU memory so the CPU can write receive doorbells.
- Added receive-queue support to `doca_gpu_verbs_create_qp_group_hl`. With `rq_nwqe > 0`, the group's main QP gets a receive queue and a host receive CQ; the companion QP does not.
- Added rank-local signal-map lifecycle APIs in `doca_gpunetio_signal.h` to create, register, unregister, reset, and destroy mappings from signal IDs to application-owned 8-byte GPU slots using GDRCopy, plus configurable immediate-data packing macros.
- Added the `MLX5_CMD_OP_QUERY_Q_COUNTER` opcode and the `mlx5_ifc_alloc_q_counter_in_bits`, `mlx5_ifc_query_q_counter_in_bits`, and `mlx5_ifc_query_q_counter_out_bits` layouts to `host/mlx5_ifc.h`, for allocating and querying Q counter sets through DevX with `DEVX_SET`/`DEVX_GET`.
- Added Data Direct to the examples (`-z 1`): GPU buffers are registered with `MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT`, QPs are created with `DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT`, and the put/write bandwidth example servers keep their buffers in GPU memory. Requires CUDA 12.8 or newer and a Data Direct-capable libmlx5, which `scripts/configure` detects as `DOCA_GPUNETIO_HAVE_DATA_DIRECT` in `doca_gpunetio_config.h`.
- Added NIC handlers 10 (GPU SM no DBR) and 17 (CPU proxy free flow) to the examples (`-p`). Handler 10 uses the NIC hardware no-DBR mode when available, otherwise the software-emulated one.
- Added the `gpunetio_verbs_get_bw` example, an RDMA Read bandwidth benchmark using the high-level `doca_gpu_dev_verbs_get` and `doca_gpu_dev_verbs_get_wait` operations. The client validates the data it reads from the server.
- Added the `gpunetio_verbs_put_signal_bw` and `gpunetio_verbs_put_signal_lat` examples, bandwidth and ping-pong latency benchmarks using the high-level `doca_gpu_dev_verbs_put_signal` operation. The bandwidth example server validates the data and logs the sum of its signal values. In warp scope (`-e 1`) only lane 0 of each warp signals.
- Added the `doca_gpu_dev_verbs_put_signal_warp` implementation: each lane posts its RDMA Write, and lane 0 posts a single RDMA Atomic Fetch-Add for the warp, with its own signal arguments. Every lane returns the ticket of that RDMA Atomic, so that waiting on any lane's ticket covers the whole put with signal. A lane with a size of 0 posts a NOP instead of its RDMA Write, as in `doca_gpu_dev_verbs_put_signal_thread`.
- Added `DOCA_GPUNETIO_VERBS_USE_CCC_ATOMIC_ADD` to enable(1)/disable(0) the use of `cuda::atomic_ref` instead of `atomicAdd` in `doca_gpu_dev_verbs_atomic_add` functions. `atomicAdd` should result in a more efficient SASS code than `cuda::atomic_ref`, thus `DOCA_GPUNETIO_VERBS_USE_CCC_ATOMIC_ADD` is disabled by default.

### Changed

- Renamed the LAG TX port affinity capability getters to `doca_verbs_device_attr_get_is_lag_tx_port_affinity_supported`, `doca_verbs_device_attr_get_is_init2_lag_tx_port_affinity_supported`, and `doca_verbs_device_attr_get_is_rts2rts_lag_tx_port_affinity_supported` to match the DOCA SDK APIs, and added SDK wrappers for the capability queries and QP attribute setter.
- Extended `doca_gpu_verbs_export_qp` with receive-CQ and CPU-accessible doorbell-record parameters (`cq_rq` and `dbr_cpu_ptr`). Pass `NULL` for both when the QP has no receive queue.
- Bumped the library version and minimum compatible host-code version to 5.0.0 for the host API changes.
- `doca_gpu_verbs_destroy_qp_group_hl`, `doca_gpu_verbs_destroy_qp_list_hl`, and `doca_gpu_verbs_destroy_qp_group_list_hl` now return `DOCA_ERROR_IN_USE` and destroy nothing while any of their QPs is monitored by a service.
- Host-memory shared UMEM blocks used by the list creation APIs are now page aligned.
- Updated dynamic loading to try versioned DOCA SDK, GDRCopy, and mlx5 library names before falling back to unversioned names.
- In `doca_gpu_dev_common_mark_wqes_ready`, removed the final GPU scope fence_acquire in the `DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU` sharing mode.
- Updated the examples to validate their command-line values and to check, when the peers connect, that both run the same example with the same number of iterations and CUDA threads.
- Updated the put/write bandwidth example servers to validate the received data once the client reports that all its runs have completed, instead of waiting for ctrl+c. A client that closes the connection without reporting it, for example on an error, fails the server.
- Updated the put bandwidth example kernel to wait for the last put only, instead of polling the completion of every put.
- QP attribute `log_ack_req_freq` set to 8 instead of 0

### Fixed

- Fixed `doca_gpu_dev_verbs_get_warp` with `DOCA_GPUNETIO_VERBS_MCST_ENABLED` to return the ticket of the WQE Dump in every lane. Lanes other than 0 returned their own RDMA Read, so waiting on their ticket could return before the data was guaranteed visible.
- Fixed high-level UMEM cleanup to track system allocations explicitly and use the correct deallocator for host CQ memory.
- Fixed host receive-CQ UMEM registration to allow NIC writes.
- Fixed UAR creation error propagation when allocation attempts fail.
- Fixed `doca_gpu_verbs_create_qp_group_hl` leaking both QPs when exporting either one fails.
- Fixed put/write bandwidth examples to propagate server data-validation failures.
- Fixed incorrect access to `gpu_dev->open`. The bug caused an undefined behavior when using the DOCA SDK mode.
- Fixed the examples to start the CPU proxy thread whenever the QP needs one, including when the AUTO NIC handler resolves to CPU proxy and with the software-emulated no-DBR mode of the write latency example (`-r 2`).
- Fixed the write latency example sequence numbers, which could match the last value of the warmup and let a peer run ahead, and the alignment of its BlueFlame shared-memory WQE.
- Fixed the write bandwidth example to fail on a CQE error reported by the completion channel, and to accept `-l`.
- Fixed the write latency example to reject BlueFlame (`-p 6`) with the software-emulated no-DBR mode (`-r 2`) at startup, instead of failing in the kernel.
- Fixed the examples resource handling: partial OOB sends and receives, leaked dmabuf file descriptors and listening socket, use of the IB device list after freeing it, IB device name prefix matching, failures reported as success, and cleanup stopping at the first error.

## [4.1.0]

### Added

- Added LAG TX port affinity support for open-source DOCA Verbs QPs, including `DOCA_VERBS_QP_ATTR_LAG_TX_PORT_AFFINITY`, `doca_verbs_qp_attr_set_lag_tx_port_affinity`, and device capability query APIs. The QP attribute is not supported in DOCA SDK mode.
- Added `doca_gpu_dev_verbs_signal_warp` implementation

### Fixed

- Fixed missing endianness conversion for compare data and compare masks in 32-bit and 64-bit extended atomic compare-and-swap WQEs.

## [4.0.1]

### Changed

- Updated all UMEM registrations to use `mlx5dv_devx_umem_reg_ex` and explicitly set `pgsz_bitmap` to the system page size, including for non-DMA-BUF memory.
- Moved DOCA SDK logger in sdk_wrapper_init

### Fixed

- Return `DOCA_ERROR_NOT_SUPPORTED` when DMA-BUF UMEM registration is requested with `DOCA_GPUNETIO_HAVE_MLX5DV_UMEM_DMABUF` disabled, and improved UMEM registration error logging.
- DBR-less UAR address calculation
- Removed unused files

## [4.0.0]

### Added

- Added GDA-KI Free Flow support, including the `DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY_FREE_FLOW` NIC handler mode.
- Added high-level batched QP and QP group creation/destruction APIs (`doca_gpu_verbs_create_qp_list_hl`, `doca_gpu_verbs_destroy_qp_list_hl`, `doca_gpu_verbs_create_qp_group_list_hl`, and `doca_gpu_verbs_destroy_qp_group_list_hl`) backed by shared control-buffer slabs.
- Added CQ type selection for 64B GPU CQs, collapsed GPU CQs, and collapsed host-memory CQs, plus device-side polling support for the new CQ modes.
- Added DEVX-backed asynchronous CQ error event APIs: `doca_verbs_comp_channel_create`, `doca_verbs_cq_attr_set_comp_channel`, `doca_verbs_get_cq_comp_channel_event`, `doca_verbs_ack_cq_events`, and `doca_verbs_comp_channel_destroy`. Works with oppen source and DOCA SDK 3.5 or newer
- Added Data Direct support through `DOCA_GPU_MEM_TYPE_GPU_CPU_DATA_DIRECT` and the high-level QP initialization flag `DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT`.
- Added DOCA SDK congestion-control group support, including CC group wrappers and `DOCA_VERBS_QP_ATTR_CC_GROUP`.
- Added device-side MMIO store-release helpers and made cached `__ldg` loads opt-in through `DOCA_GPUNETIO_VERBS_USE_LDG`.

### Changed

- Extended `doca_gpu_verbs_export_qp` and high-level QP creation paths to pass CQ type and Data Direct configuration into exported GPU QPs.
- Updated CQ and QP external UMEM handling to support UMEM offsets and separate CQ doorbell-record UMEM, enabling shared-slab suballocation.
- Updated CQ doorbell handling to use the CQ doorbell register offset from the UAR base address.
- Updated multi-QP export handling to allow holes in QP arrays.
- Updated device-side completion polling and WQE preparation paths to account for collapsed and host-memory CQ modes.

### Removed

- Removed the direct `<infiniband/verbs.h>` include from the DOCA Verbs QP SDK wrapper.

## [3.0.0]

### Added

- Introduced the dynamic loading of DOCA SDK functions: to access DOCA SDK restricted functionality from GPUNetIO open source, CPU functions now use internal `dlopen`-based dynamic linking.
- If the environment variable `DOCA_SDK_LIB_PATH` is set to a valid [DOCA SDK](https://developer.nvidia.com/doca-downloads) library installation directory (typically `/opt/mellanox/doca/libs/x86_64-linux-gnu` for x86 systems), GPUNetIO open dynamically loads the DOCA SDK functions and uses them instead of the standalone open-source implementation.
- If the environment variable `DOCA_SDK_LIB_PATH` is not set or points to an invalid DOCA SDK library installation directory, GPUNetIO open uses the standalone open-source implementation for all CPU functions. In this case, DOCA SDK closed-source features cannot be used.
- N.B. It is the users’ responsibility to properly install the [DOCA SDK](https://developer.nvidia.com/doca-downloads) on the system if restricted features are required (e.g., ordering semantic).
- The transition from open to SDK mode breaks backward compatibility of CPU functions, because function signatures and data structure types had to be updated. Since backward compatibility is broken, the GPUNetIO open version has been bumped to 3.0.0.

### Changed

- To enable atomic operations on a QP, the flag `DOCA_VERBS_QP_ATTR_ATOMIC_MODE` is required in the `attr_mask` parameter passed to `doca_verbs_qp_modify`.
- The function `doca_verbs_qp_attr_set_allow_remote_atomic` has been renamed to `doca_verbs_qp_attr_set_atomic_mode`.
- To improve compatibility between the SDK and open implementations, a new object, `doca_dev`, has been introduced and is now required for some operations. An example can be found in the `examples/verbs_common.cpp` file: open the device (`open_ib_device`), create a PD (`ibv_alloc_pd`), and then create a `doca_dev` (`doca_verbs_dev_open(pd, &net_dev)`).
- To achieve better performance, the default MTU set in the examples is now 4K (file `examples/verbs_common.cpp`, function `doca_verbs_qp_attr_set_path_mtu`). Please ensure your network interface is correctly set to, at least, 4K MTU. If not, you can change the value to 1K in the examples code.

### Removed

- To simplify the CPU-side code, some unused functions `doca_verbs_qp_init_attr_get_*` and `doca_verbs_qp_attr_get_*` have been removed. If needed, will be re-introduced on-demand.

## [2.0.1]

### Fixed

- Minor fix to the dmabuf_fd initialization value in file `doca_gpunetio_high_level.cpp`.

## [2.0.0]

### Added

- Get, Get Wait and Get Counter device APIs (`doca_gpu_dev_verbs_get`, `doca_gpu_dev_verbs_get_wait`, `doca_gpu_dev_verbs_get_counter`)
- Reliable doorbell record (DBREC) hardware supported (`DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW`) for ConnectX-8 NICs or software emulation mode (`DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED`) to support hardware (ConnectX-7 and older) without native no-DBREC capability.
- `DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR` to enable the reliable doorbell record feature in the GPU data path functions.
- `DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK` and `DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING` GPU code optimization flags in `doca_gpu_dev_verbs_gpu_code_opt`.
- `DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI` GPU code optimization flag for updating the producer index in CPU proxy mode if needed.
- QP reset feature via `doca_gpu_verbs_reset_tracking_and_memory` to reset QP tracking state and memory.
- Version querying and compatibility checking APIs: `doca_gpu_verbs_get_library_version`, `doca_gpu_verbs_check_device_code_compatibility`, `doca_gpu_verbs_check_host_code_compatibility`.
- Version macros (`DOCA_GPUNETIO_VERSION_MAJOR/MINOR/PATCH`) and minimum compatibility version definitions.
- Multi-QP export/unexport APIs (`doca_gpu_verbs_export_multi_qps_dev`, `doca_gpu_verbs_unexport_multi_qps_dev`) for batched GPU export of multiple QPs.
- `DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD` synchronization scope for cases where no memory fence is needed.
- Blocking mode enum (`doca_gpu_dev_verbs_blocking_mode`) for blocking vs. non-blocking execution.
- Multicast mode enum (`doca_gpu_dev_verbs_mcst_mode`) for controlling dump behavior on Get/Recv.
- WQE ready mode enum (`doca_gpu_dev_verbs_qp_ready_mode`) to select between `ATOMIC_CAS` and `LD_ST` strategies.
- Atomic extended operation support (4-byte and 8-byte) with `DOCA_GPUNETIO_4_BYTE_ATOMIC_EXT_OPMOD` and `DOCA_GPUNETIO_8_BYTE_ATOMIC_EXT_OPMOD`.
- QP attributes for max outstanding RDMA Read/Atomic operations (`DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC`, `DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC`).
- Collapsed CQ attribute (`doca_verbs_cq_attr_set_cq_collapsed`).
- Emulate no-DBREC ext flag for QP init attributes (`doca_verbs_qp_init_attr_set_emulate_no_dbr_ext`).
- `make install` and `make install_example` Makefile targets.

### Changed

- `doca_gpu_verbs_export_qp` now requires an additional `send_dbr_mode_ext` parameter to specify the send DBREC mode, if needed.
- `doca_gpu_verbs_cpu_proxy_progress` now accepts an `out_progressed` output parameter indicating whether the QP was progressed.
- Device-side `put`, `p` (put-inline), `putSignal`, `get`, and related operations now accept an optional `code_opt` runtime parameter for GPU code optimization (previously a template parameter).
- NIC handler enum values are now defined as composable flag bitmasks instead of sequential integers.
- In one-sided operations, the synchronization scope in `doca_gpu_dev_verbs_submit` is now automatically derived from the `resource_sharing_mode`, removing a redundant `membar` when `mark_wqes_ready` has already issued one.
- Default WQE ready mode uses `LD/ST` for `RESOURCE_SHARING_MODE_CTA` and `ATOMIC_CAS` for `RESOURCE_SHARING_MODE_GPU`

### Fixed

- BlueFlame (BF) update in TMA copy path.
