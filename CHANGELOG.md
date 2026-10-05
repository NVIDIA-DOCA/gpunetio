# Changelog

## [5.0.0]

### Added

- Added GPU/NIC capability discovery APIs for GPU-memory UMEM, host-memory UMEM, and GPU SM doorbell support: `doca_gpu_nic_cap_is_gpu_mem_umem_supported`, `doca_gpu_nic_cap_is_host_mem_umem_supported`, and `doca_gpu_nic_cap_is_nic_handler_gpu_sm_db_supported`, with open-source and DOCA SDK implementations.
- Added `DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_PREFER_UAR_SHARING` to prefer shared noncached UARs when creating high-level QPs and QP groups.
- Added open-source QP data-placement ordering semantic support for `IBTA`, `OOO_RW`, and `OOO_ALL`, with device capability validation.
- Added `doca_verbs_query_global_traffic_class` to query a port's global RoCE traffic class from sysfs using a `doca_dev` in open-source or DOCA SDK mode.
- Added receive-queue foundations for RDMA Write with Immediate, including the high-level `rq_nwqe` attribute, host receive CQs, receive doorbell initialization, and receive-state reset. Receive queues are supported for individual high-level QPs; QP groups and batched QP creation do not yet support them.
- Added rank-local signal-map lifecycle APIs in `doca_gpunetio_signal.h` to create, register, unregister, reset, and destroy mappings from signal IDs to application-owned 8-byte GPU slots using GDRCopy, plus configurable immediate-data packing macros.
- Added the `MLX5_CMD_OP_QUERY_Q_COUNTER` opcode and the `mlx5_ifc_alloc_q_counter_in_bits`, `mlx5_ifc_query_q_counter_in_bits`, and `mlx5_ifc_query_q_counter_out_bits` layouts to `host/mlx5_ifc.h`, for allocating and querying Q counter sets through DevX with `DEVX_SET`/`DEVX_GET`.

### Changed

- Renamed the LAG TX port affinity capability getters to `doca_verbs_device_attr_get_is_lag_tx_port_affinity_supported`, `doca_verbs_device_attr_get_is_init2_lag_tx_port_affinity_supported`, and `doca_verbs_device_attr_get_is_rts2rts_lag_tx_port_affinity_supported` to match the DOCA SDK APIs, and added SDK wrappers for the capability queries and QP attribute setter.
- Extended `doca_gpu_verbs_export_qp` with receive-CQ and CPU-accessible doorbell-record parameters (`cq_rq` and `dbr_cpu_ptr`). Pass `NULL` for both when the QP has no receive queue.
- Bumped the library version and minimum compatible host-code version to 5.0.0 for the host API changes.
- Updated dynamic loading to try versioned DOCA SDK, GDRCopy, and mlx5 library names before falling back to unversioned names.
- Enabled cached `__ldg` loads by default for nvcc 13.4 or newer and non-nvcc compilers, while retaining the `DOCA_GPUNETIO_VERBS_USE_LDG` override.

### Fixed

- Fixed high-level UMEM cleanup to track system allocations explicitly and use the correct deallocator for host CQ memory.
- Fixed host receive-CQ UMEM registration to allow NIC writes.
- Fixed UAR creation error propagation when allocation attempts fail.
- Fixed put/write bandwidth examples to propagate server data-validation failures.

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
