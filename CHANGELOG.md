# Changelog

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
