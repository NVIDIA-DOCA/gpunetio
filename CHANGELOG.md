# Changelog

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
