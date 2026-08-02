# GPUNetIO P0 Performance Journal

## Scope

- Goal: identify and optimize one P0 steady-state performance bottleneck.
- Target: 4x NVIDIA GB200 host; use one local GPU/NIC path for controlled runs first.
- H20 comparison branch: `tyh` at `aa7f5d7af3910f7343120902c79cc54865a50446`.
- Implementation branch: `p0/put-window`, based on official main
  `df883ff6b5b52f793a7a7eea434485e38891c12f` (live-checked 2026-08-01).
- Local coroutine baseline is based on upstream `f9525001f694c472ba2d901df20325c9aafde8b4`.

## Prior-context hypotheses (not evidence)

- The H20 coroutine proof of concept improves throughput by keeping multiple PUTs outstanding before polling completion.
- Its scheduler is much heavier than the required fixed FIFO window: global task queues, global task contexts, and per-call CUDA allocations/copies.
- It has known correctness/accounting hazards, so all claims require fresh GB200 validation.

## Host inventory

- CUDA toolkit: 13.1 (`nvcc` 13.1.80).
- GPU: 4x NVIDIA GB200, each 189471 MiB, idle at initial inspection.
- GPU interconnect: NV18 between every GPU pair.
- NICs visible to NVML: `mlx5_0`, `mlx5_1`, `mlx5_4`, `mlx5_5`, `mlx5_bond_0`, `mlx5_bond_1`.
- GPU0/GPU1 are NUMA-local (`NODE`) to `mlx5_0`, `mlx5_1`, and `mlx5_bond_0`; GPU2/GPU3 are local to `mlx5_4`, `mlx5_5`, and `mlx5_bond_1`.

## Experiment log

### E0 - Build readiness

- Status: pass.
- Command: `make -j16 CUDA_ARCH=100`.
- Result: full library and all official examples build with CUDA 13.1 for `sm_100`.

### E1 - Reproduce current sync and coroutine baselines

- Status: pass on the H20-derived `tyh` branch, two repetitions per cell on GB200.
- Configuration: GPU0 `08:01:00.0`, `mlx5_0`, RoCEv2 GID 3, thread scope,
  one CUDA thread, 4096 operations, coroutine depth 1/2/4/8.
- Selected median Gbps (`sync`, `coro2`, `coro4`, `coro8`):
  - 4 KiB: 3.796, 5.560, 4.225, 4.311.
  - 64 KiB: 52.311, 92.884, 68.570, 73.235.
  - 1 MiB: 290.979, 378.329, 391.060, 389.704.
- All completed server runs passed full-buffer validation.
- Conclusion: deferring completion is useful, but the global task scheduler is not a good
  production mechanism and depth beyond four does not help large-message bandwidth.

### E2 - Fixed-window kernel

- Status: pass on official v4.
- Implementation: compile-time FIFO depths 2/4/8; PUT tickets are scalarized into registers;
  poll the oldest ticket, immediately refill that slot, then drain exactly at the end.
- Depth sweep: one CUDA thread, 8192 operations, two repetitions per depth. Depth four reached
  392.884 Gbps at 1 MiB; depths 8/16/32 in the exploratory version were all within 0.02 Gbps,
  so the production interface is capped at 8.
- Compiler evidence for depth four on `sm_100`: 60 registers, zero stack, zero local memory,
  zero shared memory, zero barriers.

### E3 - Final controlled comparison

- Status: pass.
- Configuration: official v4, GPU0 `08:01:00.0`, `mlx5_0`, GID 3, direct GPU doorbell,
  thread scope, 8192 operations, three repetitions per final cell.
- Representative commands (start server first, then client; `127.0.0.1` is OOB control only):
  - Server: `./examples/gpunetio_verbs_put_bw/gpunetio_verbs_put_bw -g 08:01:00.0 -d mlx5_0 -l 3 -t 1 -w 4 -i 8192`
  - Client: `./examples/gpunetio_verbs_put_bw/gpunetio_verbs_put_bw -g 08:01:00.0 -d mlx5_0 -l 3 -c 127.0.0.1 -e 0 -t 1 -w 4 -i 8192`

| Message | 1 thread, depth 1 | 1 thread, depth 4 | Gain |
|---:|---:|---:|---:|
| 4 KiB | 3.658 Gbps | 5.603 Gbps | +53.2% |
| 64 KiB | 50.632 Gbps | 89.657 Gbps | +77.1% |
| 1 MiB | 287.670 Gbps | 392.902 Gbps | +36.6% |

- 1 MiB depth-four range: 392.892-392.915 Gbps across three final runs. An earlier five-run
  repetition measured 392.867-392.907 Gbps (392.896 Gbps median).
- Official default path (512 threads, depth 1, 2048 operations): 389.574 Gbps median at 1 MiB.
  The one-thread depth-four path reaches 100.85% of that throughput while launching one partial
  warp instead of 16 full warps.
- WARP-scope compatibility check (32 threads, depth 1) passed validation and reached
  388.801 Gbps at 1 MiB.
- Every final server log reported `Validation successfull! Data received correctly from client`.
- Invalid depths, zero threads, and invalid WARP thread counts fail before resource creation.

### E4 - Independent review hardening (2026-08-02)

- Status: pass.
- Upstream check: NVIDIA `main` remains `df883ff`; the checkpoint has no rebase drift.
- Confirmed issue: P0 added the `-t` control, but the peers previously allocated and addressed
  buffers from independently selected thread counts. A mismatched client could therefore address
  beyond the server registration.
- Fix: exchange a versioned PUT workload descriptor before RDMA metadata, reject mismatched
  producer geometry, and use exact-length socket transfers so EOF and short I/O are failures.
  The descriptor reserves stable fields for later QP-count and message-size selection.
- Build: `make -j16 CUDA_ARCH=100` passed for the library and all examples.
- Matched regression: one thread, depth four, 8192 operations passed; representative throughput
  was 5.604 Gbps at 4 KiB, 89.656 Gbps at 64 KiB, and 392.916 Gbps at 1 MiB.
- Negative test: server `-t 1` versus client `-t 2` made both peers exit nonzero with the explicit
  configuration-mismatch diagnostic before QP connection or kernel launch.

## Decisions

- Optimize one steady-state data-plane bottleneck, not broad control-plane cleanup.
- Keep source claims separate from measured results.
- Do not count allocation/setup removal as network throughput unless it is inside the timed interval.
- Keep upstream depth-one launch on the original kernel so the default hot path and its block
  barrier are unchanged.
- Select depth four as the GB200 optimum; keep 2 and 8 available for other message sizes/hosts.
- Remaining boundary: small/medium messages plateau on per-WQE submission/doorbell cost. This
  patch fixes completion-depth starvation; it does not claim to solve submission batching.
