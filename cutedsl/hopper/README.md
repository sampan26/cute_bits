# Hopper GEMM  — Feature Checklist
Implemented
[X] Hopper WGMMA-based batched GEMM kernel skeleton (SM90)
[X] Producer/consumer warp-group specialization (1 producer WG, 2 consumer WGs via atom_layout (2,1,1))
[X] TMA G2S for A tiles (K-major)
[X] TMA G2S for B tiles (K-major)
[X] Single-shot epilogue (no sub-tiling): Accumulate → convert → R2S → single TMA S2G
[X] Correct C TMA store setup (uses C’s SMEM layout and (BM, BN) tile)
[X] Separate shared-memory buffers for A, B, and C
[X] TMA descriptor prefetch (A/B/C)
[X] PipelineTmaAsync mainloop with configurable pipeline_stages
[X] StaticPersistentTileScheduler for persistent-CTA scheduling across M×N tiles
[X] Register management: warpgroup_reg_alloc/dealloc for MMA/producer paths
[X] Tiled copy path for R2S using sm90_get_smem_store_op and make_tiled_copy_S
[X] Accumulator dtype = FP32, input/output dtype = BF16 path validated
[X] Host-side reference check against torch.matmul(A, B.t())
[X] DLPack interop for A/B/C tensors
[X] CTA tile shape fixed at (BM=128, BN=256, BK=64) with K divisible by 64
[X] Grid derivation and local tiling via zipped_divide
[X] Basic epilogue synchronization with a NamedBarrierV

[] Epilogue sub-tiling with circular-buffered stages (multi-epi epi_stage ring buffer)
[] Overlapped epilogue with PipelineTmaStore (producer tail / acquire cadence)
[] Prologue and steady-state WGMMA pipelining (commit_group / wait_group(k) with >0 lookahead)
[] TMA multicast and cluster-enabled execution (num_mcast_ctas_{A,B}, cluster arrive/wait)
[] Configurable cluster shapes, raster order, and swizzle (swizzle_size, along_m/along_n)
[] Robust consumer arrival count computation that reflects multicast & warp-group layout
[] Dynamic tiled MMA selection based on dtypes/layout (use make_trivial_tiled_mma with tiler override)
[] Full dtype matrix support & validation (FP16, FP8 e4m3fn/e5m2, INT8/UINT8) incl. K-major constraints for 8-bit paths
[] Tensor alignment & contiguity checks (16B contiguous-major rule) with helpful error messages
[] Layout flexibility for A/B/C (m-major/n-major for the output; A/B K- or M/N-major per spec)
[] Hardware-aware SMEM capacity heuristics for choosing ab_stage/epi stages (fits-per-CTA occupancy)
[] HardwareInfo-based max active clusters and occupancy tuning
[] Dedicated epilogue store warp selection consistent with WGMMA tiling (epi_store_warp_id)
[] Proper cta_layout & cluster_coord handling (not just single-CTA layout)
[] Warmup/benchmark harness (workspaces for cold-L2 option; timing utilities)
[] Better numerical test coverage (random seeds, multiple sizes, tolerance per dtype path)
[] Error handling & assertions around tile divisibility and edge tiles (non-multiples of BM/BN/BK)
[] Optional cooperative epilogue for large tiles (reduced register pressure variants)
[] Clear separation of SMEM byte accounting and buffer alignment (e.g., 1024B alignment everywhere)