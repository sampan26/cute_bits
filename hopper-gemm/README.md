# CUTLASS vs CUBLAS Performance Comparison

This benchmark compares the performance of CUBLAS against various CUTLASS kernel implementations for matrix multiplication operations (8192x8192x8192).

## Performance Results

| Implementation | Description | Time (ms) | Performance (TFLOPS) | % of CUBLAS |
|---------------|-------------|-----------|---------------------|-------------|
| **CUBLAS** | - | 1.3043 | 844.0 | 100.0% |
| Kernel 1 | Use TMA Load/Store and WGMMA | 1.6059 | 684.7 | 81.1% |
| Kernel 2 | Add Pipeline | 1.3392 | 821.0 | 97.3% |
| Kernel 3 | Add Async Store (from Smem to Gmem) | 1.3480 | 815.4 | 96.6% |
| Kernel 4 | Persistent Kernel | 1.3129 | 837.7 | 99.3% |
| **Kernel 5** | **Take advantage of Multicast Load/Store** | **1.2903** | **852.1** | **101.0%** |

## Summary

- **Best Performer**: Kernel 5 achieves 852.1 TFLOPS, outperforming CUBLAS by 1.0%
- **Closest to CUBLAS**: Kernel 4 at 99.3% of CUBLAS performance
- **Most Room for Improvement**: Kernel 1 at 81.1% of CUBLAS performance
