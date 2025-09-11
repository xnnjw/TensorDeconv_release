# PALM Tensor Deconvolution Demo

A MATLAB demo for hyperspectral image deconvolution using PALM algorithm with CP tensor decomposition.

## Quick Start

Simply run the main demo script:

```matlab
run('main.m')
```

## Algorithm Overview

The PALM algorithm solves the tensor deconvolution problem:

$$\min_{A,B,C} \frac{1}{2} \|\mathcal{H} \circ (A, B, C) - Y\|_F^2 + R(A, B, C)$$

where:
- $\mathcal{H}$ is the convolution operator
- $(A, B, C)$ are the CP factor matrices  
- $Y$ is the observed blurred hyperspectral image
- $R(A, B, C)$ includes regularization terms

## Parameter Efficiency

| Method | Parameters | Example (512×512×31) |
|--------|------------|---------------------|
| Full-rank | $P \times Q \times N$ | 8,257,536 |
| **CP (R=20)** | $(P+Q+N) \times R$ | **21,100** |
| **Compression Ratio** | - | **391×** |

## Expected Output

```
=== PALM Tensor Deconvolution Demo ===

Loading CAVE dataset scene 27 with kernel 1...
✓ Loaded scene: superballs_ms
✓ Image dimensions: 512 × 512 × 31

Setting up PALM parameters...
✓ Rank: 20
✓ Regularization: λ₁=4.0e-05, λ₂=6.5e-05, λ₃=1.0e-07
✓ TV regularization: λ_A=0.06, λ_B=0.00

=== Parameter Efficiency Analysis ===
Full-rank representation: 8126464 parameters
CP decomposition (rank 20): 21100 parameters
Compression ratio: 385.1× parameter reduction

Reconstructing tensor and evaluating results...
=== Reconstruction Quality ===
PSNR: 42.30 dB
SSIM: 0.9586
RMSE: 1.9722
SAM: 10.40°
Relative Error: 0.0758

```

## Demo Results

The demo generates a comprehensive visualization:

![PALM Demo Results](https://github.com/xnnjw/TensorDeconv_release/blob/master/demo_fig.png)
