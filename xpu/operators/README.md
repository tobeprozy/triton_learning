# XPU Operators

This directory contains XPU implementations of operators.

## Currently Supported Operators

- [ ] add - Element-wise addition
- [ ] multiply - Element-wise multiplication
- [ ] matmul - Matrix multiplication

## Usage

XPU operators will follow the same interface as GPU operators, with:
- Triton kernel implementation
- CPU reference implementation
- PyTorch implementation for comparison
- Performance comparison between implementations

## Requirements

- XPU SDK
- XPU-compatible Triton
