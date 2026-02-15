# XPU Operators

This directory contains XPU implementations of operators.

## Currently Supported Operators

- [x] add - Element-wise addition
- [x] multiply - Element-wise multiplication  
- [x] matmul - Matrix multiplication
- [x] subtract - Element-wise subtraction
- [x] divide - Element-wise division
- [x] relu - ReLU activation function
- [x] reduce_sum - Reduction sum operation

## Usage

Each operator directory contains a standalone Python file with:
- Triton kernel implementation using `@triton.jit`
- CPU reference implementation (numpy)
- PyTorch implementation for comparison
- Performance comparison between implementations
- Automatic validation of results

To run a specific operator:
```bash
python operators/add/add.py
python operators/mm/mm.py
python operators/multiply/multiply.py
python operators/subtract/subtract.py
python operators/divide/divide.py
python operators/relu/relu.py
python operators/reduce_sum/reduce_sum.py
```

## Operator Overview

### Element-wise Operations
- **add**: Element-wise addition of two tensors
- **multiply**: Element-wise multiplication of two tensors  
- **subtract**: Element-wise subtraction of two tensors
- **divide**: Element-wise division of two tensors 

### Matrix Operations
- **matmul**: Matrix multiplication

### Activation Functions
- **relu**: ReLU activation function (max(0, x))

### Reduction Operations
- **reduce_sum**: Sum reduction over all elements

## Requirements

- XPU SDK
- XPU-compatible Triton
- PyTorch with XPU support
- NumPy for CPU reference implementations

## Results Validation

Each operator automatically validates correctness by comparing Triton, PyTorch, and CPU results using `np.allclose()` with appropriate tolerances.

## Performance Reporting

Each script reports execution times and speedup ratios for:
- CPU vs PyTorch
- CPU vs Triton
- All implementations are verified to produce identical results
