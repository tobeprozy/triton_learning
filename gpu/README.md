# Xtriton GPU Support

This directory contains GPU-specific implementations of operators using Triton.

## Installation

### Prerequisites
- Python 3.8-3.11
- CUDA 11.8-12.2
- PyTorch (compatible with your CUDA version)

### Install Triton for GPU

```bash
# Recommended version
pip install triton==2.2.0

# Install other dependencies
pip install numpy torch matplotlib pytest
```

### Verify Installation

```bash
python -c "import triton; print(triton.__version__)"
```

## Available Operators

### Add Operator
Element-wise addition implementation using Triton.

```bash
# Run the example
python operators/add.py
```

### Multiply Operator
Element-wise multiplication implementation using Triton.

```bash
# Run the example
python operators/multiply.py
```

### Matrix Multiplication Operator
Matrix multiplication implementation using Triton.

```bash
# Run the example
python operators/matmul.py
```

## Performance Comparison

Each operator implementation includes performance comparison between:
- CPU implementation (numpy)
- GPU implementation (PyTorch)
- GPU implementation (Triton)

The results are displayed in the console, showing:
- Average execution time for each implementation
- Speedup of GPU implementations over CPU
- Verification of results correctness

## Troubleshooting

### Triton Installation Issues
- Ensure your Python version is between 3.8 and 3.11
- Ensure your CUDA version is compatible with Triton (11.8-12.2)
- If using PyTorch with CUDA, ensure Triton version is compatible

### Runtime Errors
- Check if CUDA is properly installed and accessible
- Ensure you have sufficient GPU memory for the test sizes
- Try reducing the test size in the operator scripts

### Performance Issues
- Ensure you're running on a compatible GPU (NVIDIA GPU with compute capability >= 7.0)
- For best performance, run on a GPU with Tensor Cores (compute capability >= 7.5)


# Reference
https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
