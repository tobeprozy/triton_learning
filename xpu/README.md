# Xtriton XPU Support

This directory contains XPU-specific implementations of operators using Triton.

## Installation

### Prerequisites
- Python 3.8-3.11
- XPU SDK (Please refer to your XPU hardware provider's documentation)
- Compatible Triton version for XPU

### Install XPU SDK

Please follow the installation instructions provided by your XPU hardware vendor. This typically involves:

1. Downloading the XPU SDK from the vendor's website
2. Installing the XPU driver
3. Setting up environment variables (e.g., `XPU_HOME`, `PATH`, `LD_LIBRARY_PATH`)

### Install Triton for XPU

Triton support for XPU may vary depending on the hardware vendor. Please check with your XPU provider for the specific Triton version and installation method.

```bash
# Example: Install Triton for XPU (consult your vendor for exact command)
pip install triton-xpu==<version>

# Install other dependencies
pip install numpy matplotlib pytest
```

### Verify Installation

```bash
# Check XPU availability
python -c "import torch; print(torch.xpu.is_available())"  # If using PyTorch XPU

# Check Triton XPU support
python -c "import triton; print(triton.xpu.is_available())"  # Consult vendor documentation
```

## Available Operators

### Add Operator
Element-wise addition implementation using Triton for XPU.

```bash
# Run the example
python operators/add.py
```

### Multiply Operator
Element-wise multiplication implementation using Triton for XPU.

```bash
# Run the example
python operators/multiply.py
```

### Matrix Multiplication Operator
Matrix multiplication implementation using Triton for XPU.

```bash
# Run the example
python operators/matmul.py
```

## Performance Comparison

Each operator implementation includes performance comparison between:
- CPU implementation (numpy)
- XPU implementation (vendor-specific framework)
- XPU implementation (Triton)

The results are displayed in the console, showing:
- Average execution time for each implementation
- Speedup of XPU implementations over CPU
- Verification of results correctness

## XPU-Specific Notes

### Environment Variables
Ensure the following environment variables are set correctly:

```bash
# Example for XPU environment setup
export XPU_HOME=/path/to/xpu/sdk
export PATH=$PATH:$XPU_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XPU_HOME/lib
```

### Kernel Compilation
XPU kernels may require specific compilation flags or optimizations. Please refer to the comments in each operator implementation for details.

### Memory Management
XPU memory management may differ from GPU. Ensure proper handling of XPU tensors and memory allocation.

## Troubleshooting

### XPU Device Not Found
- Check if XPU driver is properly installed
- Verify XPU is connected and powered on
- Ensure environment variables are set correctly

### Triton XPU Support Issues
- Consult your XPU vendor's documentation for compatible Triton versions
- Check if you're using the correct Triton package for XPU
- Ensure XPU SDK is properly configured

### Runtime Errors
- Check if your XPU has sufficient memory for the test sizes
- Verify that all dependencies are compatible with your XPU
- Try reducing the test size in the operator scripts

### Performance Issues
- Ensure you're using the latest XPU driver and SDK
- Check if XPU-specific optimizations are enabled
- Consult vendor documentation for performance tuning tips
