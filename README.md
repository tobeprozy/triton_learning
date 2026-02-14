# Xtriton Learning Project

This project is designed to help learn and experiment with xtriton, a library for writing efficient GPU/XPU kernels using Python.

## Project Structure

```
xtriton_learning/
├── gpu/                # GPU-related operators and examples
│   ├── README.md       # GPU-specific installation and usage instructions
│   └── operators/      # GPU operators implementations
├── xpu/                # XPU-related operators and examples
│   ├── README.md       # XPU-specific installation and usage instructions
│   └── operators/      # XPU operators implementations (placeholder)
├── common/             # Common utilities and performance tools
│   └── perf/           # Performance comparison tools
├── run_gpu_operators.py # Script to run all GPU operators
└── run_xpu_operators.py # Script to run all XPU operators
```

## Installation

Please refer to the platform-specific README files for detailed installation instructions:

- **GPU Support**: See [gpu/README.md](gpu/README.md) for GPU-specific installation and usage
- **XPU Support**: See [xpu/README.md](xpu/README.md) for XPU-specific installation and usage

## Usage

### GPU Operators

Run GPU operator examples:

```bash
# Run the add operator example
python gpu/operators/add.py

# Run the multiply operator example
python gpu/operators/multiply.py

# Run the matrix multiplication operator example
python gpu/operators/matmul.py
```

### XPU Operators

Run XPU operator examples:

```bash
# Run the add operator example
python xpu/operators/add.py

# Run the multiply operator example
python xpu/operators/multiply.py

# Run the matrix multiplication operator example
python xpu/operators/matmul.py
```

## Performance Comparison

Each operator implementation includes performance comparison between:
- CPU implementation (numpy)
- Device implementation (framework-specific)
- Device implementation (Triton)

The results are displayed in the console, showing execution times and speedup factors.

## Common Tools

The `common/` directory contains shared utilities:

- **Performance Tools**: See [common/perf/](common/perf/) for performance comparison utilities
- **Utilities**: See [common/utils/](common/utils/) for common helper functions
