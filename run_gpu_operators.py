#!/usr/bin/env python3
"""
Run all GPU operators for testing and performance comparison.
"""

import os
import subprocess
import sys

# Directory containing GPU operators
GPU_OPERATORS_DIR = os.path.join(os.path.dirname(__file__), 'gpu', 'operators')

# List of operators to run
OPERATORS = ['add.py', 'multiply.py', 'matmul.py']

def run_operator(operator_name):
    """
    Run a specific GPU operator script.
    
    Args:
        operator_name (str): Name of the operator script to run.
    """
    operator_path = os.path.join(GPU_OPERATORS_DIR, operator_name)
    
    if not os.path.exists(operator_path):
        print(f"Error: Operator script {operator_name} not found at {operator_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running {operator_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, operator_path], 
                              cwd=os.path.dirname(operator_path),
                              capture_output=False, 
                              text=True, 
                              check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {operator_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error running {operator_name}: {e}")
        return False

def main():
    """
    Main function to run all GPU operators.
    """
    print("Xtriton Learning - GPU Operators Test Suite")
    print("=" * 50)
    
    # Check if GPU operators directory exists
    if not os.path.exists(GPU_OPERATORS_DIR):
        print(f"Error: GPU operators directory not found at {GPU_OPERATORS_DIR}")
        return 1
    
    print(f"Running operators from: {GPU_OPERATORS_DIR}")
    print(f"Operators to run: {', '.join(OPERATORS)}")
    
    success_count = 0
    total_count = len(OPERATORS)
    
    # Run each operator
    for operator in OPERATORS:
        if run_operator(operator):
            success_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary: {success_count}/{total_count} operators ran successfully")
    print(f"{'='*60}")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
