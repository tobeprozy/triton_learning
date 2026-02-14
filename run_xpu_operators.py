#!/usr/bin/env python3
"""
Run all XPU operators for testing and performance comparison.
"""

import os
import subprocess
import sys

# Directory containing XPU operators
XPU_OPERATORS_DIR = os.path.join(os.path.dirname(__file__), 'xpu', 'operators')

# List of operators to run
OPERATORS = ['add.py', 'multiply.py', 'matmul.py']

def run_operator(operator_name):
    """
    Run a specific XPU operator script.
    
    Args:
        operator_name (str): Name of the operator script to run.
    """
    operator_path = os.path.join(XPU_OPERATORS_DIR, operator_name)
    
    if not os.path.exists(operator_path):
        print(f"Info: Operator script {operator_name} not found at {operator_path}")
        print(f"      XPU operators are currently placeholders.")
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
    Main function to run all XPU operators.
    """
    print("Xtriton Learning - XPU Operators Test Suite")
    print("=" * 50)
    
    # Check if XPU operators directory exists
    if not os.path.exists(XPU_OPERATORS_DIR):
        print(f"Error: XPU operators directory not found at {XPU_OPERATORS_DIR}")
        return 1
    
    print(f"Running operators from: {XPU_OPERATORS_DIR}")
    print(f"Operators to run: {', '.join(OPERATORS)}")
    print("Note: XPU operators are currently placeholders.")
    print("Please refer to xpu/README.md for installation instructions.")
    
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
