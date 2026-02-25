import triton
import triton.language as tl
import torch
import time
import numpy as np

def print_end_line():
    print("----------------------------------------------\n")

@triton.jit
def demo4(x_ptr):
    pid = tl.program_id(0)
    range = tl.arange(0, 8) + pid * 8
    x = tl.load(x_ptr + range, range < 20)
    print("Print for each", pid, x)


def run_demo4():
    print("Demo4 Output: ")
    x = torch.ones(2, 4, 4)
    demo4[(3, 1, 1)](x)
    # print_end_line()


if __name__ == "__main__":
    run_demo4()
