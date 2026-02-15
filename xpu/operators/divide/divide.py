import triton
import triton.language as tl
import torch
import time
import numpy as np

@triton.jit
def divide_kernel(
    a_ptr, b_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算当前线程处理的索引
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 确保不越界
    mask = offsets < n_elements
    # 加载数据
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    # 执行除法操作
    output = tl.where(b != 0, a / b, 0.0)
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

def divide_triton(a, b):
    # 确保输入是torch张量
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32, device='cuda')
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32, device='cuda')
    
    # 确保a和b形状相同
    assert a.shape == b.shape
    n_elements = a.numel()
    
    # 分配输出内存
    output = torch.empty_like(a)
    
    # 设置块大小
    BLOCK_SIZE = 1024
    
    # 计算需要的块数
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 启动内核
    divide_kernel[grid](a, b, output, n_elements, BLOCK_SIZE)
    
    return output

def divide_cpu(a, b):
    # 确保输入是numpy数组
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    
    return np.divide(a, b)

def divide_pytorch(a, b):
    # 确保输入是torch张量
    if isinstance(a, np.ndarray):
        a = torch.tensor(a, dtype=torch.float32, device='cuda')
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float32, device='cuda')
    
    return torch.divide(a, b)

def main():
    # 测试不同大小的输入
    sizes = [2**10, 2**20, 2**25]
    
    for size in sizes:
        print(f"\nTesting with array size: {size}")
        
        # 创建随机输入数据，避开除零问题
        a_np = np.random.uniform(0.1, 1.0, size).astype(np.float32)
        b_np = np.random.uniform(0.1, 1.0, size).astype(np.float32)
        
        # CPU测试
        start_time = time.time()
        cpu_result = divide_cpu(a_np, b_np)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.6f} seconds")
        
        # PyTorch GPU测试
        a_torch = torch.tensor(a_np, device='cuda')
        b_torch = torch.tensor(b_np, device='cuda')
        
        # 预热
        _ = divide_pytorch(a_torch[:1024], b_torch[:1024])
        
        start_time = time.time()
        pytorch_result = divide_pytorch(a_torch, b_torch)
        pytorch_time = time.time() - start_time
        print(f"PyTorch GPU time: {pytorch_time:.6f} seconds")
        
        # GPU Triton测试
        # 预热
        _ = divide_triton(a_torch[:1024], b_torch[:1024])
        
        start_time = time.time()
        triton_result = divide_triton(a_torch, b_torch)
        triton_time = time.time() - start_time
        print(f"GPU Triton time: {triton_time:.6f} seconds")
        
        # 计算加速比
        speedup_pytorch = cpu_time / pytorch_time
        speedup_triton = cpu_time / triton_time
        print(f"Speedup (CPU vs PyTorch): {speedup_pytorch:.2f}x")
        print(f"Speedup (CPU vs Triton): {speedup_triton:.2f}x")
        
        # 验证结果是否正确
        pytorch_result_np = pytorch_result.cpu().numpy()
        triton_result_np = triton_result.cpu().numpy()
        
        assert np.allclose(cpu_result, pytorch_result_np, rtol=1e-05, atol=1e-08), "PyTorch results do not match CPU results!"
        assert np.allclose(cpu_result, triton_result_np, rtol=1e-05, atol=1e-08), "Triton results do not match CPU results!"
        print("All results match!")

if __name__ == "__main__":
    main()