import triton
import triton.language as tl
import torch
import time
import numpy as np

@triton.jit
def relu_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算当前线程处理的索引
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 确保不越界
    mask = offsets < n_elements
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    # 执行ReLU操作：max(0, x)
    output = tl.where(x > 0, x, 0.0)
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

def relu_triton(x):
    # 确保输入是torch张量
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
    
    n_elements = x.numel()
    
    # 分配输出内存
    output = torch.empty_like(x)
    
    # 设置块大小
    BLOCK_SIZE = 1024
    
    # 计算需要的块数
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 启动内核
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    
    return output

def relu_cpu(x):
    # 确保输入是numpy数组
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    
    return np.maximum(x, 0)

def relu_pytorch(x):
    # 确保输入是torch张量
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
    
    return torch.nn.functional.relu(x)

def main():
    # 测试不同大小的输入
    sizes = [2**10, 2**20, 2**25]
    
    for size in sizes:
        print(f"\nTesting with array size: {size}")
        
        # 创建随机输入数据，包括负数测试ReLU效果
        x_np = np.random.randn(size).astype(np.float32)
        
        # CPU测试
        start_time = time.time()
        cpu_result = relu_cpu(x_np)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.6f} seconds")
        
        # PyTorch GPU测试
        x_torch = torch.tensor(x_np, device='cuda')
        
        # 预热
        _ = relu_pytorch(x_torch[:1024])
        
        start_time = time.time()
        pytorch_result = relu_pytorch(x_torch)
        pytorch_time = time.time() - start_time
        print(f"PyTorch GPU time: {pytorch_time:.6f} seconds")
        
        # GPU Triton测试
        # 预热
        _ = relu_triton(x_torch[:1024])
        
        start_time = time.time()
        triton_result = relu_triton(x_torch)
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