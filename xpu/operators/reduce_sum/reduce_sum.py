import triton
import triton.language as tl
import torch
import time
import numpy as np

@triton.jit
def reduce_sum_kernel(
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
    # 计算块内局部和
    block_sum = tl.sum(x)
    # 存储局部和到共享内存
    tl.store(output_ptr + pid, block_sum)

@triton.jit
def reduce_final_kernel(
    partial_sum_ptr, final_sum_ptr, num_partial_sums,
    BLOCK_SIZE: tl.constexpr,
):
    # 只有一个线程（pid=0）来处理所有部分和
    pid = tl.program_id(axis=0)
    if pid == 0:
        # 使用BLOCK_SIZE来加载部分和
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_partial_sums
        # 加载部分和
        partial_sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
        # 计算最终和
        final_sum = tl.sum(partial_sums)
        # 存储最终结果
        tl.store(final_sum_ptr, final_sum)

def reduce_sum_triton(x):
    # 确保输入是torch张量
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
    
    n_elements = x.numel()
    
    # 设置块大小
    BLOCK_SIZE = 1024
    
    # 计算需要的块数
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # 分配部分和内存
    partial_sums = torch.empty((num_blocks,), dtype=torch.float32, device='cuda')
    
    # 第一步：每个块计算部分和
    grid1 = (num_blocks,)
    reduce_sum_kernel[grid1](x, partial_sums, n_elements, BLOCK_SIZE)
    
    # 第二步：直接在CPU端合并部分和（更简单可靠）
    # 当部分和数量小于等于1024时，可以用triton合并；否则用torch.sum
    if num_blocks <= 1024:
        final_result = torch.empty((1,), dtype=torch.float32, device='cuda')
        grid2 = (1,)  # 只使用1个线程来合并所有部分和
        reduce_final_kernel[grid2](partial_sums, final_result, num_blocks, num_blocks)
        return final_result
    else:
        # 使用torch.sum在GPU上合并部分和
        return torch.sum(partial_sums)

def reduce_sum_cpu(x):
    # 确保输入是numpy数组
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    
    return np.sum(x)

def reduce_sum_pytorch(x):
    # 确保输入是torch张量
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
    
    return torch.sum(x)

def main():
    # 测试不同大小的输入
    sizes = [2**10, 2**20, 2**25]
    
    for size in sizes:
        print(f"\nTesting with array size: {size}")
        
        # 创建随机输入数据
        x_np = np.random.rand(size).astype(np.float32)
        
        # CPU测试
        start_time = time.time()
        cpu_result = reduce_sum_cpu(x_np)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.6f} seconds")
        
        # PyTorch GPU测试
        x_torch = torch.tensor(x_np, device='cuda')
        
        # 预热
        _ = reduce_sum_pytorch(x_torch[:1024])
        
        start_time = time.time()
        pytorch_result = reduce_sum_pytorch(x_torch)
        pytorch_time = time.time() - start_time
        print(f"PyTorch GPU time: {pytorch_time:.6f} seconds")
        
        # GPU Triton测试
        # 预热
        _ = reduce_sum_triton(x_torch[:1024])
        
        start_time = time.time()
        triton_result_tensor = reduce_sum_triton(x_torch)
        triton_time = time.time() - start_time
        print(f"GPU Triton time: {triton_time:.6f} seconds")
        
        # 计算加速比
        speedup_pytorch = cpu_time / pytorch_time
        speedup_triton = cpu_time / triton_time
        print(f"Speedup (CPU vs PyTorch): {speedup_pytorch:.2f}x")
        print(f"Speedup (CPU vs Triton): {speedup_triton:.2f}x")
        
        # 验证结果是否正确
        pytorch_result_np = pytorch_result.cpu().numpy()
        
        # triton_result_tensor可能是标量或包含一个元素的张量
        triton_result_cpu = triton_result_tensor.cpu()
        if triton_result_cpu.dim() == 0:  # 标量
            triton_result_np = triton_result_cpu.numpy()
        else:  # 包含一个元素的张量
            triton_result_np = triton_result_cpu.numpy()[0]
        
        assert np.allclose(cpu_result, pytorch_result_np, rtol=1e-05, atol=1e-08), "PyTorch results do not match CPU results!"
        assert np.allclose(cpu_result, triton_result_np, rtol=1e-05, atol=1e-08), "Triton results do not match CPU results!"
        print("All results match!")

if __name__ == "__main__":
    main()