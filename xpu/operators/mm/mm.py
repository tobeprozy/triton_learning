import triton
import triton.language as tl
import torch
import time
import numpy as np

@triton.jit
def mm_kernel(
    a_ptr, b_ptr, output_ptr,  # 矩阵指针
    M, N, K,  # 矩阵维度
    # 步长
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 计算当前线程处理的块
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # 计算块的起始位置
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 在K维度上分段累加
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_blocks):
        # 计算当前K段的起始位置
        k_offset = k * BLOCK_SIZE_K
        
        # 创建偏移量
        offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
        offsets_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        
        # 创建掩码防止越界访问
        mask_m = offsets_m < M
        mask_n = offsets_n < N
        mask_k = offsets_k < K
        
        # 加载A矩阵的块 (行优先)
        a_ptrs = a_ptr + (offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak)
        mask_a = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # 加载B矩阵的块 (列优先)
        b_ptrs = b_ptr + (offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn)
        mask_b = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # 累加到累加器 (使用更精确的矩阵乘法)
        acc += tl.dot(a, b, allow_tf32=False)
    
    # 保存结果到C矩阵
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    
    c_ptrs = output_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)

def mm_triton(a, b):
    # 确保输入是torch张量
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32, device='cuda')
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32, device='cuda')
    
    # 获取矩阵维度
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, f"矩阵维度不匹配: A({M}, {K}) 和 B({K_check}, {N})"
    
    # 分配输出内存
    output = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    # 设置块大小
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # 计算需要的块数
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # 启动内核
    mm_kernel[grid](
        a, b, output, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    
    return output

def mm_cpu(a, b):
    # 确保输入是numpy数组
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    
    return np.dot(a, b)

def mm_pytorch(a, b):
    # 确保输入是torch张量
    if isinstance(a, np.ndarray):
        a = torch.tensor(a, dtype=torch.float32, device='cuda')
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float32, device='cuda')
    
    return torch.matmul(a, b)

def main():
    # 测试不同大小的矩阵
    configs = [
        (128, 256, 128),   # 小矩阵
        (512, 1024, 512),  # 中等矩阵
        (1024, 2048, 1024) # 大矩阵
    ]
    
    for M, K, N in configs:
        print(f"\nTesting with matrix sizes: A({M}, {K}) x B({K}, {N}) = C({M}, {N})")
        
        # 创建随机输入数据
        a_np = np.random.rand(M, K).astype(np.float32)
        b_np = np.random.rand(K, N).astype(np.float32)
        
        # CPU测试
        start_time = time.time()
        cpu_result = mm_cpu(a_np, b_np)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.6f} seconds")
        
        # PyTorch GPU测试
        a_torch = torch.tensor(a_np, device='cuda')
        b_torch = torch.tensor(b_np, device='cuda')
        
        # 预热
        warmup_size = min(32, M, K, N)
        _ = mm_pytorch(a_torch[:warmup_size, :warmup_size], b_torch[:warmup_size, :warmup_size])
        
        start_time = time.time()
        pytorch_result = mm_pytorch(a_torch, b_torch)
        pytorch_time = time.time() - start_time
        print(f"PyTorch GPU time: {pytorch_time:.6f} seconds")
        
        # GPU Triton测试
        # 预热
        _ = mm_triton(a_torch[:warmup_size, :warmup_size], b_torch[:warmup_size, :warmup_size])
        
        start_time = time.time()
        triton_result = mm_triton(a_torch, b_torch)
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
        
        assert np.allclose(cpu_result, pytorch_result_np, rtol=1e-05, atol=1e-05), "PyTorch results do not match CPU results!"
        assert np.allclose(cpu_result, triton_result_np, rtol=1e-05, atol=1e-05), "Triton results do not match CPU results!"
        print("All results match!")

if __name__ == "__main__":
    main()