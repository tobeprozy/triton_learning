import triton
import triton.language as tl
import torch
import time
import numpy as np

@triton.jit
def matmul_kernel(
    # 输入矩阵的指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵的维度
    M, N, K,
    # 矩阵的步长
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 块大小
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 获取当前线程块的坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块在矩阵中的起始位置
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算偏移量
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 遍历K维度
    for k in range(0, K, BLOCK_SIZE_K):
        # 加载A和B的块
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # 矩阵乘法计算
        accumulator += tl.dot(a, b)
        
        # 更新指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 计算输出指针
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    
    # 存储结果
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul_triton(a, b):
    # 确保输入是torch张量
    if isinstance(a, np.ndarray):
        a = torch.tensor(a, dtype=torch.float32, device='cuda')
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float32, device='cuda')
    
    # 获取矩阵维度
    M, K = a.shape
    K, N = b.shape
    
    # 分配输出内存
    c = torch.empty((M, N), dtype=torch.float32, device='cuda')
    
    # 设置块大小
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    
    # 计算需要的块数
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # 启动内核
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    
    return c

def matmul_cpu(a, b):
    # 确保输入是numpy数组
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    
    return np.dot(a, b)

def matmul_pytorch(a, b):
    # 确保输入是torch张量
    if isinstance(a, np.ndarray):
        a = torch.tensor(a, dtype=torch.float32, device='cuda')
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float32, device='cuda')
    
    return torch.matmul(a, b)

def main():
    # 测试不同大小的矩阵
    sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    
    for M, K, N in sizes:
        print(f"\nTesting with matrix size: {M}x{K} and {K}x{N}")
        
        # 创建随机输入数据
        a_np = np.random.rand(M, K).astype(np.float32)
        b_np = np.random.rand(K, N).astype(np.float32)
        
        # CPU测试
        start_time = time.time()
        cpu_result = matmul_cpu(a_np, b_np)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.6f} seconds")
        
        # PyTorch GPU测试
        a_torch = torch.tensor(a_np, device='cuda')
        b_torch = torch.tensor(b_np, device='cuda')
        
        # 预热
        _ = matmul_pytorch(a_torch[:32, :32], b_torch[:32, :32])
        
        start_time = time.time()
        pytorch_result = matmul_pytorch(a_torch, b_torch)
        pytorch_time = time.time() - start_time
        print(f"PyTorch GPU time: {pytorch_time:.6f} seconds")
        
        # GPU Triton测试
        # 预热
        _ = matmul_triton(a_torch[:32, :32], b_torch[:32, :32])
        
        start_time = time.time()
        triton_result = matmul_triton(a_torch, b_torch)
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
