import time
import numpy as np
import torch

def compare_performance(functions, input_generator, sizes, warmup_runs=3, test_runs=5):
    """
    比较多个函数的性能
    
    参数:
    functions: 字典，键为函数名称，值为函数对象
    input_generator: 生成输入数据的函数，接收size参数
    sizes: 列表，包含不同的输入大小
    warmup_runs: 预热运行的次数
    test_runs: 测试运行的次数
    
    返回:
    字典，包含每个函数在不同大小下的平均运行时间
    """
    results = {name: [] for name in functions.keys()}
    
    for size in sizes:
        print(f"\nTesting with size: {size}")
        
        # 生成输入数据
        inputs = input_generator(size)
        
        # 对每个函数进行测试
        for name, func in functions.items():
            # 预热
            for _ in range(warmup_runs):
                _ = func(*inputs)
            
            # 确保所有计算完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 测试
            total_time = 0
            for _ in range(test_runs):
                start_time = time.time()
                result = func(*inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_time = total_time / test_runs
            results[name].append(avg_time)
            print(f"{name} average time: {avg_time:.6f} seconds")
        
        # 验证所有结果是否一致
        if len(functions) > 1:
            all_results = []
            for name, func in functions.items():
                result = func(*inputs)
                if isinstance(result, torch.Tensor):
                    result = result.cpu().numpy()
                all_results.append(result)
            
            # 检查所有结果是否与第一个结果接近
            base_result = all_results[0]
            for i, result in enumerate(all_results[1:], 1):
                assert np.allclose(base_result, result, rtol=1e-05, atol=1e-08), \
                    f"{list(functions.keys())[0]} and {list(functions.keys())[i]} produce different results!"
            print("All results match!")
    
    return results

def plot_results(results, sizes, title="Performance Comparison"):
    """
    绘制性能比较图表
    
    参数:
    results: 字典，包含每个函数在不同大小下的平均运行时间
    sizes: 列表，包含不同的输入大小
    title: 图表标题
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    for name, times in results.items():
        plt.plot(sizes, times, marker='o', label=name)
    
    plt.xlabel('Input Size')
    plt.ylabel('Average Time (seconds)')
    plt.title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def print_speedup(results, baseline_name="CPU"):
    """
    计算并打印加速比
    
    参数:
    results: 字典，包含每个函数在不同大小下的平均运行时间
    baseline_name: 基线函数名称
    """
    if baseline_name not in results:
        print(f"Baseline {baseline_name} not found in results")
        return
    
    print(f"\nSpeedup compared to {baseline_name}:")
    for name, times in results.items():
        if name == baseline_name:
            continue
        
        print(f"\n{name}:")
        for i, (size, baseline_time, current_time) in enumerate(zip(sizes, results[baseline_name], times)):
            speedup = baseline_time / current_time
            print(f"Size {size}: {speedup:.2f}x")
