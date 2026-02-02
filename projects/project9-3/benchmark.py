import time
import numpy as np
import faiss
from faiss_manager import FaissManager

def generate_data(n_samples: int, dim: int) -> np.ndarray:
    """生成随机向量数据 (float32)"""
    return np.random.random((n_samples, dim)).astype('float32')

def benchmark(n_samples=100000, dim=128, k=10):
    """
    基准测试函数
    :param n_samples: 向量库大小 (默认 10w)
    :param dim: 向量维度 (默认 128)
    :param k: 搜索 Top-K
    """
    print(f"=== 开始基准测试 ===")
    print(f"数据量 (N): {n_samples}, 维度 (D): {dim}, Top-K: {k}")
    
    # 准备数据
    data = generate_data(n_samples, dim)
    query = generate_data(100, dim) # 模拟 100 次查询

    # ---------------------------
    # 1. CPU 性能测试
    # ---------------------------
    print("\n[1] CPU 模式测试中...")
    start_time = time.time()
    cpu_manager = FaissManager(dim, index_type="Flat", use_gpu=False)
    
    # 添加数据耗时
    t0 = time.time()
    cpu_manager.add(data)
    add_time_cpu = time.time() - t0
    
    # 搜索耗时
    t0 = time.time()
    _, _ = cpu_manager.search(query, k)
    search_time_cpu = time.time() - t0
    
    print(f"  - CPU 构建/添加耗时: {add_time_cpu:.4f} 秒")
    print(f"  - CPU 搜索耗时 (100次): {search_time_cpu:.4f} 秒")
    
    # ---------------------------
    # 2. GPU 性能测试
    # ---------------------------
    print("\n[2] GPU 模式测试中...")
    try:
        if not hasattr(faiss, 'StandardGpuResources'):
             print("  [Error] 未检测到 faiss-gpu，跳过 GPU 测试。")
             return

        gpu_manager = FaissManager(dim, index_type="Flat", use_gpu=True)
        
        # 添加数据耗时 (包含数据从内存 -> 显存的拷贝时间)
        t0 = time.time()
        gpu_manager.add(data) 
        add_time_gpu = time.time() - t0
        
        # 搜索耗时
        t0 = time.time()
        _, _ = gpu_manager.search(query, k)
        search_time_gpu = time.time() - t0
        
        print(f"  - GPU 构建/添加耗时: {add_time_gpu:.4f} 秒")
        print(f"  - GPU 搜索耗时 (100次): {search_time_gpu:.4f} 秒")
        
        # ---------------------------
        # 3. 结果对比
        # ---------------------------
        print("\n=== 性能对比结果 ===")
        if search_time_gpu > 0:
            speedup = search_time_cpu / search_time_gpu
            print(f"搜索加速比: {speedup:.2f}x (倍)")
            if speedup < 1:
                print("注意：对于小数据集，数据传输开销可能导致 GPU 慢于 CPU。尝试增加 n_samples。")
        else:
             print("搜索加速比: 无穷大 (GPU 太快了!)")
        
    except Exception as e:
        print(f"GPU 测试失败: {e}")

if __name__ == "__main__":
    # 建议尝试不同的数据量级来观察加速效果
    # 10万数据量可能加速不明显，建议改为 50万 或 100万
    benchmark(n_samples=500000, dim=128)