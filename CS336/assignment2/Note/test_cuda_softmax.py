import torch
import triton
import triton.language as tl
import time

# ===================================================================
# 1. Manual PyTorch Softmax 实现
# ===================================================================
def manual_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """使用基础 PyTorch 算子手动实现 Softmax (保持数值稳定性)"""
    # 提取输入维度中的最大值（为了数值稳定性：减去最大值以防数值溢出）
    x_max = x.max(dim=dim, keepdim=True).values
    x_safe = x - x_max
    
    # 指数计算
    exp_x = torch.exp(x_safe)
    
    # 求和计算分母
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    # 返回概率分布
    return exp_x / sum_exp

# ===================================================================
# 2. Triton Softmax 实现
# ===================================================================

@triton.jit
def softmax_triton_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # 行索引，通常我们在行上进行并行处理
    row_idx = tl.program_id(0)

    # 计算该行所对应指针的起始偏移
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 按照 BLOCK_SIZE 设置列的读取范围
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # 应对 n_cols 可能不是 BLOCK_SIZE 倍数的情况，使用 mask 遮挡
    mask = col_offsets < n_cols
    
    # 将 DRAM 中的数据载入到 SRAM 中
    # 当 col_offsets 即将超出 n_cols 时，使用极小值填充，防止在随后的 max 操作中出错
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # --- 数值稳定性：每一行减去最大值 ---
    row_minus_max = row - tl.max(row, axis=0)
    
    # --- 指数与归一化计算 ---
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # --- 将输出写回 DRAM ---
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton 版的 Softmax 功能"""
    x = x.contiguous()
    # 将输入视为 2D (batch x seq_len) 进行处理
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    n_rows, n_cols = x_2d.shape
    
    # 为了简化计算，在块大小上，选一个最小巧且刚超过 n_cols 的 2 的幂次方尺寸。
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # 分配一定数量的 warps (一个 warp=32 threads) 以防 BLOCK_SIZE 尺寸偏大时产生报错问题
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
        
    y_2d = torch.empty_like(x_2d)
    
    # 并行数量等于 n_rows，这样每个 program 将负责一整行数据的 softmax 运算
    softmax_triton_kernel[(n_rows,)](
        y_2d,
        x_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_2d.view(orig_shape)

# ===================================================================
# 3. 测试与验证
# ===================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA 不可用，请在带有 GPU 的环境中运行此脚本！")
        exit(1)

    print("\\n=== 初始化验证，对比计算结果 ===")
    
    # 构造测试数据 
    torch.manual_seed(42)
    # 取一个经典的 2D 张量来测试 (BatchSize, SeqLength)
    # 比如在 Attention 以及一般分类场景中常见的维度
    M, N = 4096, 4096 
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    
    # 测试三种实现的结果
    # 预热并运行，同时保留结果
    y_manual = manual_softmax(x)
    y_pt = torch.nn.functional.softmax(x, dim=-1)
    y_triton = triton_softmax(x)
    
    diff_manual = (y_manual - y_pt).abs().max().item()
    diff_triton = (y_triton - y_pt).abs().max().item()
    
    print(f"Manual Softmax 与 PyTorch 标准 Softmax 绝对误差: {diff_manual:.8e}")
    print(f"Triton Softmax 与 PyTorch 标准 Softmax 绝对误差: {diff_triton:.8e}")
    
    if diff_manual < 1e-4 and diff_triton < 1e-4:
        print("结论：验证通过 ✅，计算结果与 PyTorch 标准实现均一致！")
    else:
        print("结论：验证失败 ❌，计算结果存在偏差。")
        
    print("\\n=== 性能比对 (测速中) ===")
    # 预热 GPU (Warmup)
    for _ in range(10):
        _ = manual_softmax(x)
        _ = torch.nn.functional.softmax(x, dim=-1)
        _ = triton_softmax(x)
    torch.cuda.synchronize()
        
    runs = 1000
    
    # 测试 Manual 内核（PyTorch基础算子组合）
    start = time.perf_counter()
    for _ in range(runs):
        _ = manual_softmax(x)
    torch.cuda.synchronize()
    manual_time = time.perf_counter() - start
    print(f"PyTorch Manual 组合算子版 Softmax (运行 {runs} 次): {manual_time:.4f} 秒")
    
    # 测试自定义 Triton 内核
    start = time.perf_counter()
    for _ in range(runs):
        _ = triton_softmax(x)
    torch.cuda.synchronize()
    custom_triton_time = time.perf_counter() - start
    print(f"自定义 Triton Softmax 算子 (运行 {runs} 次):      {custom_triton_time:.4f} 秒")
    
    # 测试 PyTorch 内置内核
    start = time.perf_counter()
    for _ in range(runs):
        _ = torch.nn.functional.softmax(x, dim=-1)
    torch.cuda.synchronize()
    pt_time = time.perf_counter() - start
    print(f"PyTorch Native C++ 拓展内置 Softmax (运行 {runs} 次): {pt_time:.4f} 秒")
