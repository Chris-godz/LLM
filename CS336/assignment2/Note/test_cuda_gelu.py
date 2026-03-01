import torch
import os
import math
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
import time

# 强制限制针对特定架构进行编译，以避免本地 NVCC 不支持高版本架构引起的报错
# 加入 +PTX 使得当前版本的代码能够在更高版本架构(如 RTX 50 系列)上进行 JIT 即时编译运行。
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_cuda_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float x = input[index];
        // exact GELU formula: 0.5 * x * (1 + erf(x / sqrt(2)))
        output[index] = x * 0.5f * (1.0f + erff(x * 0.707106781f)); // 1/sqrt(2) ≈ 0.707106781
    }
}

torch::Tensor gelu_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int threads_per_block = 256;
    int blocks_per_grid = (input.numel() + threads_per_block - 1) / threads_per_block;
    
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only Float32 is supported!");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor!");

    gelu_cuda_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor gelu_cuda_forward(torch::Tensor input);
"""

print("正在即时编译 CUDA Kernel (初次编译往往需要一点时间，请稍候)...")
gelu_ext = load_inline(
    name="cuda_gelu_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gelu_cuda_forward"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

def cuda_gelu(x):
    """自定义 CUDA 版的 GeLU 函数"""
    return gelu_ext.gelu_cuda_forward(x.contiguous())


# ===================================================================
# Triton GeLU 实现部分
# ===================================================================

@triton.jit
def gelu_triton_kernel(
    in_ptr0,
    out_ptr0,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # \sqrt(2) \approx 1.41421356
    # exact GELU formula: 0.5 * x * (1 + erf(x / sqrt(2)))
    output = 0.5 * x * (1.0 + tl.math.erf(x / 1.41421356))
    
    tl.store(out_ptr0 + offsets, output, mask=mask)


def triton_gelu(x: torch.Tensor):
    """自定义 Triton 版的 GeLU 函数"""
    x = x.contiguous()
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 启发式设置：使用大于等于数据总数的最小二次幂，但最大不超过 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_triton_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return output


# ===================================================================
# 测试与验证
# ===================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA 不可用，请在带有 GPU 的环境中运行此脚本！")
        exit(1)

    print("\\n=== 编译成功，开始验证结果 ===")
    
    # 构造测试数据 (必须使用 Float32 存放在 CUDA 上)
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    
    # 测试输出正确性
    y_cuda = cuda_gelu(x)
    y_triton = triton_gelu(x)
    y_pt = torch.nn.functional.gelu(x)
    
    diff_cuda = (y_cuda - y_pt).abs().max().item()
    diff_triton = (y_triton - y_pt).abs().max().item()
    
    print(f"CUDA GeLU 与 PyTorch 标准 GeLU 最大绝对误差为: {diff_cuda:.8e}")
    print(f"Triton GeLU 与 PyTorch 标准 GeLU 最大绝对误差为: {diff_triton:.8e}")
    
    if diff_cuda < 1e-5 and diff_triton < 1e-5:
        print("结论：验证通过 ✅，计算结果与 PyTorch 均一致！")
    else:
        print("结论：验证失败 ❌，计算结果存在偏差。")
        
    print("\\n=== 性能比对 (测速中) ===")
    # 预热 GPU (Warmup)
    for _ in range(10):
        _ = cuda_gelu(x)
        _ = triton_gelu(x)
        _ = torch.nn.functional.gelu(x)
    torch.cuda.synchronize()
        
    runs = 1000
    
    # 测试自定义 CUDA 内核
    start = time.perf_counter()
    for _ in range(runs):
        _ = cuda_gelu(x)
    torch.cuda.synchronize()
    custom_cuda_time = time.perf_counter() - start
    print(f"自定义 CUDA GeLU (运行 {runs} 次): {custom_cuda_time:.4f} 秒")
    
    # 测试自定义 Triton 内核
    start = time.perf_counter()
    for _ in range(runs):
        _ = triton_gelu(x)
    torch.cuda.synchronize()
    custom_triton_time = time.perf_counter() - start
    print(f"自定义 Triton GeLU (运行 {runs} 次): {custom_triton_time:.4f} 秒")
    
    # 测试 PyTorch 内置内核
    start = time.perf_counter()
    for _ in range(runs):
        _ = torch.nn.functional.gelu(x)
    torch.cuda.synchronize()
    pt_time = time.perf_counter() - start
    print(f"PyTorch 内置 GeLU (运行 {runs} 次):   {pt_time:.4f} 秒")
