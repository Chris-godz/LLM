"""
§6 Generating Text - Text Generation / Decoding

实现从 Transformer LM 生成文本的功能，包括：
- Temperature scaling
- Top-p (Nucleus) sampling
- 主生成循环
"""

import torch
import torch.nn.functional as F
from typing import Optional

from .layers import TransformerLM
from .tokenizer import Tokenizer


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    带温度的 softmax
    
    Args:
        logits: (*, vocab_size) 未归一化的 logits
        temperature: 温度参数
            - τ → 0: 接近 argmax（确定性选择最高概率）
            - τ = 1: 标准 softmax
            - τ > 1: 更平滑的分布（更随机）
    
    Returns:
        概率分布 (*, vocab_size)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    # logits / τ 然后 softmax
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)


def top_p_filter(probs: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    """
    Top-p (Nucleus) 采样过滤
    
    只保留累积概率 >= top_p 的最小 token 集合，其余设为 0 并重新归一化。
    
    Args:
        probs: (*,vocab_size) 概率分布
        top_p: 累积概率阈值，1.0 表示不过滤
    
    Returns:
        过滤并重新归一化后的概率分布
    """
    if top_p >= 1.0:
        return probs
    
    if top_p <= 0.0:
        raise ValueError("top_p must be positive")
    
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到第一个累积概率 > top_p 的位置
    # 创建 mask：在累积概率超过 top_p 之后的位置设为 True
    # 我们需要保留累积概率刚好达到 top_p 的那个 token
    sorted_mask = cumulative_probs > top_p
    
    # 将 mask 右移一位，确保第一个超过阈值的 token 也被保留
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    
    # 将被 mask 的位置设为 0
    sorted_probs[sorted_mask] = 0.0
    
    # 还原到原始顺序
    # 创建一个空的结果 tensor
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    
    # 重新归一化
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    return filtered_probs


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
    device: Optional[torch.device] = None,
) -> str:
    """
    从 Transformer LM 生成文本
    
    Args:
        model: TransformerLM 模型
        tokenizer: Tokenizer 实例
        prompt: 输入文本提示
        max_new_tokens: 最多生成的新 token 数量
        temperature: 温度参数，控制采样的随机性
        top_p: Top-p 采样阈值
        eos_token: 结束标记
        device: 设备
    
    Returns:
        生成的完整文本（包含 prompt）
    """
    model.eval()
    
    # 确定设备
    if device is None:
        device = next(model.parameters()).device
    
    # 编码 prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)
    
    # 获取 eos token id（如果存在的话）
    eos_token_id = None
    if eos_token:
        eos_bytes = eos_token.encode('utf-8')
        if eos_bytes in tokenizer.vocab_inverse:
            eos_token_id = tokenizer.vocab_inverse[eos_bytes]
    
    # 生成循环
    generated_ids = input_ids
    
    for _ in range(max_new_tokens):
        # 如果序列太长，截断到 context_length
        if generated_ids.shape[1] > model.context_length:
            # 只保留最后 context_length 个 token
            input_for_model = generated_ids[:, -model.context_length:]
        else:
            input_for_model = generated_ids
        
        # 前向传播
        logits = model(input_for_model)  # (1, seq_len, vocab_size)
        
        # 取最后一个位置的 logits
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)
        
        # Temperature scaling
        probs = softmax_with_temperature(next_token_logits, temperature)
        
        # Top-p filtering
        probs = top_p_filter(probs, top_p)
        
        # 采样
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        
        # 检查是否是 eos token
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        
        # 追加到序列
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # 解码并返回
    output_ids = generated_ids[0].tolist()
    return tokenizer.decode(output_ids)
