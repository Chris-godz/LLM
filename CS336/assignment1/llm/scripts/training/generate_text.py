#!/usr/bin/env python3
"""
使用训练好的 TinyStories 模型生成文本

用法:
    uv run python scripts/training/generate_text.py
    uv run python scripts/training/generate_text.py --prompt "Once upon a time" --temperature 0.8
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from simple_parsing import ArgumentParser, field as sp_field

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cs336_basics.layers import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.generation import generate
from cs336_basics.experiment import ExperimentConfig


@dataclass
class GenerationConfig:
    """生成配置"""
    prompt: str = sp_field(default="Once upon a time", help="生成的起始文本")
    max_tokens: int = sp_field(default=256, help="最大生成 token 数")
    temperature: float = sp_field(default=0.8, help="温度参数 (0.0-1.0，越低越确定)")
    top_p: float = sp_field(default=0.95, help="Top-p 采样阈值")


def main():
    parser = ArgumentParser(description="Generate text from trained model")
    parser.add_arguments(ExperimentConfig, dest="config")
    parser.add_arguments(GenerationConfig, dest="gen")
    args = parser.parse_args()
    
    config: ExperimentConfig = args.config
    gen: GenerationConfig = args.gen
    
    # 设备
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        config.device = "cpu"
    device = torch.device(config.device)
    print(f"使用设备: {device}")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    vocab_path = project_root / config.tokenizer_vocab_path
    merges_path = project_root / config.tokenizer_vocab_path.replace("vocab_", "merges_").replace(".pkl", ".txt")
    tokenizer = Tokenizer.from_files(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=["<|endoftext|>"],
    )
    print(f"  - Vocab size: {len(tokenizer.vocab)}")
    print(f"  - Merges: {len(tokenizer.merges)}")
    
    # 创建模型
    print("创建模型...")
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=device,
    ).to(device)
    
    # 加载 checkpoint
    checkpoint_path = project_root / config.checkpoint_dir / "checkpoint_final.pt"
    if config.resume:
        checkpoint_path = Path(config.resume)
    
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  - 从 step {checkpoint['iteration']} 恢复")
    
    # 生成文本
    print(f"\n生成文本...")
    print(f"  - Prompt: {gen.prompt}")
    print(f"  - Temperature: {gen.temperature}")
    print(f"  - Top-p: {gen.top_p}")
    print(f"  - Max tokens: {gen.max_tokens}")
    print("-" * 50)
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=gen.prompt,
        max_new_tokens=gen.max_tokens,
        temperature=gen.temperature,
        top_p=gen.top_p,
        device=device,
    )
    
    print(generated_text)
    print("-" * 50)
    
    # 统计
    tokens = tokenizer.encode(generated_text)
    print(f"\n总 tokens: {len(tokens)}")


if __name__ == "__main__":
    main()
