#!/usr/bin/env python3
"""
TinyStories 语言模型训练脚本

使用方法：
    uv run python scripts/training/train_tinystories.py
    uv run python scripts/training/train_tinystories.py --learning_rate 5e-4 --batch_size 128
"""

import sys
from pathlib import Path

import numpy as np
import torch
from simple_parsing import ArgumentParser

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cs336_basics.layers import TransformerLM
from cs336_basics.nn_utils import (
    AdamW,
    cross_entropy,
    clip_gradients,
    get_batch,
    get_lr_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
)
from cs336_basics.experiment import ExperimentConfig, ExperimentLogger, MovingAverage


def parse_args() -> ExperimentConfig:
    """解析命令行参数，直接返回 ExperimentConfig"""
    parser = ArgumentParser(description="Train TinyStories LM")
    parser.add_arguments(ExperimentConfig, dest="config")
    args = parser.parse_args()
    return args.config


def load_data(path: str, dtype=np.uint16):
    """
    使用 memory-mapped 模式加载数据，避免占用过多内存
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    
    # 使用 mmap_mode='r' 进行内存映射
    data = np.load(path, mmap_mode='r')
    print(f"加载数据: {path}")
    print(f"  - 形状: {data.shape}")
    print(f"  - 类型: {data.dtype}")
    print(f"  - Tokens: {len(data):,}")
    return data


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, num_batches=50):
    """
    在验证集上评估模型
    """
    model.eval()
    total_loss = 0.0
    
    for _ in range(num_batches):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        
        # Reshape for cross entropy: (batch * seq_len, vocab_size) vs (batch * seq_len,)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    # 直接获取配置对象
    config = parse_args()
    
    # 设置设备
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        config.device = "cpu"
    device = torch.device(config.device)
    print(f"使用设备: {device}")
    
    # 计算总 tokens
    total_tokens = config.batch_size * config.max_steps * config.context_length
    print(f"\n训练配置:")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max steps: {config.max_steps}")
    print(f"  - Context length: {config.context_length}")
    print(f"  - 总 tokens: {total_tokens:,} ({total_tokens / 1e6:.1f}M)")
    
    # 加载数据
    print("\n加载数据...")
    train_data = load_data(project_root / config.train_data_path)
    val_data = load_data(project_root / config.val_data_path)
    
    # 创建模型
    print("\n创建模型...")
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
    
    total_params, trainable_params = count_parameters(model)
    print(f"  - 总参数: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    
    # 起始步数
    start_step = 0
    
    # 恢复训练
    if config.resume:
        print(f"\n从 checkpoint 恢复: {config.resume}")
        start_step = load_checkpoint(config.resume, model, optimizer)
        print(f"  - 恢复到 step {start_step}")
    
    # 创建 checkpoint 目录
    checkpoint_dir = Path(project_root / config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志记录器 (直接使用 config)
    logger = ExperimentLogger(config, log_dir=checkpoint_dir / "logs")
    logger.start()
    
    # 移动平均 loss
    loss_avg = MovingAverage(window_size=config.log_interval)
    
    # 训练循环
    print("\n开始训练...")
    model.train()
    
    for step in range(start_step, config.max_steps):
        # 1. 获取学习率
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_iters=config.warmup_steps,
            cosine_cycle_iters=config.max_steps,
        )
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # 2. 采样 batch
        inputs, targets = get_batch(train_data, config.batch_size, config.context_length, config.device)
        
        # 3. Forward pass
        logits = model(inputs)
        
        # 4. 计算 loss
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # 5. Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 6. 梯度裁剪
        clip_gradients(model.parameters(), config.max_grad_norm)
        
        # 7. 优化器更新
        optimizer.step()
        
        # 记录 loss
        loss_value = loss.item()
        loss_avg.update(loss_value)
        
        # 日志记录
        if (step + 1) % config.log_interval == 0:
            avg_loss = loss_avg.get()
            logger.log(step + 1, {
                "train_loss": avg_loss,
                "lr": lr,
            })
        
        # 验证评估
        if (step + 1) % config.eval_interval == 0:
            val_loss = evaluate(model, val_data, config.batch_size, config.context_length, config.device)
            perplexity = np.exp(val_loss)
            logger.log(step + 1, {
                "val_loss": val_loss,
                "perplexity": perplexity,
            })
            print(f"  [验证] val_loss: {val_loss:.4f}, perplexity: {perplexity:.2f}")
        
        # 保存 checkpoint
        if (step + 1) % config.save_interval == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_step_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"  [保存] {ckpt_path}")
    
    # 训练结束
    logger.finish()
    
    # 保存最终模型
    final_ckpt_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, config.max_steps, final_ckpt_path)
    print(f"\n训练完成！最终模型保存到: {final_ckpt_path}")
    
    # 最终评估
    final_val_loss = evaluate(model, val_data, config.batch_size, config.context_length, config.device, num_batches=100)
    final_perplexity = np.exp(final_val_loss)
    print(f"最终验证 loss: {final_val_loss:.4f}")
    print(f"最终 perplexity: {final_perplexity:.2f}")
    
    if final_val_loss <= 1.45:
        print("🎉 达到目标 (val_loss ≤ 1.45)!")
    else:
        print(f"⚠️ 未达到目标 (当前: {final_val_loss:.4f}, 目标: ≤ 1.45)")


if __name__ == "__main__":
    main()
