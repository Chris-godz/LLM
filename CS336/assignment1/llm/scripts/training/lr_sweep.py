#!/usr/bin/env python3
"""
Learning Rate Sweep 实验

用法:
    # 快速测试（2000步）
    uv run python scripts/training/lr_sweep.py --max_steps 2000
    
    # 完整实验（20000步）
    uv run python scripts/training/lr_sweep.py --max_steps 20000
"""

import sys
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# 项目根目录
project_root = Path(__file__).parent.parent.parent


def run_experiment(lr: float, max_steps: int, output_dir: Path) -> dict:
    """运行单个学习率实验"""
    exp_name = f"lr_{lr:.0e}".replace(".", "_")
    checkpoint_dir = output_dir / exp_name
    
    print(f"\n{'='*60}")
    print(f"实验: lr={lr:.0e}, max_steps={max_steps}")
    print(f"输出目录: {checkpoint_dir}")
    print(f"{'='*60}")
    
    # 构建命令
    min_lr = lr / 100  # min_lr 设为 max_lr 的 1/100
    cmd = [
        "uv", "run", "python", "scripts/training/train_tinystories.py",
        f"--learning_rate={lr}",
        f"--min_learning_rate={min_lr}",
        f"--max_steps={max_steps}",
        f"--batch_size=108",  # 与之前训练保持一致
        f"--checkpoint_dir={checkpoint_dir}",
        f"--experiment_name={exp_name}",
        f"--warmup_steps={max(int(max_steps * 0.02), 10)}",  # 2% warmup
        f"--save_interval=3725",  # 只在中间保存一次
    ]
    
    # 运行训练
    result = subprocess.run(cmd, cwd=project_root, capture_output=False)
    
    # 读取日志
    log_file = checkpoint_dir / "logs" / "logs.json"
    if log_file.exists():
        with open(log_file, "r") as f:
            logs = json.load(f)
        return {
            "lr": lr,
            "logs": logs,
            "success": result.returncode == 0,
        }
    else:
        return {
            "lr": lr,
            "logs": [],
            "success": False,
        }


def plot_sweep_results(results: list, output_path: Path):
    """绘制 sweep 结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # 1. Train Loss vs Step
    ax1 = axes[0]
    for i, res in enumerate(results):
        if not res["logs"]:
            continue
        
        steps = [e["step"] for e in res["logs"] if "train_loss" in e]
        losses = [e["train_loss"] for e in res["logs"] if "train_loss" in e]
        
        if steps:
            ax1.plot(steps, losses, label=f'lr={res["lr"]:.0e}', color=colors[i], alpha=0.8)
    
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Train Loss vs Step")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Val Loss vs Step
    ax2 = axes[1]
    for i, res in enumerate(results):
        if not res["logs"]:
            continue
        
        steps = [e["step"] for e in res["logs"] if "val_loss" in e]
        losses = [e["val_loss"] for e in res["logs"] if "val_loss" in e]
        
        if steps:
            ax2.plot(steps, losses, label=f'lr={res["lr"]:.0e}', color=colors[i], 
                    marker='o', markersize=3, alpha=0.8)
    
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Validation Loss vs Step")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Learning Rate Sweep", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n保存图片到: {output_path}")
    plt.close()


def print_summary(results: list):
    """打印实验摘要"""
    print("\n" + "=" * 70)
    print("Learning Rate Sweep 摘要")
    print("=" * 70)
    print(f"{'LR':>12} | {'Final Train':>12} | {'Final Val':>12} | {'Min Val':>12} | {'Status':>10}")
    print("-" * 70)
    
    for res in results:
        lr = res["lr"]
        
        if not res["logs"] or not res["success"]:
            print(f"{lr:>12.0e} | {'---':>12} | {'---':>12} | {'---':>12} | {'FAILED':>10}")
            continue
        
        train_losses = [e["train_loss"] for e in res["logs"] if "train_loss" in e]
        val_losses = [e["val_loss"] for e in res["logs"] if "val_loss" in e]
        
        final_train = train_losses[-1] if train_losses else float('nan')
        final_val = val_losses[-1] if val_losses else float('nan')
        min_val = min(val_losses) if val_losses else float('nan')
        
        # 检查是否发散
        if train_losses and (train_losses[-1] > train_losses[0] * 2 or np.isnan(train_losses[-1])):
            status = "DIVERGED"
        else:
            status = "OK"
        
        print(f"{lr:>12.0e} | {final_train:>12.4f} | {final_val:>12.4f} | {min_val:>12.4f} | {status:>10}")
    
    print("=" * 70)
    
    # 找出最佳学习率
    valid_results = [r for r in results if r["logs"] and r["success"]]
    if valid_results:
        best = min(valid_results, key=lambda r: min([e["val_loss"] for e in r["logs"] if "val_loss" in e], default=float('inf')))
        print(f"\n最佳学习率: {best['lr']:.0e}")


def main():
    parser = argparse.ArgumentParser(description="Learning Rate Sweep")
    parser.add_argument("--max_steps", type=int, default=7500,
                        help="每个实验的最大步数 (默认 7500)")
    parser.add_argument("--output_dir", type=str, default="scripts/training/lr_sweep_results",
                        help="输出目录")
    parser.add_argument("--learning_rates", type=str, default="1e-4,3e-4,5e-4,1e-3,3e-3,5e-3",
                        help="学习率列表，逗号分隔")
    
    args = parser.parse_args()
    
    # 解析学习率
    learning_rates = [float(lr) for lr in args.learning_rates.split(",")]
    
    # 创建输出目录
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Learning Rate Sweep 实验")
    print(f"  - 学习率: {learning_rates}")
    print(f"  - 每个实验步数: {args.max_steps}")
    print(f"  - 输出目录: {output_dir}")
    
    # 运行实验
    results = []
    for lr in learning_rates:
        res = run_experiment(lr, args.max_steps, output_dir)
        results.append(res)
    
    # 保存结果
    results_file = output_dir / "sweep_results.json"
    with open(results_file, "w") as f:
        # 只保存摘要，不保存完整日志
        summary = []
        for res in results:
            val_losses = [e["val_loss"] for e in res["logs"] if "val_loss" in e]
            train_losses = [e["train_loss"] for e in res["logs"] if "train_loss" in e]
            summary.append({
                "lr": res["lr"],
                "success": res["success"],
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "min_val_loss": min(val_losses) if val_losses else None,
            })
        json.dump(summary, f, indent=2)
    
    # 打印摘要
    print_summary(results)
    
    # 绘图
    plot_path = output_dir / "lr_sweep_curves.png"
    plot_sweep_results(results, plot_path)


if __name__ == "__main__":
    main()
