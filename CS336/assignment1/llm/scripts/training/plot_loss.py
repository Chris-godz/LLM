#!/usr/bin/env python3
"""
绘制训练 loss 曲线

用法:
    uv run python scripts/training/plot_loss.py
    uv run python scripts/training/plot_loss.py --log_file path/to/logs.json --output loss_curve.png
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_logs(log_file: Path) -> dict:
    """加载 JSON 日志文件"""
    with open(log_file, "r") as f:
        logs = json.load(f)
    
    # 分离不同指标
    data = {
        "step": [],
        "wallclock_time": [],
        "train_loss": [],
        "val_loss": [],
        "perplexity": [],
        "lr": [],
    }
    
    for entry in logs:
        step = entry.get("step")
        wallclock = entry.get("wallclock_time")
        
        if "train_loss" in entry:
            data["step"].append(step)
            data["wallclock_time"].append(wallclock)
            data["train_loss"].append(entry["train_loss"])
            data["lr"].append(entry.get("lr", None))
        
        if "val_loss" in entry:
            data["val_loss"].append((step, entry["val_loss"]))
            data["perplexity"].append((step, entry.get("perplexity", np.exp(entry["val_loss"]))))
    
    return data


def plot_loss_curves(data: dict, output_path: Path, title: str = "Training Loss Curves"):
    """绘制 loss 曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Train Loss vs Step
    ax1 = axes[0, 0]
    ax1.plot(data["step"], data["train_loss"], label="Train Loss", alpha=0.8)
    if data["val_loss"]:
        val_steps, val_losses = zip(*data["val_loss"])
        ax1.plot(val_steps, val_losses, label="Val Loss", marker='o', markersize=3)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss vs Wallclock Time
    ax2 = axes[0, 1]
    time_hours = [t / 3600 for t in data["wallclock_time"]]
    ax2.plot(time_hours, data["train_loss"], label="Train Loss", alpha=0.8)
    if data["val_loss"]:
        # 需要找到对应的 wallclock time
        val_times = []
        for step, _ in data["val_loss"]:
            idx = data["step"].index(step) if step in data["step"] else -1
            if idx >= 0:
                val_times.append(data["wallclock_time"][idx] / 3600)
        if val_times:
            ax2.plot(val_times, [v[1] for v in data["val_loss"]], label="Val Loss", marker='o', markersize=3)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Wallclock Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedule
    ax3 = axes[1, 0]
    lr_values = [lr for lr in data["lr"] if lr is not None]
    if lr_values:
        ax3.plot(data["step"][:len(lr_values)], lr_values)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Learning Rate")
        ax3.set_title("Learning Rate Schedule")
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # 4. Perplexity
    ax4 = axes[1, 1]
    if data["perplexity"]:
        ppl_steps, ppl_values = zip(*data["perplexity"])
        ax4.plot(ppl_steps, ppl_values, marker='o', markersize=3, color='green')
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Perplexity")
        ax4.set_title("Validation Perplexity")
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"保存图片到: {output_path}")
    plt.close()


def print_summary(data: dict):
    """打印训练摘要"""
    print("\n" + "=" * 50)
    print("训练摘要")
    print("=" * 50)
    
    if data["train_loss"]:
        print(f"总步数: {data['step'][-1]}")
        print(f"总时间: {data['wallclock_time'][-1] / 3600:.2f} 小时")
        print(f"最终 Train Loss: {data['train_loss'][-1]:.4f}")
    
    if data["val_loss"]:
        final_val = data["val_loss"][-1]
        print(f"最终 Val Loss: {final_val[1]:.4f}")
        
        # 找到最低 val loss
        min_val = min(data["val_loss"], key=lambda x: x[1])
        print(f"最低 Val Loss: {min_val[1]:.4f} (step {min_val[0]})")
    
    if data["perplexity"]:
        final_ppl = data["perplexity"][-1]
        print(f"最终 Perplexity: {final_ppl[1]:.2f}")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("--log_file", type=str, 
                        default="scripts/training/checkpoints/logs/logs.json",
                        help="日志文件路径")
    parser.add_argument("--output", type=str, 
                        default="scripts/training/checkpoints/loss_curves.png",
                        help="输出图片路径")
    parser.add_argument("--title", type=str, 
                        default="TinyStories LM Training",
                        help="图表标题")
    
    args = parser.parse_args()
    
    # 项目根目录
    project_root = Path(__file__).parent.parent.parent
    log_file = project_root / args.log_file
    output_path = project_root / args.output
    
    if not log_file.exists():
        print(f"错误: 日志文件不存在: {log_file}")
        return
    
    print(f"加载日志: {log_file}")
    data = load_logs(log_file)
    
    print_summary(data)
    plot_loss_curves(data, output_path, args.title)


if __name__ == "__main__":
    main()
