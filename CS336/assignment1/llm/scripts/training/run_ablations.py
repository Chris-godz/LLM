#!/usr/bin/env python3
"""
运行消融实验脚本

该脚本自动运行一系列消融实验，对比不同的模型架构选择。
基线配置基于之前的实验结果 (LR=3e-3, Batch Size=64)。

实验列表:
1. Baseline (Pre-norm, RoPE, SwiGLU, RMSNorm)
2. Ablation: No RMSNorm
3. Ablation: Post-norm
4. Ablation: No Position Embeddings (NoPE)
5. Ablation: SiLU FFN (instead of SwiGLU)

用法:
    uv run python scripts/training/run_ablations.py
"""

import sys
import subprocess
import shutil
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "training" / "train_tinystories.py"
PLOT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "training" / "plot_loss.py"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "training" / "ablation_results"

# 基线参数 (固定)
# 根据之前的 sweep，最佳 lr 是 3e-3
BASELINE_ARGS = [
    "--learning_rate", "3e-3",
    "--batch_size", "108",
    "--max_steps", "20000",   # 保持与 assignment 一致
    "--log_interval", "10",
    "--eval_interval", "500",
    "--save_interval", "5000",
]

# 实验定义
EXPERIMENTS = [
    {
        "name": "baseline",
        "description": "Baseline: Pre-norm, RMSNorm, RoPE, SwiGLU",
        "args": []
    },
    {
        "name": "ablation_no_rmsnorm",
        "description": "Remove RMSNorm",
        "args": ["--use_rmsnorm", "False"] 
    },
    {
        "name": "ablation_post_norm",
        "description": "Post-norm Transformer",
        "args": ["--norm_type", "post"]
    },
    {
        "name": "ablation_no_pos_emb",
        "description": "No Position Embeddings (NoPE)",
        "args": ["--position_encoding", "none"]
    },
    {
        "name": "ablation_silu",
        "description": "SiLU FFN (d_ff adjusted to 2048)",
        "args": ["--ffn_type", "silu", "--d_ff", "2048"]
    }
]

def run_command(command, log_file=None):
    """运行 shell 命令"""
    print(f"Running: {' '.join(command)}")
    try:
        if log_file:
            with open(log_file, "w") as f:
                subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        # 如果不是 kill 信号，则退出
        if e.returncode != -9:
            # sys.exit(1) # 也可以选择继续跑下一个
            pass 

def create_summary(results_dir: Path):
    """生成汇总报告"""
    summary_path = results_dir / "summary.md"
    
    summary = ["# Ablation Study Summary\n"]
    summary.append("| Experiment | Min Val Loss | Step | Final Val Loss | Description |")
    summary.append("|------------|--------------|------|----------------|-------------|")
    
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        exp_dir = results_dir / exp_name
        log_file = exp_dir / "logs" / "logs.json"
        
        if not log_file.exists():
            summary.append(f"| {exp_name} | N/A | N/A | N/A | {exp['description']} (Failed/Missing) |")
            continue
            
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
            
            val_losses = [(entry["step"], entry["val_loss"]) for entry in logs if "val_loss" in entry]
            
            if val_losses:
                min_loss_step, min_loss = min(val_losses, key=lambda x: x[1])
                final_loss = val_losses[-1][1]
                
                summary.append(f"| {exp_name} | {min_loss:.4f} | {min_loss_step} | {final_loss:.4f} | {exp['description']} |")
            else:
                summary.append(f"| {exp_name} | N/A | N/A | N/A | {exp['description']} (No val logs) |")

        except Exception as e:
            print(f"Error reading logs for {exp_name}: {e}")
            summary.append(f"| {exp_name} | Error | Error | Error | {exp['description']} |")

    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    
    print(f"\n汇总报告已生成: {summary_path}")

def plot_ablation_comparison(results_dir: Path):
    """绘制消融实验对比曲线。发散 run（如 No RMSNorm）会被过滤，避免 y 轴被拉爆。"""
    output_path = results_dir / "ablation_comparison.png"
    # 超过此阈值视为发散，不参与主图 y 轴缩放（仅画到阈值内或单独标注）
    LOSS_DIVERGE_THRESHOLD = 50.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(EXPERIMENTS)))

    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp["name"]
        log_file = results_dir / exp_name / "logs" / "logs.json"
        if not log_file.exists():
            continue
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
            train_steps = [e["step"] for e in logs if "train_loss" in e]
            train_losses = np.array([e["train_loss"] for e in logs if "train_loss" in e])
            val_steps = [e["step"] for e in logs if "val_loss" in e]
            val_losses = np.array([e["val_loss"] for e in logs if "val_loss" in e])

            # 发散数据截断：只画 loss < 阈值的部分，避免一根线把 y 轴拉到 1e26
            train_ok = train_losses < LOSS_DIVERGE_THRESHOLD
            val_ok = val_losses < LOSS_DIVERGE_THRESHOLD
            if not np.any(train_ok) and not np.any(val_ok):
                continue
            label = exp["description"].split(":")[0] if ":" in exp["description"] else exp["description"]
            if exp_name == "ablation_no_rmsnorm":
                label = label + " (发散，仅显示前期)"

            if np.any(train_ok):
                steps_t = [train_steps[j] for j in range(len(train_steps)) if train_ok[j]]
                loss_t = train_losses[train_ok].tolist()
                if len(loss_t) > 10:
                    window = 10
                    smooth = np.convolve(loss_t, np.ones(window) / window, mode="valid")
                    ax1.plot(steps_t[window - 1 :], smooth, label=label, color=colors[i], linewidth=1.5, alpha=0.9)
                else:
                    ax1.plot(steps_t, loss_t, label=label, color=colors[i], linewidth=1.5, alpha=0.9)
            if np.any(val_ok):
                steps_v = [val_steps[j] for j in range(len(val_steps)) if val_ok[j]]
                loss_v = val_losses[val_ok].tolist()
                ax2.plot(steps_v, loss_v, label=label, color=colors[i], marker="o", markersize=4, linewidth=1.5)
        except Exception as e:
            print(f"Error reading logs for plotting {exp_name}: {e}")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss (Smoothed)")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1.0, 2.0)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Validation Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1.2, 1.6)
    plt.suptitle("Ablation Studies Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n对比曲线已保存: {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run these experiment names (e.g. --only ablation_post_norm ablation_no_pos_emb ablation_silu)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only regenerate summary and ablation_comparison.png from existing logs")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.plot_only:
        create_summary(RESULTS_DIR)
        plot_ablation_comparison(RESULTS_DIR)
        return
    
    experiments_to_run = EXPERIMENTS
    if args.only:
        names = set(args.only)
        experiments_to_run = [e for e in EXPERIMENTS if e["name"] in names]
        if len(experiments_to_run) != len(names):
            missing = names - {e["name"] for e in experiments_to_run}
            print(f"Warning: unknown experiment names: {missing}")
        print(f"Only running: {[e['name'] for e in experiments_to_run]}\n")
    
    MAX_STEPS = 20000  # 与 BASELINE_ARGS 一致，用于判断是否已完成

    for exp in experiments_to_run:
        exp_name = exp["name"]
        print(f"\n{'='*20} Running Experiment: {exp_name} {'='*20}")
        print(f"Description: {exp['description']}")
        
        exp_dir = RESULTS_DIR / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 仅当实验已完成（logs.json 存在且跑满步数）时才跳过
        log_file = exp_dir / "logs" / "logs.json"
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
                steps = [e["step"] for e in logs if "step" in e]
                if steps and max(steps) >= MAX_STEPS:
                    print(f"实验已完成 (max step {max(steps)})，跳过: {exp_name}")
                    continue
            except Exception:
                pass
            print(f"实验未完成或损坏，重新运行: {exp_name}")
        
        # 构造训练命令
        cmd = [
            "uv", "run", "python", "-u", str(SCRIPT_PATH),
            "--checkpoint_dir", str(exp_dir)
        ] + BASELINE_ARGS + exp["args"]
        
        # 运行训练
        run_command(cmd, log_file=exp_dir / "stdout.log") # Log logs to file
        
        # 绘制单个实验的 Loss 曲线
        if (exp_dir / "logs" / "logs.json").exists():
            print(f"绘制曲线: {exp_name}")
            plot_cmd = [
                "uv", "run", "python", str(PLOT_SCRIPT_PATH),
                "--log_file", str(exp_dir / "logs" / "logs.json"),
                "--output", str(exp_dir / "loss_curve.png"),
                "--title", f"Ablation: {exp['description']}"
            ]
            run_command(plot_cmd)
        
    print(f"\n{'='*50}")
    print("所有实验完成！正在生成报告...")
    create_summary(RESULTS_DIR)
    plot_ablation_comparison(RESULTS_DIR)

if __name__ == "__main__":
    main()
