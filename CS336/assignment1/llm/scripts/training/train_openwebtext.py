#!/usr/bin/env python3
"""
OpenWebText 语言模型训练脚本 (CS336 7.4 main_experiment)

与 train_tinystories.py 同风格：按数据集命名，simple_parsing / OWTExperimentConfig。
与 TinyStories 相同的模型架构与总训练步数，仅数据与词表为 OWT（32K vocab）。
需事先准备好 owt_train.npy / owt_valid.npy。

用法：
    uv run python scripts/training/train_openwebtext.py
    uv run python scripts/training/train_openwebtext.py --checkpoint_dir scripts/training/checkpoints/openwebtext
  ~16GB 显存若 OOM，可减 batch 并加步数以对齐 baseline 总 tokens：--batch_size 32 --max_steps 67500；或 --batch_size 64 --max_steps 33750（64*256*33750=553M）

7.4 交付 (writeup)：① Learning curve（训练后用 plot_loss.py 画）；② 生成文本（用 generate_text.py，
  --checkpoint_dir scripts/training/checkpoints/openwebtext --tokenizer_vocab_path scripts/tokenization/openweb/vocab_owt.pkl --vocab_size 32000）。
  说明与 TinyStories 的 loss 差异、生成流畅度及为何同 compute 下质量更差。
"""

import sys
from pathlib import Path

from simple_parsing import ArgumentParser

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from cs336_basics.experiment import OWTExperimentConfig
from _training_common import run_training


def parse_args() -> OWTExperimentConfig:
    """解析命令行参数，返回 OpenWebText 默认的 OWTExperimentConfig"""
    parser = ArgumentParser(description="Train OpenWebText LM (7.4 main_experiment)")
    parser.add_arguments(OWTExperimentConfig, dest="config")
    args = parser.parse_args()
    return args.config


def main():
    config = parse_args()
    run_training(config, project_root=project_root)


if __name__ == "__main__":
    main()
