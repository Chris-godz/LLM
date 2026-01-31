#!/usr/bin/env python3
"""
TinyStories 语言模型训练脚本

使用方法：
    uv run python scripts/training/train_tinystories.py
    uv run python scripts/training/train_tinystories.py --learning_rate 5e-4 --batch_size 128
"""

import sys
from pathlib import Path

from simple_parsing import ArgumentParser

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from cs336_basics.experiment import ExperimentConfig
from _training_common import run_training


def parse_args() -> ExperimentConfig:
    """解析命令行参数，返回 TinyStories 默认的 ExperimentConfig"""
    parser = ArgumentParser(description="Train TinyStories LM")
    parser.add_arguments(ExperimentConfig, dest="config")
    args = parser.parse_args()
    return args.config


def main():
    config = parse_args()
    run_training(config, project_root=project_root)


if __name__ == "__main__":
    main()
