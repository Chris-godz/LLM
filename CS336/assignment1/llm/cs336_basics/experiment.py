"""
实验追踪基础设施

功能：
- 记录训练/验证 loss
- 追踪梯度步数和墙钟时间
- 支持 Weights and Biases (wandb) 集成
- 支持本地 JSON 日志
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import os

from simple_parsing import field as sp_field


@dataclass
class ExperimentConfig:
    """实验配置 - 支持命令行解析"""
    
    # 模型配置
    vocab_size: int = sp_field(default=10000, alias="-vs", help="词表大小")
    context_length: int = sp_field(default=256, alias="-ctx", help="上下文长度")
    d_model: int = sp_field(default=512, help="模型隐藏层维度")
    d_ff: int = sp_field(default=1344, help="FFN 中间层维度")
    num_layers: int = sp_field(default=4, alias="-nl", help="Transformer 层数")
    num_heads: int = sp_field(default=16, alias="-nh", help="注意力头数")
    rope_theta: float = sp_field(default=10000.0, help="RoPE theta 参数")
    
    # 训练配置 (CS336 Assignment 1: 327.68M tokens = batch_size * context_length * max_steps)
    batch_size: int = sp_field(default=64, alias="-bs", help="批次大小")
    max_steps: int = sp_field(default=20000, help="最大训练步数 (64*256*20000=327.68M tokens)")
    learning_rate: float = sp_field(default=1e-3, alias="-lr", help="最大学习率")
    min_learning_rate: float = sp_field(default=1e-5, help="最小学习率")
    warmup_steps: int = sp_field(default=400, help="学习率 warmup 步数 (约 2% of max_steps)")
    weight_decay: float = sp_field(default=0.1, alias="-wd", help="权重衰减")
    beta1: float = sp_field(default=0.9, help="AdamW beta1")
    beta2: float = sp_field(default=0.95, help="AdamW beta2")
    epsilon: float = sp_field(default=1e-8, help="AdamW epsilon")
    max_grad_norm: float = sp_field(default=1.0, help="梯度裁剪最大范数")
    
    # 数据配置
    train_data_path: str = sp_field(default="scripts/tokenization/TinyStoriesV2-GPT4-train.npy", help="训练数据路径")
    val_data_path: str = sp_field(default="scripts/tokenization/TinyStoriesV2-GPT4-valid.npy", help="验证数据路径")
    tokenizer_vocab_path: str = sp_field(default="scripts/tokenization/tinystory/vocab_tinystories.pkl", help="词表路径")
    tokenizer_merges_path: str = sp_field(default="", help="BPE merges 路径")
    
    # 日志配置
    log_interval: int = sp_field(default=10, help="每多少步打印一次")
    eval_interval: int = sp_field(default=500, help="每多少步评估一次")
    save_interval: int = sp_field(default=2000, help="每多少步保存一次")
    checkpoint_dir: str = sp_field(default="scripts/training/checkpoints", help="Checkpoint 保存目录")
    
    # 实验标识
    experiment_name: str = sp_field(default="tinystories_lm", alias="-name", help="实验名称")
    run_name: Optional[str] = sp_field(default=None, help="运行名称 (用于 wandb)")
    
    # wandb 配置
    use_wandb: bool = sp_field(default=False, help="是否使用 wandb")
    wandb_project: str = sp_field(default="cs336-assignment1", help="wandb 项目名")
    wandb_entity: Optional[str] = sp_field(default=None, help="wandb 实体名")
    
    # 设备配置
    device: str = sp_field(default="cuda", help="训练设备 (cuda/cpu/mps)")
    
    # 恢复训练
    resume: Optional[str] = sp_field(default=None, help="从 checkpoint 恢复训练的路径")
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class LogEntry:
    """单条日志记录"""
    step: int
    wallclock_time: float  # 从训练开始的秒数
    metrics: dict = field(default_factory=dict)


class ExperimentLogger:
    """
    实验日志记录器
    
    支持：
    - 控制台输出
    - JSON 文件记录
    - Weights and Biases 集成
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        log_dir: Optional[str | Path] = None,
    ):
        self.config = config
        self.start_time: Optional[float] = None
        self.current_step = 0
        self.logs: list[LogEntry] = []
        
        # 设置日志目录
        if log_dir is None:
            log_dir = Path("logs") / config.experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint 目录
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # wandb 初始化
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()
        
        # 保存配置
        self.config.save(self.log_dir / "config.json")
    
    def _init_wandb(self):
        """初始化 Weights and Biases"""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name or self.config.experiment_name,
                config=self.config.to_dict(),
            )
            print(f"[wandb] 已初始化: {wandb.run.url}")
        except Exception as e:
            print(f"[wandb] 初始化失败: {e}")
            self.wandb_run = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.current_step = 0
        print(f"[实验开始] {self.config.experiment_name}")
        print(f"  - 日志目录: {self.log_dir}")
        print(f"  - Checkpoint 目录: {self.checkpoint_dir}")
    
    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def log(
        self,
        step: int,
        metrics: dict[str, Any],
        print_to_console: bool = True,
    ):
        """
        记录一条日志
        
        Args:
            step: 当前步数
            metrics: 指标字典，例如 {"train_loss": 2.5, "lr": 1e-3}
            print_to_console: 是否打印到控制台
        """
        self.current_step = step
        elapsed = self.get_elapsed_time()
        
        # 创建日志条目
        entry = LogEntry(step=step, wallclock_time=elapsed, metrics=metrics.copy())
        self.logs.append(entry)
        
        # 控制台输出
        if print_to_console:
            self._print_metrics(step, elapsed, metrics)
        
        # wandb 记录
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    "step": step,
                    "wallclock_time": elapsed,
                    **metrics,
                })
            except Exception as e:
                print(f"[wandb] 记录失败: {e}")
    
    def _print_metrics(self, step: int, elapsed: float, metrics: dict):
        """打印指标到控制台"""
        elapsed_str = self._format_time(elapsed)
        metrics_str = " | ".join(f"{k}: {self._format_value(v)}" for k, v in metrics.items())
        print(f"[Step {step:6d}] [{elapsed_str}] {metrics_str}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.2f}h"
    
    @staticmethod
    def _format_value(v: Any) -> str:
        """格式化数值"""
        if isinstance(v, float):
            if abs(v) < 0.001 or abs(v) > 1000:
                return f"{v:.2e}"
            return f"{v:.4f}"
        return str(v)
    
    def save_logs(self, filename: str = "logs.json"):
        """保存日志到 JSON 文件"""
        logs_data = [
            {
                "step": entry.step,
                "wallclock_time": entry.wallclock_time,
                **entry.metrics,
            }
            for entry in self.logs
        ]
        with open(self.log_dir / filename, "w") as f:
            json.dump(logs_data, f, indent=2)
    
    def finish(self):
        """结束实验"""
        elapsed = self.get_elapsed_time()
        print(f"\n[实验结束] 总用时: {self._format_time(elapsed)}")
        print(f"  - 总步数: {self.current_step}")
        print(f"  - 平均速度: {self.current_step / elapsed:.1f} steps/s")
        
        # 保存日志
        self.save_logs()
        
        # 关闭 wandb
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except:
                pass


class Timer:
    """简单的计时器，用于测量代码块的执行时间"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")
    
    def reset(self):
        self.start_time = time.time()
    
    def get_elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


class MovingAverage:
    """计算移动平均值，用于平滑 loss 曲线"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values: list[float] = []
    
    def update(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self):
        self.values.clear()


def estimate_tokens_per_second(
    batch_size: int,
    context_length: int,
    step_time: float,
) -> float:
    """估算 tokens/秒吞吐量"""
    tokens_per_step = batch_size * context_length
    return tokens_per_step / step_time


def estimate_time_to_completion(
    current_step: int,
    total_steps: int,
    elapsed_time: float,
) -> float:
    """估算剩余时间（秒）"""
    if current_step == 0:
        return float('inf')
    steps_remaining = total_steps - current_step
    time_per_step = elapsed_time / current_step
    return steps_remaining * time_per_step
