"""
共享训练逻辑：train_tinystories.py 与 train_openwebtext.py 共用。

不直接运行；由各数据集脚本解析配置后调用 run_training(config)。
"""

from pathlib import Path

import numpy as np
import torch

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


def get_project_root() -> Path:
    """脚本所在项目根目录 (llm/)"""
    return Path(__file__).parent.parent.parent


def load_data(project_root: Path, path: str):
    """使用 memory-mapped 模式加载 tokenized 数据"""
    full_path = project_root / path
    if not full_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {full_path}")
    data = np.load(full_path, mmap_mode="r")
    print(f"加载数据: {full_path}")
    print(f"  - 形状: {data.shape}, 类型: {data.dtype}, Tokens: {len(data):,}")
    return data


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, num_batches=50):
    """在验证集上评估"""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_training(config: ExperimentConfig, project_root: Path | None = None):
    """
    执行完整训练流程。config 可为 ExperimentConfig（TinyStories）或 OWTExperimentConfig（OpenWebText）。
    """
    if project_root is None:
        project_root = get_project_root()

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        config.device = "cpu"
    device = torch.device(config.device)
    print(f"使用设备: {device}")

    total_tokens = config.batch_size * config.max_steps * config.context_length
    print(f"\n训练配置:")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max steps: {config.max_steps}")
    print(f"  - Context length: {config.context_length}")
    print(f"  - 总 tokens: {total_tokens:,} ({total_tokens / 1e6:.1f}M)")
    print(f"  - Vocab size: {config.vocab_size}")

    print("\n加载数据...")
    train_data = load_data(project_root, config.train_data_path)
    val_data = load_data(project_root, config.val_data_path)

    print("\n创建模型...")
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        use_rmsnorm=config.use_rmsnorm,
        norm_type=config.norm_type,
        position_encoding=config.position_encoding,
        ffn_type=config.ffn_type,
        device=device,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"  - 总参数: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  - 可训练参数: {trainable_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    start_step = 0
    if config.resume:
        print(f"\n从 checkpoint 恢复: {config.resume}")
        start_step = load_checkpoint(config.resume, model, optimizer)
        print(f"  - 恢复到 step {start_step}")

    checkpoint_dir = Path(project_root / config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(config, log_dir=checkpoint_dir / "logs")
    logger.start()
    loss_avg = MovingAverage(window_size=config.log_interval)

    print("\n开始训练...")
    model.train()
    for step in range(start_step, config.max_steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_iters=config.warmup_steps,
            cosine_cycle_iters=config.max_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        inputs, targets = get_batch(train_data, config.batch_size, config.context_length, config.device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), config.max_grad_norm)
        optimizer.step()

        loss_value = loss.item()
        loss_avg.update(loss_value)
        if (step + 1) % config.log_interval == 0:
            logger.log(step + 1, {"train_loss": loss_avg.get(), "lr": lr})
        if (step + 1) % config.eval_interval == 0:
            val_loss = evaluate(model, val_data, config.batch_size, config.context_length, config.device)
            perplexity = np.exp(val_loss)
            logger.log(step + 1, {"val_loss": val_loss, "perplexity": perplexity})
            print(f"  [验证] val_loss: {val_loss:.4f}, perplexity: {perplexity:.2f}")
        if (step + 1) % config.save_interval == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_step_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"  [保存] {ckpt_path}")

    logger.finish()
    final_ckpt_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, config.max_steps, final_ckpt_path)
    print(f"\n训练完成！最终模型保存到: {final_ckpt_path}")

    final_val_loss = evaluate(model, val_data, config.batch_size, config.context_length, config.device, num_batches=100)
    final_perplexity = np.exp(final_val_loss)
    print(f"最终验证 loss: {final_val_loss:.4f}")
    print(f"最终 perplexity: {final_perplexity:.2f}")

    # TinyStories 作业目标 (val_loss ≤ 1.45)；OWT 不适用此目标
    if config.vocab_size == 10000:
        if final_val_loss <= 1.45:
            print("🎉 达到目标 (val_loss ≤ 1.45)!")
        else:
            print(f"⚠️ 未达到目标 (当前: {final_val_loss:.4f}, 目标: ≤ 1.45)")

    # 自动绘制 loss 曲线到 checkpoint 目录
    log_file = checkpoint_dir / "logs" / "logs.json"
    out_file = checkpoint_dir / "loss_curve.png"
    if log_file.exists():
        try:
            from plot_loss import load_logs, plot_loss_curves
            data = load_logs(log_file)
            title = "OpenWebText LM Training" if config.vocab_size == 32000 else "TinyStories LM Training"
            plot_loss_curves(data, out_file, title)
            print(f"Loss 曲线已保存: {out_file}")
        except Exception as e:
            print(f"绘制 loss 曲线失败: {e}")
