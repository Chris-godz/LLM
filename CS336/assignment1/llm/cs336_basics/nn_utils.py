import math
import os
from typing import BinaryIO, Callable, IO, Iterable

import numpy as np
import numpy.typing as npt
import torch


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer (decoupled weight decay).
    
    Algorithm:
        m = β₁*m + (1-β₁)*g           # first moment estimate
        v = β₂*v + (1-β₂)*g²          # second moment estimate
        αₜ = α * √(1-β₂ᵗ) / (1-β₁ᵗ)  # bias correction
        θ = θ - αₜ * m / (√v + ε)     # parameter update
        θ = θ - α*λ*θ                  # weight decay (decoupled)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # Increment step counter (t starts at 1)
                state["t"] += 1
                t = state["t"]

                m = state["m"]
                v = state["v"]

                # Update biased first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Apply weight decay (decoupled)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients to have l2 norm at most max_l2_norm.
    Modifies gradients in-place.
    """
    eps = 1e-6
    
    # Compute total L2 norm of all gradients
    total_norm_sq = 0.0
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)
            total_norm_sq += p.grad.data.pow(2).sum().item()
    
    total_norm = math.sqrt(total_norm_sq)
    
    # Clip if needed
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.data.mul_(scale)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    
    - Warmup:        t < Tw        → αₜ = (t / Tw) * αmax
    - Cosine:        Tw ≤ t ≤ Tc  → αₜ = αmin + 0.5*(1 + cos((t-Tw)/(Tc-Tw)*π))*(αmax - αmin)
    - Post-anneal:   t > Tc       → αₜ = αmin
    """
    if it < warmup_iters:
        # Linear warmup
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        # Post-annealing
        return min_learning_rate


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute average cross-entropy loss for logits and integer targets.

    Args:
        inputs: (..., vocab_size) unnormalized logits.
        targets: (...) integer class indices.

    Returns:
        Scalar tensor with mean cross-entropy across batch-like dims.
    """
    # Work in float32 for numerical stability.
    logits = inputs.to(torch.float32)

    # logsumexp over vocab dimension with max trick.
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    logsumexp = max_logits.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=-1))

    # Gather logits at target indices.
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = logsumexp - target_logits
    return loss.mean()


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and their corresponding labels.
    
    Args:
        dataset: 1D numpy array of token IDs.
        batch_size: Number of sequences in the batch.
        context_length: Length of each sequence.
        device: PyTorch device string.
    
    Returns:
        (inputs, targets) both of shape (batch_size, context_length).
        inputs[i] = dataset[start:start+context_length]
        targets[i] = dataset[start+1:start+context_length+1]
    """
    # Maximum valid starting index
    max_start = len(dataset) - context_length - 1
    
    # Randomly sample starting indices
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)
    
    # Build input and target sequences
    inputs = np.stack([dataset[i : i + context_length] for i in start_indices])
    targets = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices])
    
    # Convert to tensors and move to device
    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)
    
    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model, optimizer state, and iteration number to a checkpoint file.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model and optimizer state from a checkpoint file.
    
    Returns:
        The iteration number saved in the checkpoint.
    """
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
