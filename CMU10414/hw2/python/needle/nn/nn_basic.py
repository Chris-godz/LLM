"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            out = out + ops.broadcast_to(self.bias, out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # (B, X_0, X_1, ...) -> (B, X_0 * X_1 * ...)
        batch_size = X.shape[0]
        flattened_dim = 1
        for dim in X.shape[1:]:
            flattened_dim *= dim
        return ops.reshape(X, (batch_size, flattened_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        from functools import reduce
        return reduce(lambda data, module: module(data), self.modules, x)
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # logits: (batch_size, num_classes)
        # y: (batch_size,) - 真实标签（数字）
        batch_size, num_classes = logits.shape
        
        # logsumexp(logits) 沿 axis=1 求和
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        
        # 构造 one-hot 编码
        y_one_hot = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)
        
        # z_y = sum(logits * one_hot, axis=1)
        z_y = ops.summation(logits * y_one_hot, axes=(1,))
        
        # loss = logsumexp - z_y，然后求平均
        loss = log_sum_exp - z_y
        return ops.summation(loss) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: (batch_size, dim)
        batch_size, dim = x.shape
        
        if self.training:
            # 计算 batch 均值和方差
            mean = ops.summation(x, axes=(0,)) / batch_size  # (dim,)
            mean_broadcast = ops.broadcast_to(ops.reshape(mean, (1, dim)), x.shape)
            
            var = ops.summation((x - mean_broadcast) ** 2, axes=(0,)) / batch_size  # (dim,)
            var_broadcast = ops.broadcast_to(ops.reshape(var, (1, dim)), x.shape)
            
            # 更新 running mean 和 running var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            
            # 归一化
            x_norm = (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5)
        else:
            # 使用 running mean 和 running var
            mean_broadcast = ops.broadcast_to(ops.reshape(self.running_mean, (1, dim)), x.shape)
            var_broadcast = ops.broadcast_to(ops.reshape(self.running_var, (1, dim)), x.shape)
            x_norm = (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5)
        
        # 应用 weight 和 bias
        weight = ops.broadcast_to(ops.reshape(self.weight, (1, dim)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, dim)), x.shape)
        
        return weight * x_norm + bias
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: (batch_size, dim)
        batch_size, dim = x.shape
        
        # 计算均值 E[x]，沿 axis=1
        mean = ops.summation(x, axes=(1,)) / dim
        mean = ops.reshape(mean, (batch_size, 1))
        mean = ops.broadcast_to(mean, x.shape)
        
        # 计算方差 Var[x]
        var = ops.summation((x - mean) ** 2, axes=(1,)) / dim
        var = ops.reshape(var, (batch_size, 1))
        var = ops.broadcast_to(var, x.shape)
        
        # 归一化
        x_norm = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
        
        # 应用 weight 和 bias
        weight = ops.broadcast_to(ops.reshape(self.weight, (1, dim)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, dim)), x.shape)
        
        return weight * x_norm + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # 生成 mask：以 1-p 的概率保留
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            # 除以 (1-p) 进行缩放
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
