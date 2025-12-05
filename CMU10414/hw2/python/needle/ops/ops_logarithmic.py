from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_val = array_api.max(Z, axis=(1,), keepdims=True)
        shifted = Z - max_val
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(shifted), axis=(1,), keepdims=True))
        return shifted - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax = exp(node)
        sum_grad = summation(out_grad, axes=(1,))
        sum_grad_reshaped = reshape(sum_grad, (node.shape[0], 1))
        sum_grad_broadcast = broadcast_to(sum_grad_reshaped, node.shape)
        return out_grad - softmax * sum_grad_broadcast
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_val = array_api.max(Z, axis=self.axes, keepdims=True)
        shifted = Z - max_val
        sum_exp = array_api.sum(array_api.exp(shifted), axis=self.axes)
        log_sum_exp = array_api.log(sum_exp)
        max_val_squeezed = array_api.max(Z, axis=self.axes)
        result = log_sum_exp + max_val_squeezed
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        input_shape = Z.shape
        
        # node 是 logsumexp 的输出
        # 需要将 out_grad 和 node reshape 回输入的形状
        if self.axes is not None:
            # 构造目标形状（在求和轴上插入维度1）
            new_shape = list(input_shape)
            for axis in self.axes:
                new_shape[axis] = 1
            # reshape out_grad 和 node 到可以 broadcast 的形状
            out_grad_reshaped = reshape(out_grad, tuple(new_shape))
            node_reshaped = reshape(node, tuple(new_shape))
        else:
            # axes=None 意味着在所有轴上求和，结果是标量
            new_shape = tuple([1] * len(input_shape))
            out_grad_reshaped = reshape(out_grad, new_shape)
            node_reshaped = reshape(node, new_shape)
        
        # broadcast 到输入形状
        out_grad_broadcast = broadcast_to(out_grad_reshaped, input_shape)
        node_broadcast = broadcast_to(node_reshaped, input_shape)
        
        # softmax = exp(Z - logsumexp(Z))
        # gradient = out_grad * softmax
        softmax = exp(Z - node_broadcast)
        return out_grad_broadcast * softmax
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)