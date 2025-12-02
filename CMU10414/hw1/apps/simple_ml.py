"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # 读取图像文件
    with gzip.open(image_filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8)
        X = X.reshape(num_images, rows * cols).astype(np.float32)
        X = X / 255.0  # 归一化到 [0, 1]
    
    # 读取标签文件
    with gzip.open(label_filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
    
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # softmax loss: log(sum(exp(z_i))) - z_y
    # 对批次求平均
    batch_size = Z.shape[0]
    
    # log(sum(exp(Z))) 对每个样本
    log_sum_exp = ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=(1,)))
    
    # z_y: 选出真实标签对应的 logit
    z_y = ndl.ops.summation(Z * y_one_hot, axes=(1,))
    
    # 平均损失
    loss = ndl.ops.summation(log_sum_exp - z_y) / batch_size
    
    return loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    
    # 按批次遍历数据
    for i in range(0, num_examples, batch):
        # 获取当前批次
        batch_end = min(i + batch, num_examples)
        X_batch = ndl.Tensor(X[i:batch_end])
        y_batch = y[i:batch_end]
        
        # 构造 one-hot 标签
        y_one_hot = np.zeros((y_batch.shape[0], num_classes))
        y_one_hot[np.arange(y_batch.size), y_batch] = 1
        y_one_hot_tensor = ndl.Tensor(y_one_hot)
        
        # 前向传播: logits = ReLU(X @ W1) @ W2
        Z1 = ndl.ops.matmul(X_batch, W1)
        A1 = ndl.ops.relu(Z1)
        logits = ndl.ops.matmul(A1, W2)
        
        # 计算损失
        loss = softmax_loss(logits, y_one_hot_tensor)
        
        # 反向传播
        loss.backward()
        
        # 更新权重（SGD）
        W1_new_data = W1.realize_cached_data() - lr * W1.grad.realize_cached_data()
        W2_new_data = W2.realize_cached_data() - lr * W2.grad.realize_cached_data()
        
        # 创建新的 Tensor（需要梯度）
        W1 = ndl.Tensor(W1_new_data, requires_grad=True)
        W2 = ndl.Tensor(W2_new_data, requires_grad=True)
    
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
