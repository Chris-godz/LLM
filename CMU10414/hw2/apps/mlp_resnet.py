import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # ResidualBlock 结构：
    # Linear → Norm → ReLU → Dropout → Linear → Norm
    # 然后加上残差连接
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    return nn.Sequential(
        nn.Residual(fn),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    # MLPResNet 结构：
    # Linear → ReLU → [ResidualBlock × num_blocks] → Linear
    layers = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 设置训练/评估模式
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    loss_fn = nn.SoftmaxLoss()
    total_loss = 0.0
    total_error = 0.0
    total_samples = 0
    
    for X, y in dataloader:
        # 展平输入：(batch, 28, 28, 1) -> (batch, 784)
        batch_size = X.shape[0]
        X = X.reshape((batch_size, -1))
        
        # 前向传播
        logits = model(X)
        loss = loss_fn(logits, y)
        
        # 计算错误率
        predictions = np.argmax(logits.numpy(), axis=1)
        errors = np.sum(predictions != y.numpy())
        
        total_loss += loss.numpy() * batch_size
        total_error += errors
        total_samples += batch_size
        
        # 反向传播（仅训练时）
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
    
    avg_error = total_error / total_samples
    avg_loss = total_loss / total_samples
    return avg_error, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 加载数据
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz"
    )
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    
    train_loader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型和优化器
    # MNIST: 28x28 = 784 维输入
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练
    for _ in range(epochs):
        train_error, train_loss = epoch(train_loader, model, opt)
    
    # 测试
    test_error, test_loss = epoch(test_loader, model, None)
    
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
