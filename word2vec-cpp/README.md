# Word2Vec

Word2Vec 复现，支持 CBOW 和 Skip-gram 模型。

## 特性

- ✅ **双模型支持**：CBOW（连续词袋）和 Skip-gram
- ✅ **双训练方法**：Hierarchical Softmax 和 Negative Sampling
- ✅ **多线程训练**：充分利用多核 CPU，速度约 200万词/秒
- ✅ **动态学习率**：训练过程中自动调整 alpha
- ✅ **实用工具**：词语相似度查询、词语类比推理

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake ..
make -j
```

### 训练模型

```bash
./word2vec -t ../data/text8 -o ../data/text8-vectors.txt \
    -c 1 \              # CBOW 模型 (0=Skip-gram, 1=CBOW)
    -s 100 \            # 向量维度
    -w 5 \              # 窗口大小
    -n 5 \              # Negative sampling 数量 (0=Hierarchical Softmax)
    -i 5 \              # 训练迭代次数
    -p 12               # 线程数
```

### 词语相似度查询

```bash
./distance ../data/text8-vectors.txt
```

```
Enter word: king
                                              Word       Cosine distance
------------------------------------------------------------------------
                                             kings              0.689
                                            prince              0.641
                                             queen              0.569
                                           monarch              0.616
```

### 词语类比推理

```bash
./analogy ../data/text8-vectors.txt
```

```
Enter three words: woman man king
                                              Word       Cosine distance
------------------------------------------------------------------------
                                             queen              0.602
                                             kings              0.561
                                           monarch              0.561
```

**计算公式：** `vec(woman) - vec(man) + vec(king) ≈ vec(queen)`

## 核心算法

### 1. CBOW (Continuous Bag-of-Words)

通过上下文词预测中心词：

```
上下文: [the, cat, on, mat] → 预测: sat
```

- 输入层：上下文词向量的平均
- 输出层：Hierarchical Softmax 或 Negative Sampling

### 2. Skip-gram

通过中心词预测上下文：

```
输入: sat → 预测: [the, cat, on, mat]
```

### 3. 训练优化

- **Hierarchical Softmax**：使用 Huffman 树，复杂度 O(log V)
- **Negative Sampling**：随机采样负样本，复杂度 O(k)，k 通常为 5-10
- **Subsampling**：高频词降采样，提升训练效率

### 4. 多线程策略

- 文件分区：每个线程处理 `file_size / num_threads` 的数据块
- 无锁训练：各线程独立更新参数（异步 SGD）
- 动态调度：根据实际进度调整学习率

## 数学原理

### 余弦相似度

$$\text{similarity}(A, B) = \frac{A \cdot B}{||A|| \times ||B||} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum a_i^2} \times \sqrt{\sum b_i^2}}$$

### 词语类比

$$\vec{v}_{result} = \vec{v}_{word_3} - \vec{v}_{word_2} + \vec{v}_{word_1}$$

例如：`woman - man + king ≈ queen`

## 项目结构

```
word2vec-cpp/
├── include/
│   ├── config.hpp          # 配置参数
│   ├── vocabulary.hpp      # 词汇表管理
│   ├── model.hpp           # Word2Vec 模型
│   └── trainer.hpp         # 训练器
├── src/
│   ├── vocabulary.cpp      # 词频统计、Huffman 树
│   ├── model.cpp           # CBOW/Skip-gram 实现
│   ├── trainer.cpp         # 多线程训练逻辑
│   ├── word2vec.cpp        # 主程序
│   ├── distance.cpp        # 相似度工具
│   └── analogy.cpp         # 类比工具
├── data/
│   ├── text8               # 训练数据（96MB）
│   ├── questions-words.txt # 类比测试集
│   └── questions-phrases.txt
└── build/                  # 编译输出
```

## 命令行参数

```
-t <file>      训练文件路径
-o <file>      输出向量文件路径
-s <int>       词向量维度 (默认: 100)
-c <int>       模型类型 (0=Skip-gram, 1=CBOW, 默认: 1)
-w <int>       上下文窗口大小 (默认: 5)
-n <int>       负采样数量 (>0 使用 NS, =0 使用 HS, 默认: 5)
-i <int>       训练迭代次数 (默认: 5)
-p <int>       线程数 (默认: 12)
-m <int>       最小词频阈值 (默认: 5)
-a <float>     初始学习率 (默认: 0.025)
-e <float>     Subsampling 阈值 (默认: 1e-3)
```

## License

MIT License

---
