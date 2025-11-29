# GloVe-cpp

GloVe 的 C++ 实现，用于学习和复现斯坦福 NLP 的 [GloVe](https://nlp.stanford.edu/projects/glove/) 算法。

## 项目结构

```
Glove-cpp/
├── CMakeLists.txt
├── README.md
├── demo.sh
└── src/
    ├── common.h        # 公共定义
    ├── vocab_count.cpp # 词汇统计
    ├── cooccur.cpp     # 共现矩阵
    ├── shuffle.cpp     # 数据打乱
    └── glove.cpp       # GloVe 训练
```

和原始 GloVe 保持一致的结构！

## 编译

```bash
mkdir build && cd build
cmake .. && make -j4
```

## 使用流程

```bash
# 1. 词汇统计
./bin/vocab_count -c 5 < corpus.txt > vocab.txt

# 2. 构建共现矩阵
./bin/cooccur -V vocab.txt -w 15 < corpus.txt > cooccurrence.bin

# 3. 打乱数据
./bin/shuffle < cooccurrence.bin > cooccurrence.shuf.bin

# 4. 训练
./bin/glove -i cooccurrence.shuf.bin -V vocab.txt -o vectors -d 50 -n 15
```

## GloVe 核心公式

损失函数：
$$J = \sum_{i,j} f(X_{ij}) (w_i^T w_j + b_i + b_j - \log X_{ij})^2$$

加权函数：
$$f(x) = (x/x_{max})^\alpha \text{ if } x < x_{max}, \text{ else } 1$$

## 参考

- [GloVe 论文](https://nlp.stanford.edu/pubs/glove.pdf)
- [原始代码](https://github.com/stanfordnlp/GloVe)
