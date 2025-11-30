# GloVe-cpp

C++ 版 GloVe 复现，严格对齐 Stanford 原始实现（`vocab_count / cooccur / shuffle / glove` 四阶段流水线）

## 📂 项目结构

```
GloVe-cpp/
├── CMakeLists.txt          # 构建配置
├── README.md               # 文档（当前文件）
├── pyproject.toml          # uv 虚拟环境 + Python 依赖
├── demo.sh                 # 全流程演示脚本
├── eval/                   # 评估脚本和问题数据（复制自原始工程）
└── src/
    ├── common.h            # 通用类型与工具函数
    ├── vocab_count.cpp     # 构建词表 + 过滤低频
    ├── cooccur.cpp         # 共现矩阵（外排序 + 1-based 索引 + 对称窗口）
    ├── shuffle.cpp         # 共现数据打散（分块 Fisher-Yates + 归并）
    └── glove.cpp           # 训练（梯度剪裁 + AdaGrad + 多线程）
```

## 🔨 编译

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

生成的可执行文件位于：`build/bin/`。

## 🚀 快速使用（最小示例）

假设语料文件为 `text8`：

```bash
# 1. 构建词表（最小词频 5）
build/bin/vocab_count -c 5 < text8 > vocab.txt

# 2. 构建共现矩阵（窗口 15，内存 4GB）
build/bin/cooccur -V vocab.txt -w 15 -m 4.0 < text8 > cooccurrence.bin

# 3. 打乱
build/bin/shuffle -m 4.0 < cooccurrence.bin > cooccurrence.shuf.bin

# 4. 训练（输出 vectors.txt）
build/bin/glove --input-file cooccurrence.shuf.bin --vocab-file vocab.txt \
    --save-file vectors --vector-size 50 --iter 15 --threads 8 --x-max 10 --verbose 2

# 5. 评估（需 uv 已同步环境）
uv run python eval/python/evaluate.py --vocab_file vocab.txt --vectors_file vectors.txt
```

## 🧪 demo.sh 全流程脚本

```bash
./demo.sh
```

该脚本包含：构建 → 下载 `text8` → 训练 → 评估。支持通过环境变量覆盖默认参数，例如：

```bash
VECTOR_SIZE=100 MAX_ITER=10 WINDOW_SIZE=10 ./demo.sh
```

## ⚙️ 参数说明（核心程序）

`vocab_count`:
- `-c / --min-count`：过滤低频词
- `-v / --verbose`：日志级别

`cooccur`:
- `-V / --vocab-file`：词表文件
- `-w / --window-size`：左右窗口大小（对称）
- `-m / --memory`：排序缓冲区内存（GB）
- `-v / --verbose`

`shuffle`:
- `-m / --memory`：分块大小（GB）
- `-v / --verbose`

`glove`:
- `--input-file`：打乱后的二进制共现文件
- `--vocab-file`：词表
- `--save-file`：输出前缀（文本加 `.txt`）
- `--vector-size`：向量维度
- `--iter`：迭代次数
- `--threads`：线程数
- `--x-max`：权重函数截断阈值
- `--alpha`：权重函数指数（默认 0.75）
- `--model`：输出模式（0=全部参数，1=词向量，2=词+上下文向量和）
- `--grad-clip`：梯度剪裁阈值（默认 100.0）
- `--binary`：是否同时保存二进制（0/1/2）

## 🔍 数学公式

损失函数：
$$J = \sum_{i,j} f(X_{ij}) \big(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2$$

权重函数：
$$f(x) = \begin{cases}(x/x_{\max})^{\alpha}, & x < x_{\max} \\ 1, & x \ge x_{\max}\end{cases}$$

自适应更新（AdaGrad）：
$$\theta_{t} = \theta_{t-1} - \eta \frac{g_t}{\sqrt{G_t}} \quad ; \quad G_t = G_{t-1} + g_t^2$$

## 📊 评估指标（text8 示例）

| 类型       | 准确率 (Top1) | 说明 |
|------------|---------------|------|
| 语义 (Semantic) | ≈ 27–29% | 国家、首都、家族关系 |
| 句法 (Syntactic) | ≈ 19–21% | 词形转换、比较级等 |
| 总体 (Overall)   | ≈ 23%    | 受训练轮数 / 维度影响 |

（与原版 GloVe float 级误差内一致）

## 🧪 创建/使用 Python 评估环境

已提供 `pyproject.toml`：
```bash
uv sync            # 创建虚拟环境并安装依赖
uv run python eval/python/evaluate.py --vocab_file vocab.txt --vectors_file vectors.txt
```

## 📚 参考资料

- 原始项目: https://github.com/stanfordnlp/GloVe
- 论文: Pennington et al. 2014, GloVe: Global Vectors for Word Representation
- 语料: http://mattmahoney.net/dc/text8.zip
