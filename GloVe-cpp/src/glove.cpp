/**
 * @file glove.cpp
 * @brief GloVe 模型训练主程序
 * 
 * 该程序读取打乱后的共现矩阵文件 (cooccurrence.shuf.bin)，
 * 使用 AdaGrad 算法训练词向量。
 * 
 * 核心目标函数:
 * J = Sum f(X_ij) * (w_i^T w_j + b_i + b_j - log(X_ij))^2
 * 
 * 其中:
 * - w_i, w_j 是词向量 (word vectors)
 * - b_i, b_j 是偏置项 (biases)
 * - X_ij 是共现次数
 * - f(x) 是权重函数: (x/x_max)^alpha if x < x_max else 1
 * 
 * 对应原始 GloVe: src/glove.c
 */

#include "common.h"
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <random>
#include <getopt.h>

// ============================================================================
// 全局变量与参数
// ============================================================================

int verbose = 2;                // 详细程度: 0, 1, 2
int use_binary = 0;             // 输出模型是否为二进制格式: 0 (text), 1 (binary), 2 (both)
int model = 2;                  // 模型输出类型: 0=all, 1=word only, 2=word+context
int num_threads = 8;            // 线程数
real x_max = 10.0;              // 权重函数的截断阈值
real alpha = 0.75;              // 权重函数的指数
real eta = 0.05;                // 初始学习率
real grad_clip_value = 100.0;   // 梯度剪裁阈值
int iter = 15;                  // 迭代次数
int vector_size = 50;           // 词向量维度
int save_gradsq = 0;            // 是否保存梯度平方累积量 (用于断点续训)
int checkpoint_every = 0;       // 每隔多少次迭代保存一次检查点 (0表示不保存)

std::string vocab_file;         // 词汇表文件路径
std::string input_file;         // 输入文件路径 (cooccurrence.shuf.bin)
std::string save_W_file;        // 输出模型文件路径
std::string save_gradsq_file;   // 输出梯度文件路径

long long vocab_size = 0;       // 词汇表大小
long long num_lines = 0;        // 共现记录总数 (行数)

real *W = nullptr;              // 词向量数组 (包含词向量、上下文向量和偏置)
real *gradsq = nullptr;         // 梯度平方累积量数组 (AdaGrad)
real *cost = nullptr;           // 每个线程的损失累积

// ============================================================================
// 核心函数实现
// ============================================================================

/**
 * @brief 初始化参数
 * 
 * 分配内存并初始化 W 和 gradsq。
 * W 初始化为 [-0.5/vector_size, 0.5/vector_size] 之间的均匀分布。
 * gradsq 初始化为 1.0 (为了避免除以零，且作为 AdaGrad 的初始平滑)。
 */
void initialize_parameters() {
    long long w_size = 2 * vocab_size * (vector_size + 1); // 2 sets of vectors + biases
    
    // 使用 posix_memalign 进行内存对齐分配 (128字节对齐)，有利于 SIMD 优化
    if (posix_memalign((void **)&W, 128, w_size * sizeof(real))) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    if (posix_memalign((void **)&gradsq, 128, w_size * sizeof(real))) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        free(W);
        exit(1);
    }

    // 随机数生成器
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<real> dist(-0.5 / vector_size, 0.5 / vector_size);

    for (long long i = 0; i < w_size; i++) {
        W[i] = dist(gen);
        gradsq[i] = 1.0; // 初始梯度平方设为 1.0
    }
}

/**
 * @brief 检查数值是否为 NaN，若是则返回 0，否则返回原值
 * 
 * 用于安全更新参数：如果梯度是 NaN，则跳过该更新
 */
inline real check_nan(real update) {
    return std::isnan(update) || std::isinf(update) ? 0.0 : update;
}

/**
 * @brief 训练线程函数
 * 
 * 每个线程处理输入文件的一部分。
 * 读取 CREC 记录，计算梯度，并使用 AdaGrad 更新参数。
 * 
 * 核心更新逻辑 (严格对照原始 glove.c):
 * 1. 计算预测值与目标值的差异: diff = w_i·w_j + b_i + b_j - log(X_ij)
 * 2. 应用权重函数: fdiff = f(X_ij) * diff
 * 3. 对向量分量: temp = clip(fdiff * w_other) * eta, 然后 w -= temp / sqrt(gradsq)
 * 4. 对偏置项: b -= fdiff / sqrt(gradsq)  (注意：偏置更新不乘 eta!)
 * 5. 累积梯度平方: gradsq += temp^2 (向量) 或 fdiff^2 (偏置)
 * 
 * @param id 线程 ID
 */
void glove_thread(int id) {
    long long lines_per_thread = num_lines / num_threads;
    long long start_line = id * lines_per_thread;
    // 最后一个线程处理剩余所有行
    long long my_lines = (id == num_threads - 1) ? (num_lines - start_line) : lines_per_thread;

    FILE *fin = fopen(input_file.c_str(), "rb");
    if (fin == NULL) {
        fprintf(stderr, "Unable to open input file %s in thread %d\n", input_file.c_str(), id);
        return;
    }

    // 定位到该线程的起始位置
    fseeko(fin, start_line * sizeof(CREC), SEEK_SET);

    CREC cr;
    real diff, fdiff, temp1, temp2;
    long long l1, l2;
    
    cost[id] = 0; // 重置该线程的损失

    // 临时数组保存更新量 (原版也是这样做的，用于检查 NaN 后再统一应用)
    std::vector<real> W_updates1(vector_size);
    std::vector<real> W_updates2(vector_size);

    for (long long a = 0; a < my_lines; a++) {
        if (fread(&cr, sizeof(CREC), 1, fin) != 1) break;
        if (feof(fin)) break;
        if (cr.word1 < 1 || cr.word2 < 1) continue;

        // ========================================
        // 计算索引 (W 数组布局)
        // ========================================
        // 每个词占 (vector_size + 1) 个 real: [v_0, v_1, ..., v_{d-1}, bias]
        // 前半部分 [0, vocab_size) 是中心词 (word vectors)
        // 后半部分 [vocab_size, 2*vocab_size) 是上下文词 (context vectors)
        l1 = (cr.word1 - 1LL) * (vector_size + 1);
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1);

        // ========================================
        // 计算损失与梯度因子
        // ========================================
        // diff = w_i · w_j (内积)
        diff = 0;
        for (int b = 0; b < vector_size; b++) {
            diff += W[b + l1] * W[b + l2];
        }
        // 加上偏置: + b_i + b_j - log(X_ij)
        diff += W[vector_size + l1] + W[vector_size + l2] - log(cr.val);

        // 加权因子: fdiff = f(X_ij) * diff
        // f(x) = (x/x_max)^alpha if x < x_max, else 1
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff;

        // 检查 NaN/Inf
        if (std::isnan(diff) || std::isnan(fdiff) || std::isinf(diff) || std::isinf(fdiff)) {
            fprintf(stderr, "Caught NaN in diff for thread %d. Skipping update\n", id);
            continue;
        }

        // 累积损失: J += 0.5 * f(X_ij) * diff^2 = 0.5 * fdiff * diff
        cost[id] += 0.5 * fdiff * diff;

        // ========================================
        // AdaGrad 更新 - 向量分量
        // ========================================
        // 原版逻辑:
        //   temp1 = clip(fdiff * W[b + l2]) * eta   // 梯度 * 学习率
        //   W_updates1[b] = temp1 / sqrt(gradsq)   // AdaGrad 缩放
        //   gradsq += temp1^2                       // 累积 (注意是累积 temp1^2，含 eta)
        //   最后统一 W -= W_updates
        real W_updates1_sum = 0;
        real W_updates2_sum = 0;
        
        for (int b = 0; b < vector_size; b++) {
            // 梯度剪裁 + 乘学习率
            temp1 = std::fmin(std::fmax(fdiff * W[b + l2], -grad_clip_value), grad_clip_value) * eta;
            temp2 = std::fmin(std::fmax(fdiff * W[b + l1], -grad_clip_value), grad_clip_value) * eta;
            
            // AdaGrad 更新量
            W_updates1[b] = temp1 / sqrt(gradsq[b + l1]);
            W_updates2[b] = temp2 / sqrt(gradsq[b + l2]);
            W_updates1_sum += W_updates1[b];
            W_updates2_sum += W_updates2[b];
            
            // 累积梯度平方 (含 eta)
            gradsq[b + l1] += temp1 * temp1;
            gradsq[b + l2] += temp2 * temp2;
        }
        
        // 检查更新是否有效，然后应用
        if (!std::isnan(W_updates1_sum) && !std::isinf(W_updates1_sum) &&
            !std::isnan(W_updates2_sum) && !std::isinf(W_updates2_sum)) {
            for (int b = 0; b < vector_size; b++) {
                W[b + l1] -= W_updates1[b];
                W[b + l2] -= W_updates2[b];
            }
        }

        // ========================================
        // AdaGrad 更新 - 偏置项
        // ========================================
        // 原版逻辑 (注意：偏置更新不乘 eta!):
        //   W[bias] -= check_nan(fdiff / sqrt(gradsq[bias]))
        //   fdiff *= fdiff
        //   gradsq[bias] += fdiff
        W[vector_size + l1] -= check_nan(fdiff / sqrt(gradsq[vector_size + l1]));
        W[vector_size + l2] -= check_nan(fdiff / sqrt(gradsq[vector_size + l2]));
        fdiff *= fdiff;  // fdiff^2 用于累积
        gradsq[vector_size + l1] += fdiff;
        gradsq[vector_size + l2] += fdiff;
    }

    fclose(fin);
}

/**
 * @brief 保存模型参数
 * 
 * model 参数决定输出格式:
 * - model == 0: 输出所有参数 (word vectors + biases + context vectors + biases)
 * - model == 1: 仅输出 word vectors (不含 bias)
 * - model == 2: 输出 word + context vectors 的和 (推荐，效果最好)
 * 
 * @param nb_iter 当前迭代次数 (用于 checkpoint 文件命名)
 * @return 0: 成功, 1: 失败
 */
int save_params(int nb_iter) {
    (void)nb_iter;
    
    // 构建输出文件名 (加 .txt 后缀)
    std::string outfile = save_W_file + ".txt";
    
    FILE *fout = fopen(outfile.c_str(), "wb");
    if (fout == NULL) {
        fprintf(stderr, "Unable to open save file %s\n", outfile.c_str());
        return 1;
    }

    // 打开词汇表文件
    FILE *fvocab = fopen(vocab_file.c_str(), "r");
    if (fvocab == NULL) {
        fprintf(stderr, "Unable to open vocab file %s\n", vocab_file.c_str());
        fclose(fout);
        return 1;
    }

    char word[MAX_STRING_LENGTH + 1];
    char format[20];
    sprintf(format, "%%%ds", MAX_STRING_LENGTH);

    for (long long a = 0; a < vocab_size; a++) {
        // 读取词
        if (fscanf(fvocab, format, word) == 0) {
            fclose(fvocab);
            fclose(fout);
            return 1;
        }
        
        // 写入词
        fprintf(fout, "%s", word);

        long long l1 = a * (vector_size + 1);
        long long l2 = (a + vocab_size) * (vector_size + 1);

        if (model == 0) {
            // 输出所有参数 (word vector + bias + context vector + bias)
            for (int b = 0; b < vector_size + 1; b++) {
                fprintf(fout, " %lf", W[l1 + b]);
            }
            for (int b = 0; b < vector_size + 1; b++) {
                fprintf(fout, " %lf", W[l2 + b]);
            }
        } else if (model == 1) {
            // 仅输出 word vector (不含 bias)
            for (int b = 0; b < vector_size; b++) {
                fprintf(fout, " %lf", W[l1 + b]);
            }
        } else if (model == 2) {
            // 输出 word + context vectors 的和 (推荐)
            for (int b = 0; b < vector_size; b++) {
                fprintf(fout, " %lf", W[l1 + b] + W[l2 + b]);
            }
        }
        
        fprintf(fout, "\n");
        
        // 跳过词汇表中的频率字段
        if (fscanf(fvocab, format, word) == 0) {
            // 文件可能结束了，忽略
        }
    }
    
    fclose(fvocab);
    fclose(fout);
    return 0;
}

/**
 * @brief 训练主循环
 */
void train_glove() {
    // 1. 统计输入文件行数
    FILE *fin = fopen(input_file.c_str(), "rb");
    if (fin == NULL) {
        fprintf(stderr, "Unable to open input file %s\n", input_file.c_str());
        exit(1);
    }
    fseeko(fin, 0, SEEK_END);
    num_lines = ftello(fin) / sizeof(CREC);
    fseeko(fin, 0, SEEK_SET);
    fclose(fin);
    
    if (verbose > 0) fprintf(stderr, "Read %lld lines.\n", num_lines);

    // 2. 初始化参数
    if (verbose > 1) fprintf(stderr, "Initializing parameters...\n");
    initialize_parameters();

    // 3. 迭代训练
    if (verbose > 0) fprintf(stderr, "vector_size: %d\n", vector_size);
    if (verbose > 0) fprintf(stderr, "vocab_size: %lld\n", vocab_size);
    if (verbose > 0) fprintf(stderr, "x_max: %lf\n", x_max);
    if (verbose > 0) fprintf(stderr, "alpha: %lf\n", alpha);
    
    // 分配 cost 数组
    cost = (real *)malloc(num_threads * sizeof(real));

    for (int i = 0; i < iter; i++) {
        std::vector<std::thread> threads;
        
        // 启动线程
        for (int j = 0; j < num_threads; j++) {
            threads.emplace_back(glove_thread, j);
        }
        
        // 等待线程结束
        for (auto &t : threads) {
            t.join();
        }
        
        // 计算总损失
        real total_cost = 0;
        for (int j = 0; j < num_threads; j++) {
            total_cost += cost[j];
        }
        total_cost /= num_lines; // 平均损失
        
        if (verbose > 1) fprintf(stderr, "iter: %03d, cost: %lf\n", i + 1, total_cost);
    }
    
    free(cost);
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char **argv) {
    int c;
    
    // 默认参数
    vocab_file = "vocab.txt";
    input_file = "cooccurrence.shuf.bin";
    save_W_file = "vectors.txt";
    save_gradsq_file = "gradsq";
    
    while (1) {
        static struct option long_options[] = {
            {"verbose", required_argument, 0, 'v'},
            {"vector-size", required_argument, 0, 's'},
            {"iter", required_argument, 0, 'i'},
            {"threads", required_argument, 0, 't'},
            {"eta", required_argument, 0, 'e'},
            {"alpha", required_argument, 0, 'a'},
            {"x-max", required_argument, 0, 'x'},
            {"binary", required_argument, 0, 'b'},
            {"model", required_argument, 0, 'm'},
            {"grad-clip", required_argument, 0, 'C'},
            {"vocab-file", required_argument, 0, 'V'},
            {"input-file", required_argument, 0, 'I'},
            {"save-file", required_argument, 0, 'S'},
            {"gradsq-file", required_argument, 0, 'G'},
            {"checkpoint-every", required_argument, 0, 'c'},
            {0, 0, 0, 0}
        };
        
        int option_index = 0;
        c = getopt_long(argc, argv, "v:s:i:t:e:a:x:b:m:C:V:I:S:G:c:", long_options, &option_index);
        
        if (c == -1) break;
        
        switch (c) {
            case 'v': verbose = atoi(optarg); break;
            case 's': vector_size = atoi(optarg); break;
            case 'i': iter = atoi(optarg); break;
            case 't': num_threads = atoi(optarg); break;
            case 'e': eta = atof(optarg); break;
            case 'a': alpha = atof(optarg); break;
            case 'x': x_max = atof(optarg); break;
            case 'b': use_binary = atoi(optarg); break;
            case 'm': model = atoi(optarg); break;
            case 'C': grad_clip_value = atof(optarg); break;
            case 'V': vocab_file = optarg; break;
            case 'I': input_file = optarg; break;
            case 'S': save_W_file = optarg; break;
            case 'G': save_gradsq_file = optarg; break;
            case 'c': checkpoint_every = atoi(optarg); break;
            default: fprintf(stderr, "Unknown option\n"); return 1;
        }
    }

    // 1. 获取词汇表大小
    FILE *fvocab = fopen(vocab_file.c_str(), "r");
    if (fvocab == NULL) {
        fprintf(stderr, "Unable to open vocab file %s\n", vocab_file.c_str());
        return 1;
    }
    // 简单的行数统计
    long long lines = 0;
    int ch;
    while(!feof(fvocab)) {
        ch = fgetc(fvocab);
        if(ch == '\n') {
            lines++;
        }
    }
    // 如果最后一行没有换行符，也算一行 (通常 vocab.txt 都有换行)
    // 重新检查一下，或者使用更稳健的方法
    // 原始 GloVe 是在 main 中循环 fgets
    vocab_size = lines;
    fclose(fvocab);
    
    if (verbose > 0) fprintf(stderr, "Building GloVe model with vocab size: %lld\n", vocab_size);

    // 2. 训练
    train_glove();

    // 3. 保存结果
    if (verbose > 0) fprintf(stderr, "Saving model to %s\n", save_W_file.c_str());
    save_params(iter);

    // 清理内存
    free(W);
    free(gradsq);

    return 0;
}
