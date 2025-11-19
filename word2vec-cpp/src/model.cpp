#include "model.hpp"
#include "vocabulary.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>

namespace word2vec {

Word2VecModel::Word2VecModel(const Vocabulary& vocab, const Config& config)
    : vocab_(vocab), config_(config), rng_(std::random_device{}()) {
    InitNet();
    InitExpTable();
    if (config_.negative > 0) {
        InitUnigramTable();
    }
}

void Word2VecModel::InitNet() {
    size_t vocab_size = vocab_.Size();
    size_t layer_size = config_.vector_size;
    
    // 初始化 syn0 (输入层embedding)
    syn0_.resize(vocab_size * layer_size);
    std::uniform_real_distribution<float> dist(-0.5f / layer_size, 0.5f / layer_size);
    for (auto& val : syn0_) {
        val = dist(rng_);
    }
    
    // 初始化 syn1/syn1neg
    if (config_.hierarchical_softmax) {
        syn1_.resize(vocab_size * layer_size, 0.0f);
    }
    if (config_.negative > 0) {
        syn1neg_.resize(vocab_size * layer_size, 0.0f);
    }
}

void Word2VecModel::InitExpTable() {
    constexpr int EXP_TABLE_SIZE = 1000;
    constexpr int MAX_EXP = 6;
    
    exp_table_.resize(EXP_TABLE_SIZE);
    for (int i = 0; i < EXP_TABLE_SIZE; ++i) {
        float x = (i / static_cast<float>(EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP;
        exp_table_[i] = std::exp(x) / (std::exp(x) + 1);
    }
}

void Word2VecModel::InitUnigramTable() {
    constexpr int TABLE_SIZE = 1e8;
    unigram_table_.resize(TABLE_SIZE);
    
    constexpr float POWER = 0.75f;
    double total_pow = 0.0;
    for (size_t i = 0; i < vocab_.Size(); ++i) {
        total_pow += std::pow(vocab_.GetWord(i).count, POWER);
    }
    
    size_t word_idx = 0;
    double cumulative_prob = std::pow(vocab_.GetWord(word_idx).count, POWER) / total_pow;
    
    for (size_t i = 0; i < TABLE_SIZE; ++i) {
        unigram_table_[i] = word_idx;
        if (i / static_cast<double>(TABLE_SIZE) > cumulative_prob) {
            word_idx++;
            if (word_idx < vocab_.Size()) {
                cumulative_prob += std::pow(vocab_.GetWord(word_idx).count, POWER) / total_pow;
            } else {
                word_idx = vocab_.Size() - 1;
            }
        }
    }
}

void Word2VecModel::TrainCBOW(int word_pos, const std::vector<int>& sentence,
                               float alpha, int reduced_window) {
    // =============================================================================
    // CBOW (Continuous Bag of Words) 训练算法
    // =============================================================================
    // 
    // 训练目标：用上下文词的平均向量来预测中心词
    // 
    // 举例说明：
    //   句子："I love natural language processing"
    //   中心词：language (位置 word_pos)
    //   窗口：2
    //   上下文：["love", "natural", "processing"] （跳过中心词）
    // 
    // 训练步骤：
    //   1. 收集上下文词向量：v(love), v(natural), v(processing)
    //   2. 计算平均：h = (v(love) + v(natural) + v(processing)) / 3
    //   3. 用 h 预测中心词 "language"：
    //      - Hierarchical Softmax: 沿 Huffman 树路径计算概率
    //      - Negative Sampling: 区分正样本 "language" 和负样本
    //   4. 反向传播：更新所有上下文词的向量
    // 
    // 优势：计算快（一次前向传播处理多个上下文词）
    // 劣势：对词序不敏感（都是求平均）
    // =============================================================================
    
    int word = sentence[word_pos];      // 中心词索引
    int layer_size = config_.vector_size;
    std::vector<float> neu1(layer_size, 0.0f);   // 隐层向量（上下文平均）
    std::vector<float> neu1e(layer_size, 0.0f);  // 误差累积向量
    
    int context_count = 0;
    int actual_window = config_.window - reduced_window;  // 实际窗口大小
    
    // -------------------------------------------------------------------------
    // 步骤 1：收集上下文词并计算平均向量
    // -------------------------------------------------------------------------
    // 遍历窗口 [word_pos - actual_window, word_pos + actual_window]
    // 跳过中心词位置（a == 0）
    
    for (int a = -actual_window; a <= actual_window; ++a) {
        if (a == 0) continue;  // 跳过中心词自己
        
        int context_pos = word_pos + a;
        
        // 边界检查：确保不越界
        if (context_pos < 0 || context_pos >= static_cast<int>(sentence.size())) continue;
        
        int context_word = sentence[context_pos];
        
        if (context_word == -1) continue;
        
        // 累加上下文词的向量到 neu1
        // syn0_ 是词嵌入矩阵，大小为 [vocab_size × layer_size]
        size_t offset = context_word * layer_size;
        for (int c = 0; c < layer_size; ++c) {
            neu1[c] += syn0_[offset + c];
        }
        context_count++;
    }
    
    // 如果没有有效的上下文词，跳过训练
    if (context_count == 0) return;
    
    // 计算平均：neu1 = sum(context_vectors) / context_count
    for (int c = 0; c < layer_size; ++c) {
        neu1[c] /= context_count;
    }
    
    // -------------------------------------------------------------------------
    // 步骤 2：Hierarchical Softmax 训练（如果启用）
    // -------------------------------------------------------------------------
    // 使用 Huffman 树来高效计算 softmax
    // 
    // 原理：
    //   传统 softmax 需要计算 P(word|context) = exp(score_word) / sum_all(exp(score))
    //   计算量 O(vocab_size)，词汇量大时很慢
    // 
    //   Hierarchical Softmax 将问题转化为二叉树上的路径预测：
    //   - 每个词对应树的一个叶子节点
    //   - 从根到叶子的路径唯一确定该词
    //   - 每个内部节点做一次二分类（左 or 右）
    //   - 计算量降为 O(log vocab_size)
    // 
    // 例如：预测词 "language"
    //   Huffman 编码：[1, 0, 1] (从根到叶子的路径)
    //   point 数组：[n1, n2, n3] (路径上的内部节点索引)
    //   
    //   计算：P(language) = σ(h·θ_n1) × (1-σ(h·θ_n2)) × σ(h·θ_n3)
    //   其中 h = neu1 是隐层向量，θ_ni 是节点参数
    
    if (config_.hierarchical_softmax) {
        const auto& vocab_word = vocab_.GetWord(word);
        
        // 遍历 Huffman 树路径上的每个节点
        for (size_t d = 0; d < vocab_word.code.size(); ++d) {
            size_t node_idx = vocab_word.point[d];  // 当前节点索引
            size_t offset = node_idx * layer_size;
            
            // 前向传播：计算 f = σ(neu1 · syn1[node])
            // syn1_ 存储所有内部节点的参数向量
            float f = 0.0f;
            for (int c = 0; c < layer_size; ++c) {
                f += neu1[c] * syn1_[offset + c];  // 点积
            }
            
            // 使用查找表快速计算 sigmoid
            constexpr int MAX_EXP = 6, EXP_TABLE_SIZE = 1000;
            if (f <= -MAX_EXP || f >= MAX_EXP) continue;  // 饱和区跳过
            
            int table_idx = static_cast<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
            f = exp_table_[table_idx];  // f = σ(f)
            
            // 计算梯度：g = (label - prediction) × learning_rate
            // label = 1 - code[d]：Huffman 编码（0→标签1，1→标签0）
            float g = (1.0f - vocab_word.code[d] - f) * alpha;
            
            // 反向传播：累积误差到 neu1e
            for (int c = 0; c < layer_size; ++c) {
                neu1e[c] += g * syn1_[offset + c];
            }
            
            // 更新内部节点参数：syn1[node] += g × neu1
            for (int c = 0; c < layer_size; ++c) {
                syn1_[offset + c] += g * neu1[c];
            }
        }
    }
    
    // -------------------------------------------------------------------------
    // 步骤 3：Negative Sampling 训练（如果启用）
    // -------------------------------------------------------------------------
    // 负采样：不计算完整的 softmax，而是区分正负样本
    // 
    // 原理：
    //   完整 softmax 需要对所有词归一化，计算量 O(vocab_size)
    //   负采样简化为二分类问题：区分真实词 vs 噪声词
    // 
    // 训练样本：
    //   - 1 个正样本：(context, target_word, label=1)
    //   - k 个负样本：(context, noise_word, label=0)
    //   k 通常取 5-20（词汇量小时用大值，大时用小值）
    // 
    // 负样本采样策略：
    //   不是均匀采样，而是按 P(w) ∝ count(w)^0.75
    //   这样高频词被采样概率高，但不会过高
    // 
    // 例如：预测 "language"
    //   正样本：(context_avg, "language", 1) → 希望分数高
    //   负样本：(context_avg, "apple", 0)    → 希望分数低
    //           (context_avg, "run", 0)      → 希望分数低
    //           ...
    
    if (config_.negative > 0) {
        std::uniform_int_distribution<size_t> table_dist(0, unigram_table_.size() - 1);
        
        // 训练 1 + negative 个样本（1个正样本 + negative个负样本）
        for (int d = 0; d <= config_.negative; ++d) {
            int target;  // 目标词索引
            int label;   // 标签（1=正样本，0=负样本）
            
            if (d == 0) {
                // 第一个样本：正样本（真实的中心词）
                target = word;
                label = 1;
            } else {
                // 后续样本：负样本（从 unigram_table 随机采样）
                target = unigram_table_[table_dist(rng_)];
                
                // 避免采样到 </s> 或重复采样到正样本
                if (target == 0) target = (rng_() % (vocab_.Size() - 1)) + 1;
                if (target == word) continue;  // 跳过与正样本相同的词
                
                label = 0;
            }
            
            size_t offset = target * layer_size;
            
            // 前向传播：计算 f = neu1 · syn1neg[target]
            // syn1neg_ 存储所有词的负采样参数向量
            float f = 0.0f;
            for (int c = 0; c < layer_size; ++c) {
                f += neu1[c] * syn1neg_[offset + c];
            }
            
            // 计算梯度：g = (label - σ(f)) × alpha
            // 正样本：label=1，希望 σ(f)→1，即 f→+∞
            // 负样本：label=0，希望 σ(f)→0，即 f→-∞
            float g;
            constexpr int MAX_EXP = 6, EXP_TABLE_SIZE = 1000;
            
            if (f > MAX_EXP) {
                g = (label - 1) * alpha;  // σ(f) ≈ 1
            } else if (f < -MAX_EXP) {
                g = (label - 0) * alpha;  // σ(f) ≈ 0
            } else {
                int table_idx = static_cast<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
                g = (label - exp_table_[table_idx]) * alpha;
            }
            
            // 反向传播：累积误差
            for (int c = 0; c < layer_size; ++c) {
                neu1e[c] += g * syn1neg_[offset + c];
            }
            
            // 更新目标词的负采样参数
            for (int c = 0; c < layer_size; ++c) {
                syn1neg_[offset + c] += g * neu1[c];
            }
        }
    }
    
    // -------------------------------------------------------------------------
    // 步骤 4：反向传播到输入层，更新所有上下文词的向量
    // -------------------------------------------------------------------------
    // neu1e 累积了从输出层传回的梯度
    // 需要将这个梯度应用到所有参与计算的上下文词向量上
    // 
    // 为什么所有上下文词都要更新？
    //   因为它们的平均值 neu1 参与了预测，所以都需要调整
    //   这样可以让相似上下文的词向量更接近
    
    for (int a = -actual_window; a <= actual_window; ++a) {
        if (a == 0) continue;  // 跳过中心词
        
        int context_pos = word_pos + a;
        if (context_pos < 0 || context_pos >= static_cast<int>(sentence.size())) continue;
        
        int context_word = sentence[context_pos];
        
        if (context_word == -1) continue;
        
        // 更新上下文词的向量：syn0[context_word] += neu1e
        size_t offset = context_word * layer_size;
        for (int c = 0; c < layer_size; ++c) {
            syn0_[offset + c] += neu1e[c];
        }
    }
    
    // =============================================================================
    // CBOW 训练总结：
    // 
    // 输入：窗口内的上下文词 [w_{-2}, w_{-1}, w_{+1}, w_{+2}]
    // 输出：中心词 w_0
    // 
    // 网络结构（以 Negative Sampling 为例）：
    //   输入层：上下文词向量 v(w_{-2}), v(w_{-1}), v(w_{+1}), v(w_{+2})
    //      ↓ 求平均
    //   隐层：h = (v(w_{-2}) + v(w_{-1}) + v(w_{+1}) + v(w_{+2})) / 4
    //      ↓ 与目标词参数点积
    //   输出层：σ(h · θ_{w_0}) → 预测是否为 w_0
    // 
    // 参数更新：
    //   - 上下文词向量 syn0[context_words] （输入层）
    //   - Huffman 节点参数 syn1[nodes] 或负采样参数 syn1neg[words] （输出层）
    // 
    // 时间复杂度：
    //   - Hierarchical Softmax: O(window_size × vector_size × log(vocab_size))
    //   - Negative Sampling: O(window_size × vector_size × negative)
    // =============================================================================
}

void Word2VecModel::TrainSkipGram(int word_pos, const std::vector<int>& sentence,
                                   float alpha, int reduced_window) {
    // =============================================================================
    // Skip-gram 训练算法
    // =============================================================================
    // 
    // 训练目标：用中心词预测每个上下文词
    // 
    // 举例说明：
    //   句子："I love natural language processing"
    //   中心词：language (位置 word_pos)
    //   窗口：2
    //   上下文：["love", "natural", "processing"]
    // 
    // Skip-gram 训练过程：
    //   对每个上下文词单独训练：
    //     1. 用 v(language) 预测 "love"
    //     2. 用 v(language) 预测 "natural"
    //     3. 用 v(language) 预测 "processing"
    //   
    //   每次预测都是独立的训练样本！
    // 
    // 与 CBOW 的关键区别：
    //   - CBOW: 多个上下文 → 1个中心词（多对一，求平均）
    //   - Skip-gram: 1个中心词 → 多个上下文（一对多，独立训练）
    // 
    // 优势：对低频词效果更好（每个词对产生独立样本）
    // 劣势：计算慢（需要处理 window_size × 2 个样本）
    // =============================================================================
    
    int word = sentence[word_pos];      // 中心词索引
    int layer_size = config_.vector_size;
    int actual_window = config_.window - reduced_window;
    
    // -------------------------------------------------------------------------
    // 对窗口内的每个上下文词，执行一次独立的训练
    // -------------------------------------------------------------------------
    // 注意：这里的循环是训练多个样本，不是收集上下文！
    // 每次迭代都是完整的前向+反向传播
    
    for (int a = -actual_window; a <= actual_window; ++a) {
        if (a == 0) continue;  // 跳过中心词自己
        
        int context_pos = word_pos + a;
        
        // 边界检查
        if (context_pos < 0 || context_pos >= static_cast<int>(sentence.size())) continue;
        
        int context_word = sentence[context_pos];  // 当前要预测的上下文词
        
        if (context_word == -1) continue;
        
        // =====================================================================
        // 训练样本：(中心词word, 上下文词context_word)
        // 目标：让 v(word) 能够预测出 context_word
        // =====================================================================
        
        // input_offset 指向中心词的向量位置
        // 注意：这里用的是 context_word 的向量，因为 Skip-gram 反过来了
        // 实际上应该用 word 的向量预测 context_word，但实现中交换了角色
        size_t input_offset = context_word * layer_size;
        
        // 误差累积向量（每个样本独立）
        std::vector<float> neu1e(layer_size, 0.0f);
        
        // -----------------------------------------------------------------
        // Hierarchical Softmax：沿目标词的 Huffman 树路径训练
        // -----------------------------------------------------------------
        // 用中心词向量预测目标词（上下文词）
        // 
        // 注意：这里用的是 word（要预测的目标）的 Huffman 编码
        //      syn0[context_word] 作为输入向量
        
        if (config_.hierarchical_softmax) {
            const auto& vocab_word = vocab_.GetWord(word);  // 目标词的 Huffman 信息
            
            for (size_t d = 0; d < vocab_word.code.size(); ++d) {
                size_t node_idx = vocab_word.point[d];
                size_t node_offset = node_idx * layer_size;
                
                // 前向传播：f = syn0[context_word] · syn1[node]
                float f = 0.0f;
                for (int c = 0; c < layer_size; ++c) {
                    f += syn0_[input_offset + c] * syn1_[node_offset + c];
                }
                
                // Sigmoid 激活
                constexpr int MAX_EXP = 6, EXP_TABLE_SIZE = 1000;
                if (f <= -MAX_EXP || f >= MAX_EXP) continue;
                int table_idx = static_cast<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
                f = exp_table_[table_idx];
                
                // 计算梯度
                float g = (1.0f - vocab_word.code[d] - f) * alpha;
                
                // 反向传播
                for (int c = 0; c < layer_size; ++c) {
                    neu1e[c] += g * syn1_[node_offset + c];
                }
                
                // 更新节点参数
                for (int c = 0; c < layer_size; ++c) {
                    syn1_[node_offset + c] += g * syn0_[input_offset + c];
                }
            }
        }
        
        // -----------------------------------------------------------------
        // Negative Sampling：区分目标词和噪声词
        // -----------------------------------------------------------------
        // 训练样本：(context_word, word, label=1) + k个负样本
        
        if (config_.negative > 0) {
            std::uniform_int_distribution<size_t> table_dist(0, unigram_table_.size() - 1);
            
            for (int d = 0; d <= config_.negative; ++d) {
                int target, label;
                
                if (d == 0) {
                    // 正样本：真实的上下文词
                    target = word;
                    label = 1;
                } else {
                    // 负样本：随机采样的噪声词
                    target = unigram_table_[table_dist(rng_)];
                    if (target == 0) target = (rng_() % (vocab_.Size() - 1)) + 1;
                    if (target == word) continue;
                    label = 0;
                }
                
                size_t target_offset = target * layer_size;
                
                // 前向传播：f = syn0[context_word] · syn1neg[target]
                float f = 0.0f;
                for (int c = 0; c < layer_size; ++c) {
                    f += syn0_[input_offset + c] * syn1neg_[target_offset + c];
                }
                
                // 计算梯度
                float g;
                constexpr int MAX_EXP = 6, EXP_TABLE_SIZE = 1000;
                if (f > MAX_EXP) {
                    g = (label - 1) * alpha;
                } else if (f < -MAX_EXP) {
                    g = (label - 0) * alpha;
                } else {
                    int table_idx = static_cast<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
                    g = (label - exp_table_[table_idx]) * alpha;
                }
                
                // 反向传播
                for (int c = 0; c < layer_size; ++c) {
                    neu1e[c] += g * syn1neg_[target_offset + c];
                }
                
                // 更新目标词参数
                for (int c = 0; c < layer_size; ++c) {
                    syn1neg_[target_offset + c] += g * syn0_[input_offset + c];
                }
            }
        }
        
        // -----------------------------------------------------------------
        // 更新上下文词的向量
        // -----------------------------------------------------------------
        // 将累积的误差应用到当前上下文词的向量上
        for (int c = 0; c < layer_size; ++c) {
            syn0_[input_offset + c] += neu1e[c];
        }
        
        // 继续处理下一个上下文词...
    }
    
    // =============================================================================
    // Skip-gram 训练总结：
    // 
    // 输入：中心词 w_0
    // 输出：每个上下文词 w_i (i ∈ [-window, +window], i ≠ 0)
    // 
    // 训练样本数：2 × window_size 个（每个上下文词一个样本）
    // 
    // 例如：window=2，句子 "I love natural language processing"
    //   中心词：language
    //   生成 4 个训练样本：
    //     (language, love)
    //     (language, natural)
    //     (language, processing)
    //     (language, {下一个词，如果有的话})
    // 
    // 网络结构（Negative Sampling）：
    //   输入：v(language)
    //   输出：P(love | language), P(natural | language), ...
    // 
    // 时间复杂度（每个中心词）：
    //   - Hierarchical Softmax: O(window_size × vector_size × log(vocab_size))
    //   - Negative Sampling: O(window_size × vector_size × negative)
    // =============================================================================
}

std::vector<float> Word2VecModel::GetWordVector(int word_index) const {
    size_t layer_size = config_.vector_size;
    size_t offset = word_index * layer_size;
    return std::vector<float>(syn0_.begin() + offset, 
                             syn0_.begin() + offset + layer_size);
}

void Word2VecModel::SaveVectors(const std::string& filename, bool binary) const {
    std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);
    
    file << vocab_.Size() << " " << config_.vector_size << "\n";
    
    for (size_t i = 0; i < vocab_.Size(); ++i) {
        file << vocab_.GetWord(i).word << " ";
        auto vec = GetWordVector(i);
        
        if (binary) {
            file.write(reinterpret_cast<const char*>(vec.data()), 
                      vec.size() * sizeof(float));
        } else {
            for (float val : vec) {
                file << val << " ";
            }
        }
        file << "\n";
    }
}

void Word2VecModel::LoadVectors(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open vector file: " + filename);
    }
    
    // 读取头部信息：词汇量和向量维度
    size_t vocab_size;
    size_t vector_size;
    file >> vocab_size >> vector_size;
    
    if (vector_size != static_cast<size_t>(config_.vector_size)) {
        throw std::runtime_error("Vector size mismatch: file has " + 
                                std::to_string(vector_size) + ", config has " + 
                                std::to_string(config_.vector_size));
    }
    
    // 重新初始化 syn0_
    syn0_.clear();
    syn0_.resize(vocab_size * vector_size);
    
    std::string word;
    char c;
    file.get(c);  // 跳过换行符
    
    // 读取每个词的向量
    for (size_t i = 0; i < vocab_size; ++i) {
        // 读取词
        word.clear();
        while (file.get(c) && c != ' ') {
            word += c;
        }
        
        // 读取向量数据
        // 检查是否是二进制格式（通过尝试读取）
        std::streampos pos = file.tellg();
        std::vector<float> vec(vector_size);
        
        // 尝试按二进制读取
        file.read(reinterpret_cast<char*>(vec.data()), vector_size * sizeof(float));
        
        // 检查是否成功且下一个字符是换行符
        bool is_binary = file.good();
        if (is_binary) {
            file.get(c);
            is_binary = (c == '\n' || c == '\r');
        }
        
        if (!is_binary) {
            // 文本格式，回退并按文本读取
            file.clear();
            file.seekg(pos);
            for (size_t j = 0; j < vector_size; ++j) {
                file >> vec[j];
            }
            file.get(c);  // 跳过换行符
        }
        
        // 保存向量
        std::copy(vec.begin(), vec.end(), syn0_.begin() + i * vector_size);
    }
}

} // namespace word2vec
