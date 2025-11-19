#pragma once

#include <vector>
#include <memory>
#include <random>

namespace word2vec {

class Vocabulary;

class Word2VecModel {
public:
    struct Config {
        int vector_size = 100;           // 向量维度 (C: layer1_size)
        int window = 5;                  // 窗口大小
        float alpha = 0.025f;            // 学习率 (skip-gram默认), CBOW会在构造时改为0.05
        float sample = 1e-3f;            // 下采样阈值
        int negative = 5;                // 负采样数量 (C默认: 5)
        int min_count = 5;               // 最小词频 (C默认: 5)
        bool use_cbow = true;            // 使用CBOW (C默认: 1)
        bool hierarchical_softmax = false; // 使用层次softmax (C默认: 0)
        bool binary = false;             // 二进制输出 (C默认: 0)
        
        Config() = default;
    };
    
    Word2VecModel(const Vocabulary& vocab, const Config& config);
    
    // 训练
    void TrainSentence(const std::vector<int>& sentence, float alpha);
    
    // 获取词向量
    std::vector<float> GetWordVector(int word_index) const;
    
    // 保存/加载模型
    void SaveVectors(const std::string& filename, bool binary = false) const;
    void LoadVectors(const std::string& filename);
    
    // CBOW 训练
    void TrainCBOW(int word_pos, const std::vector<int>& sentence, 
                   float alpha, int reduced_window);
    
    // Skip-gram 训练
    void TrainSkipGram(int word_pos, const std::vector<int>& sentence,
                       float alpha, int reduced_window);
    
private:
    const Vocabulary& vocab_;
    Config config_;
    
    // 网络权重
    std::vector<float> syn0_;      // 输入层 embedding
    std::vector<float> syn1_;      // Hierarchical Softmax 权重
    std::vector<float> syn1neg_;   // Negative Sampling 权重
    
    // 负采样表
    std::vector<int> unigram_table_;
    
    // Sigmoid 查找表
    std::vector<float> exp_table_;
    
    // 随机数生成器
    std::mt19937_64 rng_;
    
    void InitNet();
    void InitUnigramTable();
    void InitExpTable();
};

} // namespace word2vec
