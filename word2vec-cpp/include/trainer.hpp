#pragma once

#include <string>
#include <memory>
#include <atomic>
#include "model.hpp"

namespace word2vec {

class Vocabulary;

class Trainer {
public:
    struct Config {
        std::string train_file;
        std::string output_file;
        int num_threads = 12;            // 线程数 (C默认: 12)
        int iterations = 5;              // 迭代次数 (C默认: 5)
        Word2VecModel::Config model_config;
        
        Config() = default;
    };
    
    Trainer(const Vocabulary& vocab, const Config& config);
    
    // 开始训练
    void Train();
    
private:
    const Vocabulary& vocab_;
    Config config_;
    std::unique_ptr<Word2VecModel> model_;
    
    std::atomic<long long> word_count_actual_{0};
    std::atomic<float> current_alpha_;
    
    void TrainThread(int thread_id);
};

} // namespace word2vec
