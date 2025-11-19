#include "trainer.hpp"
#include "vocabulary.hpp"
#include <thread>
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace word2vec {

Trainer::Trainer(const Vocabulary& vocab, const Config& config)
    : vocab_(vocab), config_(config) {
    model_ = std::make_unique<Word2VecModel>(vocab_, config_.model_config);
    current_alpha_ = config_.model_config.alpha;
}

void Trainer::Train() {
    auto start_time = std::chrono::steady_clock::now();
    
    std::cout << "Starting training...\n";
    std::cout << "Vocabulary size: " << vocab_.Size() << "\n";
    std::cout << "Training file: " << config_.train_file << "\n";
    std::cout << "Threads: " << config_.num_threads << "\n";
    
    // 创建训练线程
    std::vector<std::thread> threads;
    for (int i = 0; i < config_.num_threads; ++i) {
        threads.emplace_back(&Trainer::TrainThread, this, i);
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nTraining completed in " << duration.count() << " seconds\n";
    
    // 保存模型
    model_->SaveVectors(config_.output_file);
    std::cout << "Vectors saved to " << config_.output_file << "\n";
}

void Trainer::TrainThread(int thread_id) {
    // =============================================================================
    // 线程局部变量初始化
    // =============================================================================
    
    // 句子缓冲区：存储当前句子的词索引（最多1000个词）
    std::vector<int> sentence;
    sentence.reserve(1000);
    
    // 词计数：当前线程处理的词数
    long long word_count = 0;
    long long last_word_count = 0;
    
    // 迭代计数：每个线程独立完成指定次数的迭代
    int local_iter = config_.iterations;
    
    // 随机数生成器：用于子采样和随机窗口
    std::mt19937_64 rng(thread_id);
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    
    // =============================================================================
    // 打开训练文件并定位到该线程负责的起始位置
    // =============================================================================
    
    std::ifstream file(config_.train_file, std::ios::binary);
    if (!file) {
        std::cerr << "Thread " << thread_id << ": Cannot open training file\n";
        return;
    }
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    long long file_size = file.tellg();
    
    // 计算该线程负责的文件起始位置
    // 将文件平均分配给所有线程：thread_id * (file_size / num_threads)
    long long start_pos = file_size / config_.num_threads * thread_id;
    file.seekg(start_pos);
    
    // 如果不是第一个线程，跳过当前词的剩余部分
    // 读取并丢弃一个词，确保从完整的词开始
    if (thread_id != 0) {
        std::string word;
        file >> word;  // 跳过可能不完整的第一个词
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // =============================================================================
    // 主训练循环：重复迭代训练数据
    // =============================================================================
    
    while (true) {
        // -------------------------------------------------------------------------
        // 学习率调度和进度显示（每处理10000个词更新一次）
        // -------------------------------------------------------------------------
        
        if (word_count - last_word_count > 10000) {
            // 更新全局词计数（原子操作）
            word_count_actual_ += word_count - last_word_count;
            last_word_count = word_count;
            
            // 计算当前训练进度
            long long current_count = word_count_actual_.load();
            long long total_words = config_.iterations * vocab_.TotalWords();
            
            // 动态降低学习率：alpha = starting_alpha * (1 - progress)
            // 随着训练进度增加，学习率线性衰减
            float progress = static_cast<float>(current_count) / (total_words + 1);
            current_alpha_ = config_.model_config.alpha * (1.0f - progress);
            
            // 设置学习率下限：不低于初始学习率的0.01%
            if (current_alpha_ < config_.model_config.alpha * 0.0001f) {
                current_alpha_ = config_.model_config.alpha * 0.0001f;
            }
            
            // 显示训练进度（所有线程都可以显示，但加锁避免混乱输出）
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            float words_per_sec = elapsed > 0 ? current_count / (float)elapsed / 1000.0f : 0.0f;
            float alpha = current_alpha_.load();
            
            printf("\rAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  ", 
                   alpha, progress * 100.0f, words_per_sec);
            fflush(stdout);
        }
        
        // -------------------------------------------------------------------------
        // 读取句子：从文件读取一个完整句子
        // -------------------------------------------------------------------------
        
        if (sentence.empty()) {
            std::string word_str;
            
            while (true) {
                file >> word_str;
                if (file.eof()) break;  // 关键:遇到EOF立即break
                if (file.fail()) break;  // 读取失败也break
                
                // 查找词在词汇表中的索引
                int word_index = vocab_.GetWordIndex(word_str);
                
                // 未登录词（不在词汇表中）：跳过
                if (word_index == -1) continue;
                
                word_count++;
                
                // 遇到句子分隔符 </s>：句子结束
                if (word_index == 0) break;
                
                // =====================================================================
                // 子采样（Subsampling）：随机丢弃高频词
                // =====================================================================
                // 高频词（如 "the", "a"）对训练帮助不大，按概率丢弃可以：
                // 1. 加速训练（减少计算量）
                // 2. 增加低频词的相对权重
                // 
                // 保留概率公式：P(keep) = (sqrt(freq/t) + 1) * t/freq
                // 其中：
                // - freq = word_count / total_words（词频）
                // - t = sample（阈值，默认 1e-3）
                // 
                // 特性分析：
                // - 当 freq >> t（高频词）：P(keep) ≈ sqrt(t/freq)，接近 0 → 容易丢弃
                // - 当 freq ≈ t（中频词）：P(keep) ≈ 1 → 50% 概率保留
                // - 当 freq << t（低频词）：P(keep) >> 1 → 几乎全部保留
                // 
                // =====================================================================
                
                if (config_.model_config.sample > 0) {
                    const auto& vocab_word = vocab_.GetWord(word_index);
                    float freq = static_cast<float>(vocab_word.count) / vocab_.TotalWords();
                    
                    // 计算保留概率：ran = (sqrt(freq/sample) + 1) * sample/freq
                    float keep_prob = (std::sqrt(freq / config_.model_config.sample) + 1.0f) 
                                    * (config_.model_config.sample / freq);
                    
                    // 生成随机数 [0, 1)，如果保留概率 < 随机数，则丢弃该词
                    if (keep_prob < uniform_dist(rng)) {
                        continue;  // 丢弃该词
                    }
                }
                
                // 将词添加到句子缓冲区
                sentence.push_back(word_index);
                
                // 句子长度限制：最多1000个词
                if (sentence.size() >= 1000) break;
            }
        }
        
        // -------------------------------------------------------------------------
        // 迭代控制：检查是否完成一轮迭代
        // -------------------------------------------------------------------------
        
        // 条件1：文件读到EOF
        // 条件2：当前轮已处理完分配的词数
        // 每个线程每轮负责 total_words / num_threads 个词
        if (file.eof() || word_count > vocab_.TotalWords() / config_.num_threads) {
            // 更新全局计数
            word_count_actual_ += word_count - last_word_count;
            local_iter--;
            
            // 所有迭代完成：退出训练
            if (local_iter == 0) break;
            
            // 重置计数器，准备下一轮迭代
            word_count = 0;
            last_word_count = 0;
            sentence.clear();
            
            // 重新定位到该线程负责的文件起始位置,开始下一轮迭代
            file.clear();
            file.seekg(start_pos);
            if (thread_id != 0) {
                std::string word;
                file >> word;  // 跳过可能不完整的第一个词
            }
            
            continue;
        }
        
        // -------------------------------------------------------------------------
        // 句子训练：对句子中的每个词进行训练
        // -------------------------------------------------------------------------
        
        if (!sentence.empty()) {
            // 遍历句子中的每个位置
            for (size_t sentence_pos = 0; sentence_pos < sentence.size(); ++sentence_pos) {
                int word = sentence[sentence_pos];
                
                // 跳过未登录词（理论上不应该出现）
                if (word == -1) {
                    std::cout << "Thread " << thread_id << ": Encountered unknown word index -1\n";
                    continue;
                }
                
                // =================================================================
                // 动态窗口大小：随机选择 [1, window] 之间的实际窗口
                // =================================================================
                // 不是总用最大窗口，而是随机缩减窗口大小
                // 这样可以：
                // 1. 给近距离词对更高权重（小窗口出现频率更高）
                // 2. 增加训练样本的多样性
                // =================================================================
                
                std::uniform_int_distribution<int> window_dist(0, config_.model_config.window - 1);
                int reduced_window = window_dist(rng);
                
                // 调用模型的训练函数
                // - CBOW: 用上下文预测中心词
                // - Skip-gram: 用中心词预测上下文
                if (config_.model_config.use_cbow) {
                    model_->TrainCBOW(sentence_pos, sentence, current_alpha_.load(), reduced_window);
                } else {
                    model_->TrainSkipGram(sentence_pos, sentence, current_alpha_.load(), reduced_window);
                }
            }
            
            // 当前句子训练完成，清空缓冲区，准备读取下一个句子
            sentence.clear();
        }
    }
    
    // =============================================================================
    // 线程结束清理
    // =============================================================================
    
    file.close();
}

} // namespace word2vec
