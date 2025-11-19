#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <queue>
#include <sstream>

struct WordVector {
    std::string word;
    std::vector<float> vec;
};

// 计算余弦相似度
float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: distance <vector_file>\n";
        return 1;
    }
    
    std::string vector_file = argv[1];
    std::cout << "Loading vectors from " << vector_file << "...\n";
    
    // 读取向量文件
    std::ifstream file(vector_file);
    if (!file) {
        std::cerr << "Error: Cannot open file " << vector_file << "\n";
        return 1;
    }
    
    size_t vocab_size, vector_size;
    file >> vocab_size >> vector_size;
    
    std::vector<WordVector> word_vectors(vocab_size);
    std::cout << "Vocabulary size: " << vocab_size << ", Vector size: " << vector_size << "\n";
    
    // 读取所有词向量
    for (size_t i = 0; i < vocab_size; ++i) {
        file >> word_vectors[i].word;
        word_vectors[i].vec.resize(vector_size);
        for (size_t j = 0; j < vector_size; ++j) {
            file >> word_vectors[i].vec[j];
        }
        
        // 归一化向量
        float norm = 0.0f;
        for (float v : word_vectors[i].vec) {
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (float& v : word_vectors[i].vec) {
            v /= norm;
        }
    }
    
    std::cout << "Vectors loaded successfully!\n\n";
    std::cout << "Enter word or sentence (EXIT to break): ";
    
    std::string input;
    while (std::getline(std::cin, input)) {
        if (input == "EXIT") break;
        if (input.empty()) {
            std::cout << "\nEnter word or sentence (EXIT to break): ";
            continue;
        }
        
        // 查找输入词
        std::istringstream iss(input);
        std::vector<std::string> words;
        std::string word;
        while (iss >> word) {
            words.push_back(word);
        }
        
        // 计算查询向量（多个词则求平均）
        std::vector<float> query_vec(vector_size, 0.0f);
        int found_count = 0;
        
        for (const auto& w : words) {
            auto it = std::find_if(word_vectors.begin(), word_vectors.end(),
                                  [&w](const WordVector& wv) { return wv.word == w; });
            if (it != word_vectors.end()) {
                for (size_t i = 0; i < vector_size; ++i) {
                    query_vec[i] += it->vec[i];
                }
                found_count++;
            } else {
                std::cout << "Word \"" << w << "\" not found in vocabulary\n";
            }
        }
        
        if (found_count == 0) {
            std::cout << "\nEnter word or sentence (EXIT to break): ";
            continue;
        }
        
        // 平均并归一化
        float norm = 0.0f;
        for (float& v : query_vec) {
            v /= found_count;
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (float& v : query_vec) {
            v /= norm;
        }
        
        // 找最相似的词（使用优先队列，保留 top 40）
        auto cmp = [](const std::pair<float, std::string>& a, 
                     const std::pair<float, std::string>& b) {
            return a.first > b.first;  // 小根堆
        };
        std::priority_queue<std::pair<float, std::string>,
                          std::vector<std::pair<float, std::string>>,
                          decltype(cmp)> top_words(cmp);
        
        for (const auto& wv : word_vectors) {
            // 跳过输入词本身
            if (std::find(words.begin(), words.end(), wv.word) != words.end()) {
                continue;
            }
            
            float sim = CosineSimilarity(query_vec, wv.vec);
            top_words.push({sim, wv.word});
            if (top_words.size() > 40) {
                top_words.pop();
            }
        }
        
        // 输出结果（从小根堆中取出，需要反转）
        std::vector<std::pair<float, std::string>> results;
        while (!top_words.empty()) {
            results.push_back(top_words.top());
            top_words.pop();
        }
        std::reverse(results.begin(), results.end());
        
        std::cout << "\n                                              Word       Cosine distance\n";
        std::cout << "------------------------------------------------------------------------\n";
        for (const auto& r : results) {
            printf("%50s\t\t%f\n", r.second.c_str(), r.first);
        }
        
        std::cout << "\nEnter word or sentence (EXIT to break): ";
    }
    
    return 0;
}
