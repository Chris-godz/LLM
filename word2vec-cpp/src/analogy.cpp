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
        std::cout << "Usage: analogy <vector_file>\n";
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
    std::cout << "Word analogy: Enter three words (e.g., 'king man woman' to find 'queen')\n";
    std::cout << "Computation: vec(third) - vec(second) + vec(first)\n";
    std::cout << "Example: woman - man + king = queen\n";
    std::cout << "Enter EXIT to quit\n\n";
    std::cout << "Enter three words: ";
    
    std::string input;
    while (std::getline(std::cin, input)) {
        if (input == "EXIT") break;
        if (input.empty()) {
            std::cout << "\nEnter three words: ";
            continue;
        }
        
        // 解析输入
        std::istringstream iss(input);
        std::vector<std::string> words;
        std::string word;
        while (iss >> word) {
            words.push_back(word);
        }
        
        if (words.size() != 3) {
            std::cout << "Error: Please enter exactly three words\n";
            std::cout << "\nEnter three words: ";
            continue;
        }
        
        // 查找三个词的向量
        std::vector<std::vector<float>*> vecs(3, nullptr);
        bool all_found = true;
        
        for (size_t i = 0; i < 3; ++i) {
            auto it = std::find_if(word_vectors.begin(), word_vectors.end(),
                                  [&words, i](const WordVector& wv) { 
                                      return wv.word == words[i]; 
                                  });
            if (it != word_vectors.end()) {
                vecs[i] = &(it->vec);
            } else {
                std::cout << "Word \"" << words[i] << "\" not found in vocabulary\n";
                all_found = false;
                break;
            }
        }
        
        if (!all_found) {
            std::cout << "\nEnter three words: ";
            continue;
        }
        
        // 计算类比向量: vec[2] - vec[1] + vec[0]
        // 例如: woman - man + king = queen
        std::vector<float> analogy_vec(vector_size);
        for (size_t i = 0; i < vector_size; ++i) {
            analogy_vec[i] = (*vecs[2])[i] - (*vecs[1])[i] + (*vecs[0])[i];
        }
        
        // 归一化
        float norm = 0.0f;
        for (float v : analogy_vec) {
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (float& v : analogy_vec) {
            v /= norm;
        }
        
        // 找最相似的词（排除输入的三个词）
        auto cmp = [](const std::pair<float, std::string>& a, 
                     const std::pair<float, std::string>& b) {
            return a.first > b.first;  // 小根堆
        };
        std::priority_queue<std::pair<float, std::string>,
                          std::vector<std::pair<float, std::string>>,
                          decltype(cmp)> top_words(cmp);
        
        for (const auto& wv : word_vectors) {
            // 跳过输入的三个词
            if (wv.word == words[0] || wv.word == words[1] || wv.word == words[2]) {
                continue;
            }
            
            float sim = CosineSimilarity(analogy_vec, wv.vec);
            top_words.push({sim, wv.word});
            if (top_words.size() > 40) {
                top_words.pop();
            }
        }
        
        // 输出结果
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
        
        std::cout << "\nEnter three words: ";
    }
    
    return 0;
}
