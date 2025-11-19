#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace word2vec {

struct VocabWord {
    std::string word;
    long long count;
    std::vector<int> point;      // Huffman树路径
    std::vector<char> code;      // Huffman编码
    
    VocabWord(const std::string& w = "", long long c = 0) 
        : word(w), count(c) {}
};

class Vocabulary {
public:
    Vocabulary() = default;
    
    // 从训练文件学习词汇表
    void LearnFromFile(const std::string& filename, int min_count = 5);
    
    // 保存/加载词汇表
    void Save(const std::string& filename) const;
    void Load(const std::string& filename);
    
    // 查询
    int GetWordIndex(const std::string& word) const;
    const VocabWord& GetWord(int index) const { return vocab_[index]; }
    size_t Size() const { return vocab_.size(); }
    long long TotalWords() const { return train_words_; }
    
    // Huffman树构建
    void BuildHuffmanTree();
    
private:
    std::vector<VocabWord> vocab_;
    std::unordered_map<std::string, int> word_to_index_;
    long long train_words_ = 0;
    
    void SortAndFilter(int min_count);
};

} // namespace word2vec
