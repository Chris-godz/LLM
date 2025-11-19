#include "vocabulary.hpp"
#include <climits>
#include <fstream>
#include <algorithm>
#include <iostream>

namespace word2vec {

void Vocabulary::LearnFromFile(const std::string& filename, int min_count) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open training file: " + filename);
    }
    
    std::cout << "Learning vocabulary from " << filename << "...\n";
    
    // 初始化
    vocab_.clear();
    word_to_index_.clear();
    train_words_ = 0;
    
    // 添加 </s> 作为句子分隔符
    vocab_.emplace_back("</s>", 0);
    word_to_index_["</s>"] = 0;
    
    // 统计词频
    std::string word;
    long long word_count = 0;
    
    while (file >> word) {
        if (word.empty()) continue;
        
        train_words_++;
        word_count++;
        
        // 进度显示（每10万词）
        if (word_count % 100000 == 0) {
            std::cout << word_count / 1000 << "K\r" << std::flush;
        }
        
        // 查找或添加词
        auto it = word_to_index_.find(word);
        if (it == word_to_index_.end()) {
            // 新词
            size_t idx = vocab_.size();
            vocab_.emplace_back(word, 1);
            word_to_index_[word] = idx;
        } else {
            // 已存在的词，增加计数
            vocab_[it->second].count++;
        }
    }
    
    // 排序并过滤
    SortAndFilter(min_count);
    
    // 构建Huffman树
    BuildHuffmanTree();
    
    std::cout << "\nVocabulary size: " << vocab_.size() << "\n";
    std::cout << "Words in train file: " << train_words_ << "\n";
}

void Vocabulary::SortAndFilter(int min_count) {
    // 按词频降序排序（保持</s>在第一位）
    if (vocab_.size() > 1) {
        std::sort(vocab_.begin() + 1, vocab_.end(), 
                  [](const VocabWord& a, const VocabWord& b) {
                      return a.count > b.count;
                  });
    }
    
    // 重建哈希表并过滤低频词
    word_to_index_.clear();
    std::vector<VocabWord> filtered_vocab;
    
    for (size_t i = 0; i < vocab_.size(); ++i) {
        // 保留</s>或词频>=min_count的词
        if (i == 0 || vocab_[i].count >= min_count) {
            word_to_index_[vocab_[i].word] = filtered_vocab.size();
            filtered_vocab.push_back(std::move(vocab_[i]));
        }
    }
    
    vocab_ = std::move(filtered_vocab);
    
    // 重新计算总词数
    train_words_ = 0;
    for (const auto& word : vocab_) {
        train_words_ += word.count;
    }
}

void Vocabulary::BuildHuffmanTree() {
    size_t vocab_size = vocab_.size();
    if (vocab_size < 2) return;
    
    // 分配Huffman树所需的数组
    std::vector<long long> count(vocab_size * 2);
    std::vector<long long> binary(vocab_size * 2);
    std::vector<long long> parent_node(vocab_size * 2);
    
    // 初始化：前vocab_size个是叶子节点（词）
    for (size_t i = 0; i < vocab_size; ++i) {
        count[i] = vocab_[i].count;
    }
    // 后面的是内部节点，初始化为大值
    for (size_t i = vocab_size; i < vocab_size * 2; ++i) {
        count[i] = LONG_LONG_MAX;
    }
    
    // 构建Huffman树：贪心地合并最小频率的两个节点
    long long pos1 = vocab_size - 1;  // 指向叶子节点
    long long pos2 = vocab_size;      // 指向新生成的内部节点
    long long min1i, min2i;
    
    for (size_t a = 0; a < vocab_size - 1; ++a) {
        // 找到两个最小的节点
        if (pos1 >= 0) {
            min1i = (count[pos1] < count[pos2]) ? pos1-- : pos2++;
        } else {
            min1i = pos2++;
        }
        
        if (pos1 >= 0) {
            min2i = (count[pos1] < count[pos2]) ? pos1-- : pos2++;
        } else {
            min2i = pos2++;
        }
        
        // 创建父节点
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;  // 右子树标记为1
    }
    
    // 为每个词生成Huffman编码
    std::vector<char> code(40);  // MAX_CODE_LENGTH
    std::vector<int> point(40);
    
    for (size_t a = 0; a < vocab_size; ++a) {
        long long b = a;
        int i = 0;
        
        // 从叶子向上遍历到根
        while (true) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == static_cast<long long>(vocab_size * 2 - 2)) break;  // 到达根节点
        }
        
        // 保存编码长度和路径
        vocab_[a].code.resize(i);
        vocab_[a].point.resize(i + 1);
        vocab_[a].point[0] = vocab_size - 2;
        
        // 反转编码（从根到叶）
        for (int j = 0; j < i; ++j) {
            vocab_[a].code[i - j - 1] = code[j];
            vocab_[a].point[i - j] = point[j];
        }
    }
}

int Vocabulary::GetWordIndex(const std::string& word) const {
    auto it = word_to_index_.find(word);
    return it != word_to_index_.end() ? it->second : -1;
}

void Vocabulary::Save(const std::string& filename) const {
    std::ofstream file(filename);
    for (const auto& word : vocab_) {
        file << word.word << " " << word.count << "\n";
    }
}

void Vocabulary::Load(const std::string& filename) {
    // TODO: 实现加载
    (void)filename;
}

} // namespace word2vec
