/**
 * @file vocab_count.cpp
 * @brief 词汇统计工具
 * 
 * 对应原始 GloVe: src/vocab_count.c
 * 
 * 用法: vocab_count [选项] < corpus.txt > vocab.txt
 */

#include "common.h"
#include <getopt.h>

// ============================================================================
// 全局变量
// ============================================================================

int verbose = 2;           // 详细程度 0, 1, 2
long long min_count = 1;   // 最小词频
long long max_vocab = 0;   // 最大词汇量，0 表示无限制

// ============================================================================
// 数据结构
// ============================================================================

struct VOCAB {
    std::string word;
    long long count;
};

// ============================================================================
// 核心函数
// ============================================================================

/**
 * @brief 词频比较函数（用于排序）
 * 按词频降序，词频相同则按字母顺序
 */
bool compare_vocab_tie(const VOCAB& a, const VOCAB& b) {
    if (a.count != b.count) return a.count > b.count;
    return a.word < b.word;
}

/**
 * @brief 词频比较函数（不考虑字母顺序）
 */
bool compare_vocab(const VOCAB& a, const VOCAB& b) {
    return a.count > b.count;
}

/**
 * @brief 构建词汇表
 * 
 * TODO: 实现这个函数
 * 参考: GloVe/src/vocab_count.c 中的 get_counts
 * 
 * 步骤:
 * 1. 使用哈希表统计词频
 * 2. 转移到数组并排序
 * 3. 应用过滤条件
 * 4. 输出结果
 */
int get_counts() {
    // TODO: 实现词汇统计
    //
    // std::unordered_map<std::string, long long> word_counts;
    // long long total_tokens = 0;
    // std::string word;
    //
    // std::cerr << "BUILDING VOCABULARY\n";
    // if (verbose > 1) std::cerr << "Processed " << total_tokens << " tokens.";
    //
    // while (!std::cin.eof()) {
    //     int nl = get_word(word, std::cin);
    //     if (nl) continue;
    //     if (word.empty()) continue;
    //     if (word == "<unk>") {
    //         std::cerr << "\nError: <unk> found in corpus. Please remove it.\n";
    //         return 1;
    //     }
    //     word_counts[word]++;
    //     total_tokens++;
    //     if (verbose > 1 && total_tokens % 100000 == 0) {
    //         std::cerr << "\033[11G" << total_tokens << " tokens.";
    //     }
    // }
    // if (verbose > 1) std::cerr << "\033[0GProcessed " << total_tokens << " tokens.\n";
    //
    // // 转移到数组
    // std::vector<VOCAB> vocab;
    // vocab.reserve(word_counts.size());
    // for (const auto& pair : word_counts) {
    //     vocab.push_back({pair.first, pair.second});
    // }
    // if (verbose > 1) std::cerr << "Counted " << vocab.size() << " unique words.\n";
    //
    // // 排序
    // long long actual_max = (max_vocab > 0 && max_vocab < vocab.size()) ? max_vocab : vocab.size();
    // if (max_vocab > 0 && max_vocab < vocab.size()) {
    //     std::partial_sort(vocab.begin(), vocab.begin() + max_vocab, vocab.end(), compare_vocab);
    // }
    // std::sort(vocab.begin(), vocab.begin() + actual_max, compare_vocab_tie);
    //
    // // 输出
    // long long i;
    // for (i = 0; i < actual_max; i++) {
    //     if (vocab[i].count < min_count) {
    //         if (verbose > 0) std::cerr << "Truncating vocabulary at min count " << min_count << ".\n";
    //         break;
    //     }
    //     std::cout << vocab[i].word << " " << vocab[i].count << "\n";
    // }
    //
    // if (i == max_vocab && max_vocab < vocab.size()) {
    //     if (verbose > 0) std::cerr << "Truncating vocabulary at size " << max_vocab << ".\n";
    // }
    // std::cerr << "Using vocabulary of size " << i << ".\n\n";
    //
    // return 0;
    
    std::cerr << "Error: get_counts() not implemented yet\n";
    return 1;
}

// ============================================================================
// 主函数
// ============================================================================

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [OPTIONS] < corpus.txt > vocab.txt\n\n"
              << "Options:\n"
              << "  -h, --help              显示帮助信息\n"
              << "  -v, --verbose <int>     详细程度 (0, 1, 2), 默认 2\n"
              << "  -c, --min-count <int>   最小词频, 默认 1\n"
              << "  -m, --max-vocab <int>   最大词汇量, 默认 0 (无限制)\n";
}

int main(int argc, char** argv) {
    static struct option long_options[] = {
        {"help",      no_argument,       nullptr, 'h'},
        {"verbose",   required_argument, nullptr, 'v'},
        {"min-count", required_argument, nullptr, 'c'},
        {"max-vocab", required_argument, nullptr, 'm'},
        {nullptr, 0, nullptr, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "hv:c:m:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                verbose = std::atoi(optarg);
                break;
            case 'c':
                min_count = std::atoll(optarg);
                break;
            case 'm':
                max_vocab = std::atoll(optarg);
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    return get_counts();
}
