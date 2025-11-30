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
 * 从标准输入读取语料，统计词频，按频率降序输出到标准输出。
 * 
 * 算法流程:
 * 1. 逐词读取语料，用哈希表统计每个词的出现次数
 * 2. 将哈希表转移到数组，便于排序
 * 3. 按词频降序排序（词频相同则按字母顺序）
 * 4. 应用 min_count 和 max_vocab 过滤
 * 5. 输出格式: "word count\n"
 * 
 * @return 0 成功, 1 失败
 */
int get_counts() {
    // ========================================================================
    // 第一步：统计词频
    // ========================================================================
    // 使用 unordered_map 作为哈希表，key 是单词，value 是出现次数
    // 时间复杂度 O(n)，n 为语料中的 token 数量
    std::unordered_map<std::string, long long> word_counts;
    long long total_tokens = 0;  // 总 token 数（含重复）
    std::string word;
    
    std::cerr << "BUILDING VOCABULARY\n";
    if (verbose > 1) {
        std::cerr << "Processed " << total_tokens << " tokens.";
    }
    
    // 从标准输入逐词读取
    // get_word 返回 1 表示遇到换行或 EOF（文档边界），0 表示正常读取
    while (!std::cin.eof()) {
        int is_newline = get_word(word, std::cin);
        
        // 跳过空词和文档边界
        if (is_newline || word.empty()) {
            continue;
        }
        
        // <unk> 是保留词，不允许出现在原始语料中
        // 后续处理会用 <unk> 表示未登录词
        if (word == "<unk>") {
            std::cerr << "\nError: <unk> found in corpus.\n";
            std::cerr << "Please remove <unk> tokens from corpus.\n";
            return 1;
        }
        
        // 统计词频：如果词不存在，unordered_map 会自动初始化为 0
        word_counts[word]++;
        total_tokens++;
        
        // 每处理 10 万个 token 打印一次进度
        // \033[11G 是 ANSI 转义码，将光标移动到第 11 列，实现原地更新
        if (verbose > 1 && total_tokens % 100000 == 0) {
            std::cerr << "\033[11G" << total_tokens << " tokens.";
        }
    }
    
    // 打印最终统计
    if (verbose > 1) {
        std::cerr << "\033[0GProcessed " << total_tokens << " tokens.\n";
    }
    
    // ========================================================================
    // 第二步：转移到数组
    // ========================================================================
    // 哈希表不支持排序，需要转移到 vector
    std::vector<VOCAB> vocab;
    vocab.reserve(word_counts.size());  // 预分配空间，避免多次扩容
    
    for (const auto& pair : word_counts) {
        vocab.push_back({pair.first, pair.second});
    }
    
    if (verbose > 1) {
        std::cerr << "Counted " << vocab.size() << " unique words.\n";
    }
    
    // ========================================================================
    // 第三步：排序
    // ========================================================================
    // 按词频降序排序，词频相同则按字母升序（保证输出稳定）
    // 
    // 如果设置了 max_vocab 限制，使用 partial_sort 只排序前 max_vocab 个
    // partial_sort 时间复杂度 O(n * log(k))，比完全排序 O(n * log(n)) 更高效
    long long vocab_size = static_cast<long long>(vocab.size());
    long long actual_max = (max_vocab > 0 && max_vocab < vocab_size) 
                           ? max_vocab : vocab_size;
    
    if (max_vocab > 0 && max_vocab < vocab_size) {
        // 只需要前 max_vocab 个最高频词
        std::partial_sort(vocab.begin(), 
                          vocab.begin() + max_vocab, 
                          vocab.end(), 
                          compare_vocab_tie);
    } else {
        // 无限制，全部排序
        std::sort(vocab.begin(), vocab.end(), compare_vocab_tie);
    }
    
    // ========================================================================
    // 第四步：输出结果
    // ========================================================================
    // 输出格式: "word count\n"
    // 按词频降序输出，同时应用 min_count 和 max_vocab 过滤
    long long i;
    for (i = 0; i < actual_max; i++) {
        // 如果词频低于阈值，停止输出
        if (vocab[i].count < min_count) {
            if (verbose > 0) {
                std::cerr << "Truncating vocabulary at min count " 
                          << min_count << ".\n";
            }
            break;
        }
        // 输出到标准输出
        std::cout << vocab[i].word << " " << vocab[i].count << "\n";
    }
    
    // 如果是因为 max_vocab 限制而截断
    if (i == max_vocab && max_vocab < vocab_size) {
        if (verbose > 0) {
            std::cerr << "Truncating vocabulary at size " << max_vocab << ".\n";
        }
    }
    
    std::cerr << "Using vocabulary of size " << i << ".\n\n";
    
    return 0;
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
