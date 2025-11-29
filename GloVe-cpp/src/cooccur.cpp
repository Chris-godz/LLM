/**
 * @file cooccur.cpp
 * @brief 共现矩阵统计工具
 * 
 * 对应原始 GloVe: src/cooccur.c
 * 
 * 用法: cooccur [选项] < corpus.txt > cooccurrence.bin
 */

#include "common.h"
#include <getopt.h>

// ============================================================================
// 全局变量
// ============================================================================

int verbose = 2;
int window_size = 15;       // 上下文窗口大小
int symmetric = 1;          // 是否对称
int distance_weighting = 1; // 是否距离加权
real memory_limit = 4.0;    // 内存限制 (GB)
std::string vocab_file;     // 词汇表文件

// ============================================================================
// 核心函数
// ============================================================================

/**
 * @brief 加载词汇表到哈希表
 * 
 * TODO: 实现这个函数
 */
std::unordered_map<std::string, long long> load_vocab() {
    std::unordered_map<std::string, long long> vocab;
    
    // TODO: 实现
    // std::ifstream fin(vocab_file);
    // if (!fin.is_open()) {
    //     log_file_loading_error("vocabulary file", vocab_file.c_str());
    //     return vocab;
    // }
    // 
    // std::string word;
    // long long count;
    // long long id = 0;
    // while (fin >> word >> count) {
    //     vocab[word] = id++;
    // }
    // 
    // if (verbose > 0) std::cerr << "Loaded vocab with " << vocab.size() << " words.\n";
    
    return vocab;
}

/**
 * @brief 构建共现矩阵
 * 
 * TODO: 实现这个函数
 * 参考: GloVe/src/cooccur.c
 * 
 * 步骤:
 * 1. 加载词汇表
 * 2. 读取语料，用滑动窗口统计共现
 * 3. 应用距离加权
 * 4. 写入二进制文件
 */
int get_cooccurrence() {
    std::cerr << "Error: get_cooccurrence() not implemented yet\n";
    return 1;
}

// ============================================================================
// 主函数
// ============================================================================

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [OPTIONS] < corpus.txt > cooccurrence.bin\n\n"
              << "Options:\n"
              << "  -h, --help                  显示帮助信息\n"
              << "  -v, --verbose <int>         详细程度 (0, 1, 2), 默认 2\n"
              << "  -V, --vocab-file <file>     词汇表文件 (必需)\n"
              << "  -w, --window-size <int>     窗口大小, 默认 15\n"
              << "  -s, --symmetric <int>       是否对称 (0, 1), 默认 1\n"
              << "  -d, --distance-weighting <int>  距离加权 (0, 1), 默认 1\n"
              << "  -m, --memory <float>        内存限制 (GB), 默认 4.0\n";
}

int main(int argc, char** argv) {
    static struct option long_options[] = {
        {"help",              no_argument,       nullptr, 'h'},
        {"verbose",           required_argument, nullptr, 'v'},
        {"vocab-file",        required_argument, nullptr, 'V'},
        {"window-size",       required_argument, nullptr, 'w'},
        {"symmetric",         required_argument, nullptr, 's'},
        {"distance-weighting", required_argument, nullptr, 'd'},
        {"memory",            required_argument, nullptr, 'm'},
        {nullptr, 0, nullptr, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "hv:V:w:s:d:m:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                verbose = std::atoi(optarg);
                break;
            case 'V':
                vocab_file = optarg;
                break;
            case 'w':
                window_size = std::atoi(optarg);
                break;
            case 's':
                symmetric = std::atoi(optarg);
                break;
            case 'd':
                distance_weighting = std::atoi(optarg);
                break;
            case 'm':
                memory_limit = std::atof(optarg);
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    if (vocab_file.empty()) {
        std::cerr << "Error: --vocab-file is required\n\n";
        print_usage(argv[0]);
        return 1;
    }
    
    return get_cooccurrence();
}
