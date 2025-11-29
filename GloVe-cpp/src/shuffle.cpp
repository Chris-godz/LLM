/**
 * @file shuffle.cpp
 * @brief 共现数据打乱工具
 * 
 * 对应原始 GloVe: src/shuffle.c
 * 
 * 用法: shuffle [选项] < cooccurrence.bin > cooccurrence.shuf.bin
 */

#include "common.h"
#include <getopt.h>
#include <ctime>
#include <random>

// ============================================================================
// 全局变量
// ============================================================================

int verbose = 2;
int seed = 0;               // 随机种子，0 表示使用时间
real memory_limit = 2.0;    // 内存限制 (GB)
long long array_size;       // 每次打乱的数组大小

// ============================================================================
// 核心函数
// ============================================================================

/**
 * @brief Fisher-Yates 洗牌算法
 * 
 * TODO: 实现这个函数
 * 参考: GloVe/src/shuffle.c 中的 shuffle
 */
void shuffle_array(std::vector<CREC>& array) {
    // TODO: 实现
    // std::mt19937 rng(seed);
    // for (size_t i = array.size() - 1; i > 0; --i) {
    //     std::uniform_int_distribution<size_t> dist(0, i);
    //     size_t j = dist(rng);
    //     std::swap(array[i], array[j]);
    // }
}

/**
 * @brief 分块打乱数据
 * 
 * TODO: 实现这个函数
 * 参考: GloVe/src/shuffle.c 中的 shuffle_by_chunks
 */
int shuffle_by_chunks() {
    std::cerr << "Error: shuffle_by_chunks() not implemented yet\n";
    return 1;
}

// ============================================================================
// 主函数
// ============================================================================

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " [OPTIONS] < cooccurrence.bin > cooccurrence.shuf.bin\n\n"
              << "Options:\n"
              << "  -h, --help            显示帮助信息\n"
              << "  -v, --verbose <int>   详细程度 (0, 1, 2), 默认 2\n"
              << "  -m, --memory <float>  内存限制 (GB), 默认 2.0\n"
              << "  -s, --seed <int>      随机种子, 默认 0 (使用时间)\n";
}

int main(int argc, char** argv) {
    static struct option long_options[] = {
        {"help",    no_argument,       nullptr, 'h'},
        {"verbose", required_argument, nullptr, 'v'},
        {"memory",  required_argument, nullptr, 'm'},
        {"seed",    required_argument, nullptr, 's'},
        {nullptr, 0, nullptr, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "hv:m:s:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                verbose = std::atoi(optarg);
                break;
            case 'm':
                memory_limit = std::atof(optarg);
                break;
            case 's':
                seed = std::atoi(optarg);
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // 初始化随机种子
    if (seed == 0) {
        seed = static_cast<int>(std::time(nullptr));
    }
    if (verbose > 0) {
        std::cerr << "Using random seed " << seed << "\n";
    }
    
    // 计算数组大小
    array_size = static_cast<long long>(memory_limit * 1e9 / sizeof(CREC));
    
    return shuffle_by_chunks();
}
