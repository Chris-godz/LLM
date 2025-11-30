/**
 * @file shuffle.cpp
 * @brief 共现数据打乱工具
 * 
 * 对应原始 GloVe: src/shuffle.c
 * 
 * 用法: shuffle [选项] < cooccurrence.bin > cooccurrence.shuf.bin
 * 
 * 算法流程：
 * 1. 从 stdin 读取二进制 CREC 记录到内存缓冲区
 * 2. 缓冲区满时，用 Fisher-Yates 算法打乱，写入临时文件
 * 3. 所有数据读完后，合并所有临时文件：
 *    - 从每个临时文件轮流读取一批记录
 *    - 混合后再次打乱，写入 stdout
 * 4. 删除临时文件
 */

#include "common.h"
#include <getopt.h>
#include <ctime>
#include <random>
#include <sstream>
#include <iomanip>

// ============================================================================
// 全局变量
// ============================================================================

int verbose = 2;
int seed = 0;               // 随机种子，0 表示使用时间
real memory_limit = 2.0;    // 内存限制 (GB)
long long array_size;       // 每次打乱的数组大小
std::string file_head = "temp_shuffle";  // 临时文件前缀

// 全局随机数生成器
std::mt19937_64 rng;

// ============================================================================
// 核心函数
// ============================================================================

/**
 * @brief Fisher-Yates 洗牌算法
 * 
 * 时间复杂度 O(n)，原地打乱数组
 * 每个元素被放到任意位置的概率相等
 */
void shuffle_array(std::vector<CREC>& array, size_t n) {
    for (size_t i = n - 1; i > 0; --i) {
        // 生成 [0, i] 范围内的随机数
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);
        std::swap(array[i], array[j]);
    }
}

/**
 * @brief 将数组写入二进制文件
 */
void write_chunk(const std::vector<CREC>& array, size_t n, std::ostream& out) {
    out.write(reinterpret_cast<const char*>(array.data()), n * sizeof(CREC));
}

/**
 * @brief 合并并打乱所有临时文件
 * 
 * 从每个临时文件轮流读取 array_size/num 条记录，
 * 混合后打乱，写入 stdout。
 * 这样可以让不同文件的记录交错混合。
 */
int shuffle_merge(int num_files) {
    if (verbose > 0) {
        std::cerr << "Merging " << num_files << " temp files...\n";
    }

    // 打开所有临时文件
    std::vector<std::ifstream> files(num_files);
    for (int i = 0; i < num_files; ++i) {
        std::ostringstream oss;
        oss << file_head << "_" << std::setw(4) << std::setfill('0') << i << ".bin";
        files[i].open(oss.str(), std::ios::binary);
        if (!files[i].is_open()) {
            log_file_loading_error("temp file", oss.str().c_str());
            return 1;
        }
    }

    std::vector<CREC> array(array_size);
    long long total_lines = 0;
    size_t per_file = array_size / num_files;  // 每个文件每轮读取的记录数

    if (verbose > 1) {
        std::cerr << "Merging temp files: processed 0 lines.";
    }

    while (true) {
        size_t count = 0;

        // 从每个文件轮流读取一批
        for (int j = 0; j < num_files; ++j) {
            if (files[j].eof()) continue;

            for (size_t k = 0; k < per_file && count < static_cast<size_t>(array_size); ++k) {
                if (!files[j].read(reinterpret_cast<char*>(&array[count]), sizeof(CREC))) {
                    break;  // EOF
                }
                count++;
            }
        }

        if (count == 0) break;  // 所有文件都读完了

        // 打乱并写出
        shuffle_array(array, count);
        write_chunk(array, count, std::cout);
        total_lines += count;

        if (verbose > 1) {
            std::cerr << "\033[31G" << total_lines << " lines.";
        }
    }

    std::cout.flush();
    if (verbose > 0) {
        std::cerr << "\033[0GMerging temp files: processed " << total_lines << " lines.\n";
    }

    // 关闭并删除临时文件
    for (int i = 0; i < num_files; ++i) {
        files[i].close();
        std::ostringstream oss;
        oss << file_head << "_" << std::setw(4) << std::setfill('0') << i << ".bin";
        std::remove(oss.str().c_str());
    }

    std::cerr << "\n";
    return 0;
}

/**
 * @brief 分块打乱数据
 * 
 * 主流程：
 * 1. 从 stdin 读取 CREC 记录到缓冲区
 * 2. 缓冲区满时打乱并写入临时文件
 * 3. 最后合并所有临时文件
 */
int shuffle_by_chunks() {
    std::cerr << "SHUFFLING COOCCURRENCES\n";
    if (verbose > 0) {
        std::cerr << "Array size: " << array_size << " records ("
                  << (array_size * sizeof(CREC) / (1024.0 * 1024.0)) << " MB)\n";
    }

    std::vector<CREC> array(array_size);
    long long i = 0;           // 当前缓冲区中的记录数
    long long total = 0;       // 总处理记录数
    int file_counter = 0;      // 临时文件计数

    // 打开第一个临时文件
    std::ostringstream oss;
    oss << file_head << "_" << std::setw(4) << std::setfill('0') << file_counter << ".bin";
    std::ofstream fout(oss.str(), std::ios::binary);
    if (!fout.is_open()) {
        log_file_loading_error("temp file", oss.str().c_str());
        return 1;
    }

    if (verbose > 1) {
        std::cerr << "Shuffling by chunks: processed 0 lines.";
    }

    // 从 stdin 读取二进制 CREC
    while (std::cin.read(reinterpret_cast<char*>(&array[i]), sizeof(CREC))) {
        i++;

        // 缓冲区满，打乱并写入临时文件
        if (i >= array_size) {
            shuffle_array(array, i);
            write_chunk(array, i, fout);
            total += i;

            if (verbose > 1) {
                std::cerr << "\033[22Gprocessed " << total << " lines.";
            }

            fout.close();
            file_counter++;
            
            // 打开下一个临时文件
            oss.str("");
            oss << file_head << "_" << std::setw(4) << std::setfill('0') << file_counter << ".bin";
            fout.open(oss.str(), std::ios::binary);
            if (!fout.is_open()) {
                log_file_loading_error("temp file", oss.str().c_str());
                return 1;
            }
            i = 0;
        }
    }

    // 处理最后一批（可能不满）
    if (i > 0) {
        shuffle_array(array, i);
        write_chunk(array, i, fout);
        total += i;
    }
    fout.close();

    if (verbose > 1) {
        std::cerr << "\033[22Gprocessed " << total << " lines.\n";
        std::cerr << "Wrote " << (file_counter + 1) << " temporary file(s).\n";
    }

    // 合并所有临时文件
    return shuffle_merge(file_counter + 1);
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
              << "  -s, --seed <int>      随机种子, 默认 0 (使用时间)\n"
              << "  -t, --temp-file <str> 临时文件前缀, 默认 temp_shuffle\n";
}

int main(int argc, char** argv) {
    static struct option long_options[] = {
        {"help",      no_argument,       nullptr, 'h'},
        {"verbose",   required_argument, nullptr, 'v'},
        {"memory",    required_argument, nullptr, 'm'},
        {"seed",      required_argument, nullptr, 's'},
        {"temp-file", required_argument, nullptr, 't'},
        {nullptr, 0, nullptr, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "hv:m:s:t:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'h': print_usage(argv[0]); return 0;
            case 'v': verbose = std::atoi(optarg); break;
            case 'm': memory_limit = std::atof(optarg); break;
            case 's': seed = std::atoi(optarg); break;
            case 't': file_head = optarg; break;
            default: print_usage(argv[0]); return 1;
        }
    }
    
    // 初始化随机种子
    if (seed == 0) {
        seed = static_cast<int>(std::time(nullptr));
    }
    if (verbose > 0) {
        std::cerr << "Using random seed " << seed << "\n";
    }
    rng.seed(seed);
    
    // 计算数组大小：95% 内存用于缓冲区
    array_size = static_cast<long long>(0.95 * memory_limit * 1024 * 1024 * 1024 / sizeof(CREC));
    
    return shuffle_by_chunks();
}
