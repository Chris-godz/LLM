/**
 * @file cooccur.cpp
 * @brief 共现矩阵统计工具
 * 
 * 对应原始 GloVe: src/cooccur.c
 * 
 * 改进版实现：
 * 1. 使用内存缓冲区存储共现记录
 * 2. 当缓冲区满时，进行排序并写入临时文件（外部排序）
 * 3. 最后合并所有临时文件到标准输出
 * 
 * 相比之前的 unordered_map 实现，这种方法可以处理任意大小的语料库，
 * 只要磁盘空间足够，且严格遵守内存限制。
 */

#include "common.h"
#include <getopt.h>
#include <cstdint>
#include <queue>
#include <sstream>
#include <iomanip>

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
// 辅助结构与函数
// ============================================================================

/**
 * @brief 用于合并的优先队列元素
 */
struct MergeEntry {
    CREC rec;
    int file_id;

    // 最小堆比较：我们希望 word1 小的在堆顶；word1 相同则 word2 小的在堆顶
    // priority_queue 默认是最大堆，使用 std::greater 时需要 operator>
    bool operator>(const MergeEntry& other) const {
        if (rec.word1 != other.rec.word1) return rec.word1 > other.rec.word1;
        return rec.word2 > other.rec.word2;
    }
};

/**
 * @brief CREC 排序比较函数
 */
bool compare_crec(const CREC& a, const CREC& b) {
    if (a.word1 != b.word1) return a.word1 < b.word1;
    return a.word2 < b.word2;
}

/**
 * @brief 将缓冲区排序、去重并写入临时文件
 */
void write_chunk(std::vector<CREC>& buffer, int chunk_id) {
    if (buffer.empty()) return;

    if (verbose > 1) std::cerr << "Sorting and writing chunk " << chunk_id << " (" << buffer.size() << " entries)...\n";

    // 1. 排序
    std::sort(buffer.begin(), buffer.end(), compare_crec);

    // 2. 打开临时文件
    std::ostringstream oss;
    oss << "cooccurrence_tmp_" << std::setw(4) << std::setfill('0') << chunk_id << ".bin";
    std::string filename = oss.str();
    
    std::ofstream fout(filename, std::ios::binary);
    if (!fout.is_open()) {
        log_file_loading_error("temp file", filename.c_str());
        return;
    }

    // 3. 去重并写入
    CREC old = buffer[0];
    for (size_t i = 1; i < buffer.size(); ++i) {
        if (buffer[i].word1 == old.word1 && buffer[i].word2 == old.word2) {
            old.val += buffer[i].val;
        } else {
            fout.write(reinterpret_cast<char*>(&old), sizeof(CREC));
            old = buffer[i];
        }
    }
    // 写入最后一个
    fout.write(reinterpret_cast<char*>(&old), sizeof(CREC));
    
    fout.close();
    buffer.clear(); // 清空缓冲区
}

/**
 * @brief 合并所有临时文件到标准输出
 */
int merge_files(int num_chunks) {
    if (verbose > 1) std::cerr << "Merging " << num_chunks << " temporary files...\n";

    std::vector<std::ifstream> files(num_chunks);
    std::priority_queue<MergeEntry, std::vector<MergeEntry>, std::greater<MergeEntry>> pq;

    // 打开所有文件并读取第一个元素
    for (int i = 0; i < num_chunks; ++i) {
        std::ostringstream oss;
        oss << "cooccurrence_tmp_" << std::setw(4) << std::setfill('0') << i << ".bin";
        std::string filename = oss.str();

        files[i].open(filename, std::ios::binary);
        if (!files[i].is_open()) {
            log_file_loading_error("temp file for merge", filename.c_str());
            return 1;
        }

        CREC rec;
        if (files[i].read(reinterpret_cast<char*>(&rec), sizeof(CREC))) {
            pq.push({rec, i});
        }
    }

    // 多路归并
    long long total_written = 0;
    if (!pq.empty()) {
        MergeEntry current = pq.top();
        pq.pop();

        // 尝试从该文件读取下一个
        CREC next_rec;
        if (files[current.file_id].read(reinterpret_cast<char*>(&next_rec), sizeof(CREC))) {
            pq.push({next_rec, current.file_id});
        }

        while (!pq.empty()) {
            MergeEntry next_entry = pq.top();
            
            // 检查是否是同一个词对（来自不同文件）
            if (next_entry.rec.word1 == current.rec.word1 && next_entry.rec.word2 == current.rec.word2) {
                // 合并值
                current.rec.val += next_entry.rec.val;
                pq.pop();
                
                // 补充
                if (files[next_entry.file_id].read(reinterpret_cast<char*>(&next_rec), sizeof(CREC))) {
                    pq.push({next_rec, next_entry.file_id});
                }
            } else {
                // 写入当前记录
                std::cout.write(reinterpret_cast<char*>(&current.rec), sizeof(CREC));
                total_written++;
                
                // 更新当前记录
                current = next_entry;
                pq.pop();
                
                // 补充
                if (files[next_entry.file_id].read(reinterpret_cast<char*>(&next_rec), sizeof(CREC))) {
                    pq.push({next_rec, next_entry.file_id});
                }
            }
        }
        // 写入最后一个
        std::cout.write(reinterpret_cast<char*>(&current.rec), sizeof(CREC));
        total_written++;
    }

    std::cout.flush();
    if (verbose > 0) std::cerr << "Wrote " << total_written << " records.\n";

    // 清理临时文件
    for (int i = 0; i < num_chunks; ++i) {
        files[i].close();
        std::ostringstream oss;
        oss << "cooccurrence_tmp_" << std::setw(4) << std::setfill('0') << i << ".bin";
        std::remove(oss.str().c_str());
    }

    return 0;
}

// ============================================================================
// 核心函数
// ============================================================================

/**
 * @brief 加载词汇表
 */
std::unordered_map<std::string, long long> load_vocab() {
    std::unordered_map<std::string, long long> vocab;
    std::ifstream fin(vocab_file);
    if (!fin.is_open()) {
        log_file_loading_error("vocabulary file", vocab_file.c_str());
        return vocab;
    }

    std::string word;
    long long count;
    long long id = 1;  // 从 1 开始，与原版 GloVe 保持一致 (1-indexed)
    while (fin >> word >> count) {
        if (vocab.find(word) == vocab.end()) {
            vocab[word] = id++;
        }
    }
    if (verbose > 0) std::cerr << "Loaded vocab with " << vocab.size() << " words.\n";
    return vocab;
}

/**
 * @brief 构建共现矩阵
 */
int get_cooccurrence() {
    // 1. 加载词表
    auto vocab = load_vocab();
    if (vocab.empty()) return 1;

    // 2. 初始化缓冲区
    // 预留 80% 的内存限制给缓冲区，剩余给词表和其他开销
    long long max_buffer_size = static_cast<long long>(memory_limit * 1024 * 1024 * 1024 * 0.8) / sizeof(CREC);
    std::vector<CREC> buffer;
    buffer.reserve(max_buffer_size);
    
    if (verbose > 0) {
        std::cerr << "Buffer size: " << max_buffer_size << " records (" 
                  << (max_buffer_size * sizeof(CREC) / (1024.0*1024.0)) << " MB)\n";
    }

    // 3. 处理语料
    std::vector<int> sentence;
    sentence.reserve(1024);
    std::string token;
    long long total_tokens = 0;
    int chunk_counter = 0;

    while (!std::cin.eof()) {
        int is_newline = get_word(token, std::cin);

        if (!is_newline && !token.empty()) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                sentence.push_back(static_cast<int>(it->second));
                total_tokens++;
            }
        }

        if (is_newline || std::cin.eof()) {
            // 处理句子
            // 原版 GloVe 的逻辑：对于每个词 w2，遍历它左侧窗口内的所有词 w1
            // 记录 (w1, w2)，如果 symmetric 则同时记录 (w2, w1)
            const int n = static_cast<int>(sentence.size());
            for (int j = 0; j < n; ++j) {
                int w2 = sentence[j];  // 当前词（目标词）
                
                // 遍历左侧窗口
                int left_bound = std::max(0, j - window_size);
                for (int k = j - 1; k >= left_bound; --k) {
                    int w1 = sentence[k];  // 上下文词（在左侧）
                    
                    // 距离加权
                    int dist = j - k;
                    real weight = (distance_weighting) ? 1.0 / dist : 1.0;

                    // 记录 (w1, w2): w1 在左边，w2 在右边
                    buffer.push_back({w1, w2, weight});
                    
                    // 如果 symmetric，也记录 (w2, w1)
                    // 这相当于也考虑了右侧上下文
                    if (symmetric) {
                        buffer.push_back({w2, w1, weight});
                    }

                    // 检查缓冲区是否已满
                    if (buffer.size() >= static_cast<size_t>(max_buffer_size)) {
                        write_chunk(buffer, chunk_counter++);
                    }
                }
            }
            sentence.clear();
        }
    }

    // 写入剩余数据
    if (!buffer.empty()) {
        write_chunk(buffer, chunk_counter++);
    }

    if (verbose > 0) std::cerr << "Processed " << total_tokens << " tokens.\n";

    // 4. 合并文件
    return merge_files(chunk_counter);
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
            case 'h': print_usage(argv[0]); return 0;
            case 'v': verbose = std::atoi(optarg); break;
            case 'V': vocab_file = optarg; break;
            case 'w': window_size = std::atoi(optarg); break;
            case 's': symmetric = std::atoi(optarg); break;
            case 'd': distance_weighting = std::atoi(optarg); break;
            case 'm': memory_limit = std::atof(optarg); break;
            default: print_usage(argv[0]); return 1;
        }
    }
    
    if (vocab_file.empty()) {
        std::cerr << "Error: --vocab-file is required\n";
        print_usage(argv[0]);
        return 1;
    }
    
    return get_cooccurrence();
}
