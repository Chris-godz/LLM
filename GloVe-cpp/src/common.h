#ifndef COMMON_H
#define COMMON_H

/**
 * @file common.h
 * @brief 公共定义和工具函数
 * 
 * 对应原始 GloVe: src/common.h
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// ============================================================================
// 常量定义
// ============================================================================

constexpr int MAX_STRING_LENGTH = 1000;
constexpr size_t TSIZE = 1048576;      // 哈希表大小 2^20
constexpr unsigned int SEED = 1159241; // 哈希种子

// ============================================================================
// 类型定义
// ============================================================================

using real = double;

/**
 * @brief 共现记录
 */
struct CREC {
    int word1;
    int word2;
    real val;
};

/**
 * @brief 哈希表记录
 */
struct HASHREC {
    std::string word;
    long long num;  // count 或 id
};

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 位运算哈希函数
 * 
 * 参考: GloVe/src/common.c 中的 bitwisehash
 */
inline unsigned int bitwisehash(const std::string& word, size_t tsize, unsigned int seed) {
    unsigned int h = seed;
    for (char c : word) {
        h ^= ((h << 5) + static_cast<unsigned char>(c) + (h >> 2));
    }
    return (h & 0x7fffffff) % tsize;
}

/**
 * @brief 从输入流读取一个单词
 * @return 如果遇到换行或 EOF 返回 1，否则返回 0
 * 
 * TODO: 实现这个函数
 * 参考: GloVe/src/common.c 中的 get_word
 */
inline int get_word(std::string& word, std::istream& fin) {
    word.clear();
    int ch;
    
    while (true) {
        ch = fin.get();
        if (ch == EOF) return 1;
        if (ch == '\r') continue;
        if (ch == '\n') return 1;
        if (ch != ' ' && ch != '\t') break;
    }
    
    while (true) {
        if (word.size() < MAX_STRING_LENGTH - 1) {
            word += static_cast<char>(ch);
        }
        ch = fin.get();
        if (ch == EOF || ch == ' ' || ch == '\t') break;
        if (ch == '\n') {
            fin.unget();
            break;
        }
        if (ch == '\r') continue;
    }
    
    return 0;
}

/**
 * @brief 打印文件加载错误
 */
inline void log_file_loading_error(const char* desc, const char* filename) {
    std::cerr << "Error: Unable to open " << desc << " " << filename << std::endl;
}

#endif // COMMON_H
