#include <gtest/gtest.h>
#include "vocabulary.hpp"
#include <fstream>
#include <sstream>

using namespace word2vec;

// 测试 Vocabulary 基本功能
class VocabularyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建临时测试文件
        std::ofstream file("test_vocab.txt");
        file << "hello world\n";
        file << "hello test\n";
        file << "world test hello\n";
        file.close();
    }

    void TearDown() override {
        std::remove("test_vocab.txt");
    }
};

TEST_F(VocabularyTest, BasicConstruction) {
    Vocabulary vocab;
    EXPECT_EQ(vocab.Size(), 0);
    EXPECT_EQ(vocab.TotalWords(), 0);
}

TEST_F(VocabularyTest, WordIndexLookup) {
    Vocabulary vocab;
    // TODO: 实现后取消注释
    // vocab.LearnFromFile("test_vocab.txt", 1);
    
    // EXPECT_GT(vocab.Size(), 0);
    // EXPECT_GE(vocab.GetWordIndex("hello"), 0);
    // EXPECT_EQ(vocab.GetWordIndex("nonexistent"), -1);
}

TEST_F(VocabularyTest, MinCountFiltering) {
    Vocabulary vocab;
    // TODO: 测试最小词频过滤
    // vocab.LearnFromFile("test_vocab.txt", 2);
    
    // 词频低于min_count的词应该被过滤
    // EXPECT_EQ(vocab.GetWordIndex("world"), -1);
}

TEST_F(VocabularyTest, SaveAndLoad) {
    Vocabulary vocab1;
    // TODO: 实现后测试保存和加载
    // vocab1.LearnFromFile("test_vocab.txt", 1);
    // vocab1.Save("test_vocab_saved.txt");
    
    // Vocabulary vocab2;
    // vocab2.Load("test_vocab_saved.txt");
    
    // EXPECT_EQ(vocab1.Size(), vocab2.Size());
    // std::remove("test_vocab_saved.txt");
}

// 测试 Huffman 树构建
TEST(VocabularyHuffmanTest, BuildHuffmanTree) {
    Vocabulary vocab;
    // TODO: 测试 Huffman 树构建
    // vocab.BuildHuffmanTree();
    
    // 验证高频词编码较短
}
