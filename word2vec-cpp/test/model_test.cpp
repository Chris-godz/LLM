#include <gtest/gtest.h>
#include "model.hpp"
#include "vocabulary.hpp"

using namespace word2vec;

class ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建简单的词汇表
        // vocab = std::make_unique<Vocabulary>();
    }
    
    // std::unique_ptr<Vocabulary> vocab;
};

TEST_F(ModelTest, InitExpTable) {
    // TODO: 测试 sigmoid 查找表初始化
    // 验证表的大小和边界值
}

TEST_F(ModelTest, InitUnigramTable) {
    // TODO: 测试负采样表初始化
    // 验证词频分布 P(w) = count(w)^0.75
}

TEST_F(ModelTest, VectorInitialization) {
    // TODO: 测试词向量初始化
    // 验证随机初始化范围
}

TEST_F(ModelTest, CBOWTraining) {
    // TODO: 测试 CBOW 训练
    // 验证前向和反向传播
}

TEST_F(ModelTest, SkipGramTraining) {
    // TODO: 测试 Skip-gram 训练
}

TEST_F(ModelTest, NegativeSampling) {
    // TODO: 测试负采样
    // 验证负样本选择
}

TEST_F(ModelTest, HierarchicalSoftmax) {
    // TODO: 测试 Hierarchical Softmax
    // 验证沿 Huffman 树路径的计算
}

TEST_F(ModelTest, SaveAndLoadVectors) {
    // TODO: 测试向量保存和加载
    // 测试文本和二进制格式
}

// 测试 Sigmoid 函数近似
TEST(SigmoidTest, LookupTable) {
    // TODO: 验证查找表的精度
    // 与真实 sigmoid 函数比较
}
