#include <gtest/gtest.h>

// 主测试入口
// gtest_main 会自动提供 main 函数

// 添加全局测试配置
class GlobalEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        // 全局测试初始化
    }

    void TearDown() override {
        // 全局测试清理
    }
};

// 注册全局环境
::testing::Environment* const global_env = 
    ::testing::AddGlobalTestEnvironment(new GlobalEnvironment);
