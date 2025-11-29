cmake_minimum_required(VERSION 3.14)
project(GloVe-cpp VERSION 1.0.0 LANGUAGES CXX)

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 查找 pthread
find_package(Threads REQUIRED)

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/src)

# 可执行程序（和原始 GloVe 结构一致）
add_executable(vocab_count src/vocab_count.cpp)

add_executable(cooccur src/cooccur.cpp)

add_executable(shuffle src/shuffle.cpp)

add_executable(glove src/glove.cpp)
target_link_libraries(glove Threads::Threads)

# 打印配置信息
message(STATUS "=== GloVe-cpp ===")
