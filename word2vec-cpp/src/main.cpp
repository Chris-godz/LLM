#include <iostream>
#include <string>
#include <getopt.h>
#include "word2vec.hpp"

void PrintUsage(const char* prog_name) {
    std::cout << "Word2Vec - C++ Implementation\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " -t <file> -o <file> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -t, --train <file>      训练文件路径 (必需)\n";
    std::cout << "  -o, --output <file>     输出向量文件路径 (必需)\n";
    std::cout << "  -s, --size <int>        向量维度 (默认: 100)\n";
    std::cout << "  -w, --window <int>      窗口大小 (默认: 5)\n";
    std::cout << "  -n, --negative <int>    负采样数量 (默认: 5, 0=不使用)\n";
    std::cout << "  -m, --min-count <int>   最小词频 (默认: 5)\n";
    std::cout << "  -p, --threads <int>     线程数 (默认: 12)\n";
    std::cout << "  -i, --iter <int>        迭代次数 (默认: 5)\n";
    std::cout << "  -c, --cbow <0|1>        使用CBOW(1)或Skip-gram(0) (默认: 1)\n";
    std::cout << "  -a, --alpha <float>     学习率 (默认: skip-gram=0.025, CBOW=0.05)\n";
    std::cout << "  -e, --hs <0|1>          使用层次Softmax (默认: 0)\n";
    std::cout << "  -b, --binary <0|1>      二进制格式保存 (默认: 0)\n";
    std::cout << "  -S, --sample <float>    下采样阈值 (默认: 1e-3)\n";
    std::cout << "  -h, --help              显示帮助信息\n";
}

int main(int argc, char** argv) {
    word2vec::Trainer::Config config;
    
    static struct option long_options[] = {
        {"train",     required_argument, 0, 't'},
        {"output",    required_argument, 0, 'o'},
        {"size",      required_argument, 0, 's'},
        {"window",    required_argument, 0, 'w'},
        {"negative",  required_argument, 0, 'n'},
        {"min-count", required_argument, 0, 'm'},
        {"threads",   required_argument, 0, 'p'},
        {"iter",      required_argument, 0, 'i'},
        {"cbow",      required_argument, 0, 'c'},
        {"alpha",     required_argument, 0, 'a'},
        {"hs",        required_argument, 0, 'e'},
        {"binary",    required_argument, 0, 'b'},
        {"sample",    required_argument, 0, 'S'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    bool alpha_set = false;
    
    while ((opt = getopt_long(argc, argv, "t:o:s:w:n:m:p:i:c:a:e:b:S:h", 
                              long_options, &option_index)) != -1) {
        switch (opt) {
            case 't':
                config.train_file = optarg;
                break;
            case 'o':
                config.output_file = optarg;
                break;
            case 's':
                config.model_config.vector_size = std::stoi(optarg);
                break;
            case 'w':
                config.model_config.window = std::stoi(optarg);
                break;
            case 'n':
                config.model_config.negative = std::stoi(optarg);
                break;
            case 'm':
                config.model_config.min_count = std::stoi(optarg);
                break;
            case 'p':
                config.num_threads = std::stoi(optarg);
                break;
            case 'i':
                config.iterations = std::stoi(optarg);
                break;
            case 'c':
                config.model_config.use_cbow = (std::stoi(optarg) != 0);
                break;
            case 'a':
                config.model_config.alpha = std::stof(optarg);
                alpha_set = true;
                break;
            case 'e':
                config.model_config.hierarchical_softmax = (std::stoi(optarg) != 0);
                break;
            case 'b':
                config.model_config.binary = (std::stoi(optarg) != 0);
                break;
            case 'S':
                config.model_config.sample = std::stof(optarg);
                break;
            case 'h':
            default:
                PrintUsage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }
    
    // 如果使用CBOW且未手动设置alpha，使用0.05 (与C代码对齐)
    if (config.model_config.use_cbow && !alpha_set) {
        config.model_config.alpha = 0.05f;
    }
    
    // 验证必需参数
    if (config.train_file.empty() || config.output_file.empty()) {
        std::cerr << "Error: -train and -output are required\n";
        PrintUsage(argv[0]);
        return 1;
    }
    
    try {
        // 学习词汇表
        word2vec::Vocabulary vocab;
        vocab.LearnFromFile(config.train_file);
        
        // 训练模型
        word2vec::Trainer trainer(vocab, config);
        trainer.Train();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
