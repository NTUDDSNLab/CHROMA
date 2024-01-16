#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

int main() {
    std::ifstream inFile("facebook.txt"); // 原始文件
    std::ofstream outFile("facebook2.txt"); // 修改后的文件

    if (!inFile.is_open()) {
        std::cerr << "无法打开原始文件" << std::endl;
        return 1;
    }

    if (!outFile.is_open()) {
        std::cerr << "无法创建输出文件" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        if (line.size() > 0 && line[0] == 'e') {
            std::istringstream iss(line);
            char e;
            int num1, num2;
            if (iss >> e >> num1 >> num2) {
                outFile << e << " " << num1 + 1 << " " << num2 + 1 << std::endl;
            }
        } else {
            outFile << line << std::endl; // 其他行直接复制
        }
    }

    inFile.close();
    outFile.close();

    return 0;
}