#include <array>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace isomorphism {

static std::string exec(const char* cmd) {
    std::array<char, 4096> buffer{};
    std::string result;

    FILE* pipe = _popen(cmd, "r");
    if (!pipe) {
        throw std::runtime_error("_popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    _pclose(pipe);

    return result;
}

bool isomorphic(const std::vector<std::pair<int, int>>& edges1,
                const std::vector<std::pair<int, int>>& edges2) {
    std::string command = "python check_if_isomorphism.py ";
    command += std::to_string(edges1.size());
    for (const auto& edge : edges1) {
        command += " " + std::to_string(edge.first) + " " + std::to_string(edge.second);
    }

    command += " " + std::to_string(edges2.size());
    for (const auto& edge : edges2) {
        command += " " + std::to_string(edge.first) + " " + std::to_string(edge.second);
    }

    std::string output = exec(command.c_str());
    int result = std::stoi(output);
    return result == 1;
}

}

int main() {
    std::vector<std::pair<int, int>> edges1 = {{0, 1}, {1, 2}, {2, 0}};
    std::vector<std::pair<int, int>> edges2 = {{1, 2}, {2, 0}, {0, 1}};

    try {
        bool result = isomorphism::isomorphic(edges1, edges2);
        std::cout << "Are they isomorphic? " << (result ? "Yes" : "No") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
