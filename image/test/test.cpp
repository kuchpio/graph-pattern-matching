#include <iostream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include "json.hpp"

using json = nlohmann::json;

std::string exec(const char* cmd) {
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <imagePath> <vertexCount>" << std::endl;
        return 1;
    }

    std::string imagePath = argv[1];
    int vertexCount = std::stoi(argv[2]);

    std::string command =
        "python "
        "edge_detection/graph.py " +
        imagePath + " " + std::to_string(vertexCount);

    try {
        std::string output = exec(command.c_str());

        json graph_data = json::parse(output);

        int num_nodes = graph_data["nodes"].size();
        for (const auto& edge : graph_data["edges"]) {
            int source = edge["source"];
            int target = edge["target"];
            std::cout << source << " -- " << target << std::endl;
        }
        for (const auto& vertex : graph_data["nodes"]) {
            int id = vertex["id"];
            auto pos = vertex["pos"];
            std::cout << id << " -- "
                      << "x: " << pos[0] << " y: " << pos[1] << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "An error occurred: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}