#include <iostream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include "json.hpp"
#include "core.h"

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

std::pair<core::Graph, std::vector<std::pair<vertex, std::pair<float, float>>>> generateGraphFromImage(std::string& imagePath, int vertexCount) {
    std::string command = "python edge_detection/graph.py " + imagePath + " " + std::to_string(vertexCount);

    std::string output = exec(command.c_str());

    json graph_data = json::parse(output);

    std::vector<std::tuple<vertex, vertex>> edges;
    for (const auto& edge : graph_data["edges"]) {
        edges.emplace_back(edge["source"], edge["target"]);
    }
    core::Graph graph(edges);

    std::vector<std::pair<vertex, std::pair<float, float>>> vertex_positions;
    for (const auto& node : graph_data["nodes"]) {
        int id = node["id"];
        auto pos = node["pos"];
        vertex_positions.emplace_back(id, std::make_pair(pos[0], pos[1]));
    }

    return {graph, vertex_positions};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <imagePath> <vertexCount>" << std::endl;
        return 1;
    }

    std::string imagePath = argv[1];
    int vertexCount = atoi(argv[2]);

    try {
        auto [graph, vertex_positions] = generateGraphFromImage(imagePath, vertexCount);

        std::cout << "Graph created with " << graph.size() << " vertices and " << graph.edge_count() << " edges.\n";

        std::cout << "Edges:\n";
        for (const auto& edge : graph.edges()) {
            std::cout << std::get<0>(edge) << " -- " << std::get<1>(edge) << "\n";
        }

        std::cout << "Vertex positions:\n";
        for (const auto& [id, pos] : vertex_positions) {
            std::cout << "Vertex " << id << ": x = " << pos.first << ", y = " << pos.second << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "An error occurred: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
