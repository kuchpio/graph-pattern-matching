#include "image.h"
#include "nlohmann/json.hpp"
#include "core.h"
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <string>
#include <array>

#ifndef EDGE_DETECTION_DIR
#define EDGE_DETECTION_DIR "./"
#endif

#ifdef __linux__
#define _popen(cmd, mode) popen(cmd, mode)
#define _pclose(pipe) pclose(pipe)
#endif

namespace image
{

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

std::pair<core::Graph, std::vector<std::pair<float, float>>> grapherize(const std::string& imagePath, int vertexCount) {
    if (vertexCount <= 0) {
        throw std::runtime_error("Vertex count must be greater than 0.");
    }

    std::string scriptPath = std::string(EDGE_DETECTION_DIR) + "/graph.py";

    if (!std::filesystem::exists(scriptPath)) {
        throw std::runtime_error("Python script not found: " + scriptPath);
    }

    if (!std::filesystem::exists(imagePath)) {
        throw std::runtime_error("Image file not found: " + imagePath);
    }

    std::string command = "python " + scriptPath + " " + imagePath + " " + std::to_string(vertexCount);

    std::string output = exec(command.c_str());

    nlohmann::json graphData = nlohmann::json::parse(output);

    std::vector<std::tuple<vertex, vertex>> edges;
    for (const auto& edge : graphData["edges"]) {
        edges.emplace_back(edge["source"], edge["target"]);
        edges.emplace_back(edge["target"], edge["source"]);
    }
    core::Graph graph(edges);

    std::vector<std::pair<float, float>> vertexPositions(vertexCount);
    for (const auto& node : graphData["nodes"]) {
        const auto& pos = node["pos"];
        vertexPositions[node["id"]] = std::make_pair(pos[0], pos[1]);
    }

    return {graph, vertexPositions};
}

} // namespace image
