#include "generateSamples.h"
#include "core.h"
#include <filesystem>
#include <fstream>

#define SEED 2000

namespace utils
{

void EfficiencyTests::generateSamples(int count) {
    srand(SEED);
    generateSubgraphSamples(count);
    generateSubgraphSamples(count, true);
    // generateCudaSubgraphSamples(3);

    generateMinorSamples(30);
    generateInducedMinorSamples(20);

    generateTopologicalMinorSamples(20);
}

void EfficiencyTests::generateCudaSubgraphSamples(int count) {
    const std::size_t bigGraphSize = 100;

    const auto bigGraph = GraphFactory::random_connected_graph(bigGraphSize, 0.02);
    searchGraphs_.emplace("cuda_subgraph", bigGraph);

    std::size_t subgraphSize = 5;
    const std::size_t delta = subgraphSize;

    std::vector<core::Graph> subgraphs;
    for (int i = 0; i < count; i++) {
        auto subgraph = GraphFactory::random_connected_subgraph(bigGraph, subgraphSize);
        subgraphs.push_back(subgraph);
        subgraphSize += delta;
    }
    patternGraphs_.emplace("cuda_subgraph", subgraphs);
}

void EfficiencyTests::generateSubgraphSamples(int count, bool induced) {
    const std::size_t bigGraphSize = 100;

    const auto bigGraph = GraphFactory::random_connected_graph(bigGraphSize, 0.2);
    if (induced)
        searchGraphs_.emplace("induced_subgraph", bigGraph);
    else
        searchGraphs_.emplace("subgraph", bigGraph);

    std::size_t subgraphSize = bigGraphSize / count;
    const std::size_t delta = subgraphSize;

    std::vector<core::Graph> subgraphs;
    for (int i = 0; i < count; i++) {
        auto subgraph = GraphFactory::random_connected_subgraph(bigGraph, subgraphSize, induced);
        subgraphs.push_back(subgraph);
        subgraphSize += delta;
    }

    if (induced)
        patternGraphs_.emplace("induced_subgraph", subgraphs);
    else
        patternGraphs_.emplace("subgraph", subgraphs);
}

void EfficiencyTests::generateInducedMinorSamples(int count) {
    const std::size_t bigGraphSize = 20;
    const auto bigGraph = GraphFactory::random_connected_graph(bigGraphSize, 0.05);
    searchGraphs_.emplace("induced_minor", bigGraph);

    std::size_t minorSize = bigGraphSize / count;
    const std::size_t delta = minorSize;

    std::vector<core::Graph> minors;
    for (int i = 0; i < count; i++) {
        minors.push_back(GraphFactory::random_induced_minor(bigGraph, minorSize));
        minorSize += delta;
    }
    patternGraphs_.emplace("induced_minor", minors);
}

void EfficiencyTests::generateMinorSamples(int count) {

    const std::size_t bigGraphSize = 30;
    const auto bigGraph = GraphFactory::random_connected_graph(bigGraphSize, 0.05);
    searchGraphs_.emplace("minor", bigGraph);

    std::size_t minorSize = bigGraphSize / count;
    const std::size_t delta = minorSize;

    std::vector<core::Graph> minors;
    for (int i = 0; i < count; i++) {
        minors.push_back(GraphFactory::random_minor(bigGraph, minorSize));
        minorSize += delta;
    }
    patternGraphs_.emplace("minor", minors);
}

void EfficiencyTests::generateTopologicalMinorSamples(int count) {
    const std::size_t startingMinorSize = 3;
    const std::size_t delta = 1;

    auto minor = GraphFactory::random_connected_graph(startingMinorSize);
    std::vector<core::Graph> topolgicalMinors;
    for (int i = 0; i < count; i++) {
        topolgicalMinors.push_back(minor);
        minor = GraphFactory::random_edge_subdivisions(minor, delta);
    }
    searchGraphs_.emplace("topologicalMinor", minor);
    patternGraphs_["topologicalMinor"] = topolgicalMinors;
}

void EfficiencyTests::run() {
    generateSamples(100);
    createDirectory(path_);
    for (const auto& pattern : searchGraphs_) {
        testMatching(pattern.first);
    }
}

void EfficiencyTests::testMatching(const std::string& pattern) {

    auto bigGraph = searchGraphs_.at(pattern);
    auto smallGraphs = patternGraphs_.at(pattern);
    auto matcher = matchingAlgorithms_.at(pattern);
    const std::string filepath(path_ + '/' + pattern);
    std::ofstream outFile(filepath, std::ios::app);
    if (!outFile) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    int index = 0;
    for (const auto& smallGraph : smallGraphs) {
        auto start = std::chrono::high_resolution_clock::now();
        auto matching = matcher.get()->match(bigGraph, smallGraph);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        printf("Processed %d %s\n", index++, pattern.c_str());

        if (matching)
            outFile << "success; ";
        else
            outFile << "fail; ";
        outFile << elapsed.count() << " seconds; ";
        outFile << "big=" << bigGraph.size() << "; ";
        outFile << "small=" << smallGraph.size() << ";  \n";
    }
    outFile.close();
}

void EfficiencyTests::saveGraph(const core::Graph& G, const std::string& path, const std::string& baseName, int index) {
    std::string fileName = path + "/" + baseName + "_" + std::to_string(index) + ".g6";
    auto serializedGraph = core::Graph6Serializer::Serialize(G);

    std::ofstream outFile(fileName, std::ios::out);
    if (!outFile) {
        throw std::runtime_error("Failed to open file: " + fileName);
    }
    outFile << serializedGraph;
    outFile.close();
}

void EfficiencyTests::createDirectory(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        if (!std::filesystem::create_directories(path)) {
            throw std::runtime_error("Failed to create directory: " + path);
        }
    }
}
} // namespace utils