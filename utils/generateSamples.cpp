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
    generateCudaSubgraphSamples(55);

    generateMinorSamples(30);
    generateInducedMinorSamples(20);

    generateTopologicalMinorSamples(20);
}

void EfficiencyTests::generateCudaSubgraphSamples(int count) {
    std::size_t bigGraphSize = 1000;
    const std::size_t bigDelta = 100;

    std::size_t subgraphSize = 10;
    cudaSmallGraph_ = GraphFactory::random_connected_graph(subgraphSize, 0.8);

    std::vector<core::Graph> bigGraphs;
    for (int i = 0; i < count; i++) {
        const auto bigGraph = GraphFactory::random_bigger_graph(cudaSmallGraph_, bigGraphSize, 0.4);
        cudaBigGraphs_.push_back(bigGraph);
        bigGraphSize += bigDelta;
    }
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

void EfficiencyTests::testMatchings(const std::vector<std::string>& patterns, const std::string& directoryName,
                                    const std::vector<core::Graph>& bigGraphs,
                                    const std::vector<core::Graph>& smallGraphs) {
    std::string directoryPath = path_ + "/" + directoryName;
    createDirectory(directoryPath);

    for (const auto& pattern : patterns) {
        auto matcher = matchingAlgorithms_.at(pattern);
        const std::string& baseFilename(directoryPath + "/" + pattern);
        if (smallGraphs.size() > 1) processMatching(baseFilename, matcher, bigGraphs, smallGraphs, true);
        if (bigGraphs.size() > 1) processMatching(baseFilename, matcher, smallGraphs, bigGraphs, false);
    }
}

void EfficiencyTests::run() {
    generateSamples(50);
    createDirectory(path_);
    for (const auto& pattern : searchGraphs_) {
        // testMatching(path_, pattern.first);
    }
    /*
    testMatchings(std::vector<std::string>{"cuda_subgraph", "subgraph"}, "subgraphGPU_CPU", cudaBigGraphs_,
                  std::vector<core::Graph>{cudaSmallGraph_});
    testMatchings(std::vector<std::string>{"subgraph", "induced_subgraph"}, "subgraphs",
                  std::vector<core::Graph>{searchGraphs_.at("induced_subgraph")}, patternGraphs_["induced_subgraph"]);

    /*testMatchings(std::vector<std::string>{"subgraph", "induced_subgraph", "cuda_subgraph"}, "subgraphs_all",
                  std::vector<core::Graph>{searchGraphs_["induced_subgraph"]},
       patternGraphs_["induced_subgraph"]);*/

    /* testMatchings(std::vector<std::string>{"minor", "induced_minor", "topologicalMinor"}, "minors",
                       std::vector<core::Graph>{searchGraphs_.at("induced_minor")}, patternGraphs_["induced_minor"]);
    */

    testMatchings(std::vector<std::string>{"induced_topologicalMinor", "topologicalMinor"}, "topologicalMinors",
                  std::vector<core::Graph>{searchGraphs_.at("topologicalMinor")}, patternGraphs_["topologicalMinor"]);
}

void EfficiencyTests::processMatching(const std::string& path, std::shared_ptr<pattern::PatternMatcher> matcher,
                                      const std::vector<core::Graph>& baseGraphs,
                                      const std::vector<core::Graph>& graphs, bool bigConst) {
    int index = 1;
    for (const auto& baseGraph : baseGraphs) {
        std::string whoChanges = bigConst ? "_bigConst_" : "_smallConst";
        const std::string& currentFileName(path + whoChanges + std::to_string(index++));
        std::ofstream outFile(currentFileName, std::ios::app);
        if (!outFile) {
            throw std::runtime_error("Failed to open file: " + currentFileName);
        }
        for (const auto& changingGraph : graphs) {
            auto start = std::chrono::high_resolution_clock::now();
            auto matching =
                bigConst ? matcher->match(baseGraph, changingGraph) : matcher->match(changingGraph, baseGraph);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            if (matching)
                outFile << "success; ";
            else
                outFile << "fail; ";
            outFile << elapsed.count() << " seconds; ";
            if (bigConst) {
                outFile << "big=" << baseGraph.size() << "; ";
                outFile << "small=" << changingGraph.size() << ";  \n";
            } else {
                outFile << "big=" << changingGraph.size() << "; ";
                outFile << "small=" << baseGraph.size() << ";  \n";
            }
        }
        outFile.close();
    }
}

void EfficiencyTests::testMatching(const std::string& path, const std::string& pattern) {
    auto bigGraph = searchGraphs_.at(pattern);
    auto smallGraphs = patternGraphs_.at(pattern);
    auto matcher = matchingAlgorithms_.at(pattern);
    const std::string filepath(path + '/' + pattern);
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

void EfficiencyTests::createDirectory(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        if (!std::filesystem::create_directories(path)) {
            throw std::runtime_error("Failed to create directory: " + path);
        }
    }
}
} // namespace utils