#pragma once
#include "utils.h"
#include <map>
#include "pattern.h"
#include "solvers.h"

namespace utils
{

class EfficiencyTests {
  public:
    void generateSamples(int count);
    void run();

  private:
    std::map<std::string, core::Graph> searchGraphs_;
    std::map<std::string, std::vector<core::Graph>> patternGraphs_;

    std::vector<core::Graph> cudaBigGraphs_;
    core::Graph cudaSmallGraph_ = core::Graph(0);

    std::map<std::string, std::shared_ptr<pattern::PatternMatcher>> matchingAlgorithms_ = {
        {"subgraph", std::make_shared<pattern::Vf2SubgraphSolver>()},
        {"minor", std::make_shared<pattern::MinerMinorMatcher>()},
        {"topologicalMinor", std::make_shared<pattern::TopologicalMinorHeuristicSolver>(true)},
        {"induced_subgraph", std::make_shared<pattern::Vf2InducedSubgraphSolver>()},
        {"induced_minor", std::make_shared<pattern::InducedMinorHeuristic>(true)},
        {"induced_topologicalMinor", std::make_shared<pattern::InducedTopologicalMinorHeuristicSolver>(true)},
        {"cuda_subgraph", std::make_shared<pattern::CudaSubgraphMatcher>()}};
    const std::string path_ = "./EfficiencyTests";

    void generateSubgraphSamples(int count, bool induced = false);
    void generateMinorSamples(int count);
    void generateInducedMinorSamples(int count);
    void generateTopologicalMinorSamples(int count);
    void testMatching(const std::string& path, const std::string& pattern);
    void generateCudaSubgraphSamples(int count);

    void testMatchings(const std::vector<std::string>& patterns, const std::string& directoryPath,
                       const std::vector<core::Graph>& bigGraphs, const std::vector<core::Graph>& smallGraphs);
    void processMatching(const std::string& path, std::shared_ptr<pattern::PatternMatcher> matcher,
                         const std::vector<core::Graph>& baseGraph, const std::vector<core::Graph>& graphs,
                         bool bigConst = true);

    void createDirectory(const std::string& path);
};
} // namespace utils