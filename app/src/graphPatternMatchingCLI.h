#pragma once
#include <string>
#include "core.h"
#include "pattern.h"
#include "vf2_subgraph_solver.hpp"
#include "vf2_induced_subgraph_solver.hpp"
#include "topological_minor_heuristic_solver.h"
#include "miner_minor_matcher.hpp"
#include "induced_minor_heuristic.h"
#include "topological_induced_minor_heuristic_solver.h"
#include <map>
#include <CLI/CLI.hpp>

struct CLIGraph {
  public:
    std::string filepath;
    core::Graph graph;
};

class GraphPatternMatchingCLI {
  public:
    GraphPatternMatchingCLI();
    void parse(int argc, char** argv);
    int exit(const CLI::ParseError& error);
    void run() const;

  private:
    CLI::App app_{"Graph Pattern Matching"};
    bool induced_ = false;
    std::string pattern_ = "subgraph";
    std::string input1_{""};
    std::string input2_{""};

    std::map<std::string, std::shared_ptr<pattern::PatternMatcher>> matchingAlgorithms_ = {
        {"subgraph", std::make_shared<pattern::Vf2SubgraphSolver>()},
        {"minor", std::make_shared<pattern::MinerMinorMatcher>()},
        {"topologicalMinor", std::make_shared<pattern::TopologicalMinorHeuristicSolver>()},
    };

    std::map<std::string, std::shared_ptr<pattern::PatternMatcher>> inducedMatchingAlgorithms_ = {
        {"subgraph", std::make_shared<pattern::Vf2InducedSubgraphSolver>()},
        {"minor", std::make_shared<pattern::InducedMinorHeuristic>()},
        {"topologicalMinor", std::make_shared<pattern::InducedTopologicalMinorHeuristicSolver>()},
    };

    void addMainCommandOptions();
    static void printMatching(const std::vector<vertex>& matching);
    static core::Graph loadGraph(const std::string& filepath);
};
