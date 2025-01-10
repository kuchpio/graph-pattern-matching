#pragma once

#include "topological_minor_heuristic.h"
#include "vf2_subgraph_solver.hpp"

namespace pattern
{
class TopologicalMinorHeuristicSolver : public TopologicalMinorHeuristic {
  public:
    TopologicalMinorHeuristicSolver(bool direct = false) : TopologicalMinorHeuristic(nullptr, direct) {
        subgraphMatcher_ = new Vf2SubgraphSolver();
    };
    ~TopologicalMinorHeuristicSolver() override {
        free(subgraphMatcher_);
    }
};
} // namespace pattern