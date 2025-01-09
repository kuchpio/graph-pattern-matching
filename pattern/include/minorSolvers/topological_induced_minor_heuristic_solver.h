#pragma once

#include "topological_minor_heuristic.h"
#include "vf2_induced_subgraph_solver.hpp"

namespace pattern
{
class InducedTopologicalMinorHeuristicSolver : public TopologicalMinorHeuristic {
  public:
    InducedTopologicalMinorHeuristicSolver() : TopologicalMinorHeuristic(nullptr) {
        subgraphMatcher_ = new Vf2InducedSubgraphSolver();
    };
    ~InducedTopologicalMinorHeuristicSolver() override {
        free(subgraphMatcher_);
    }
};
} // namespace pattern