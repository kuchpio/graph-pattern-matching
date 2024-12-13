#pragma once

#include "minor_matcher.h"
#include "vf2_subgraph_solver.hpp"
namespace pattern
{
class TopologicalMinorHeuristic : public MinorMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& Q) override;

  private:
    void topologicalMinorRecursion();
    Vf2SubgraphSolver subgraphSolver = Vf2SubgraphSolver();
};
} // namespace pattern