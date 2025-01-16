#pragma once

#include "core.h"
#include "find_embedding/graph.hpp"
#include "minor_matcher.h"
#include "vf2_subgraph_solver.hpp"
namespace pattern
{
class MinerMinorMatcher : public MinorMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& H) override;

  private:
    graph::input_graph convert_graph(const core::Graph& G);
    Vf2SubgraphSolver subgraphSolver_ = Vf2SubgraphSolver();
};

} // namespace pattern