#pragma once

#include "core.h"
#include "isomorphism_matcher.h"
#include "argraph.h"
#include <optional>

namespace pattern
{
class Vf2IsomorphismSolver : public IsomorphismMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph);

  private:
    Graph convert_graph(const core::Graph& G);
};
} // namespace pattern