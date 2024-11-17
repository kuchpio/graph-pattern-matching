#pragma once

#include "core.h"
#include "isomorphism_matcher.h"
#include "argraph.h"
#include <optional>
#include <tuple>

namespace pattern
{
class Vf2IsomorphismSolver : public IsomorphismMatcher {
  public:
    bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);
    std::optional<std::vector<vertex>> matching(const core::Graph& bigGraph, const core::Graph& smallGraph);

  private:
    Graph convert_graph(const core::Graph& G);
};
} // namespace pattern