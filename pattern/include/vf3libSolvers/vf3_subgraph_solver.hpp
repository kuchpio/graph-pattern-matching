#pragma once

#include "core.h"
#include "pattern.h"
#include "VFLib.h"

namespace pattern
{
class Vf3SubgraphSolver : public PatternMatcher {
  public:
    bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);

  private:
    vflib::ARGraph<vertex, int> convert_graph(const core::Graph& G);
};
} // namespace pattern