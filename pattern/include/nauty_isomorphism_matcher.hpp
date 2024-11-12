#pragma once

#include "core.h"
#include "pattern.h"
#include "nauty_traces.h"

namespace pattern
{
class NautyIsomorphismMatcher : public PatternMatcher {
  public:
    bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);

  private:
    NTSparseGraph convert_graph(const core::Graph& G);
};
} // namespace pattern