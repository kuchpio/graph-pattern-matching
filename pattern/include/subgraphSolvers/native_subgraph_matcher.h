#pragma once

#include "core.h"
#include "pattern.h"
#include <unordered_map>
#include "subgraph_matcher.h"

namespace pattern
{
class NativeSubgraphMatcher : public SubgraphMatcher {
  public:
    bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) override;

  private:
    bool sub_isomorphism_recursion(const core::Graph& bigGraph, const core::Graph& smallGraph,
                                   std::unordered_map<vertex, vertex>& small_big_mapping,
                                   std::unordered_map<vertex, vertex>& big_small_mapping, vertex v);
    bool can_match_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph,
                               const std::unordered_map<vertex, vertex>& mapping_big_small,
                               const std::unordered_map<vertex, vertex>& mapping_small_big, vertex v, vertex big_v);
    vertex find_first_unmapped(const core::Graph& G, std::unordered_map<vertex, vertex> map);
};
} // namespace pattern