#pragma once

#include "core.h"
#include "pattern.h"
#include <unordered_map>

namespace pattern
{
class SubgraphMatcher : public PatternMatcher {
  public:
    bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) override;

  private:
    bool sub_isomorphism_recursion(const core::Graph& bigGraph, const core::Graph& smallGraph,
                                   std::unordered_map<int, int>& small_big_mapping,
                                   std::unordered_map<int, int>& big_small_mapping, int v);
    bool can_match_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph,
                               const std::unordered_map<int, int>& mapping_big_small,
                               const std::unordered_map<int, int>& mapping_small_big, int v, int big_v);
    int find_first_unmapped(const core::Graph& G, std::unordered_map<int, int> map);
};
} // namespace pattern