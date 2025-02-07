#pragma once

#include "core.h"
#include "isomorphism_matcher.h"
#include <optional>
#include <unordered_map>
#include <vector>
namespace pattern
{
class NativeIsomorphismMatcher : public IsomorphismMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph);

  private:
    std::optional<std::vector<vertex>> matchIsomorphismComponents(
        std::vector<std::vector<core::Graph>>& G_components_by_size,
        std::vector<std::vector<core::Graph>>& Q_components_by_size);

    std::optional<std::vector<vertex>> connectedIsomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph);
    std::optional<std::vector<vertex>> isomorphismRecursion(const core::Graph& G, const core::Graph& Q,
                                                            std::unordered_map<vertex, vertex>& Q_G_mapping,
                                                            std::unordered_map<vertex, vertex>& G_Q_mapping, vertex v,
                                                            std::optional<vertex> parent);
};
} // namespace pattern