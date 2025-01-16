#pragma once

#include "minor_exact.h"

namespace pattern
{
class TopologicalMinorExact : public MinorExact {
  public:
    TopologicalMinorExact(std::unique_ptr<SubgraphMatcher> subgraphMatcher, bool directed = false)
        : MinorExact(std::move(subgraphMatcher), directed) {
        if (directed) maxDeegre_ = 1;
    };

    inline std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& H) override {
        if (H.size() > G.size()) return std::nullopt;
        std::vector<vertex> mapping(G.size());
        std::iota(mapping.begin(), mapping.end(), 0);
        omittableEdges_ = findOmittableEdges(G);
        auto matching = minorRecursion(G, H, mapping, 0, 0);
        if (matching) return getResult(mapping_, matching.value());
        return std::nullopt;
    }

  protected:
    std::size_t maxDeegre_ = 2;

    std::optional<std::vector<vertex>> minorRecursion(const core::Graph& G, const core::Graph& H,
                                                      const std::vector<vertex>& mapping, int depth,
                                                      int lastSkippedEdge);
};

} // namespace pattern