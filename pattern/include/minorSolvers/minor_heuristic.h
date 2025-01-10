#pragma once
#include "minor_matcher.h"
#include "subgraph_matcher.h"
#include <memory>
#include <numeric>

namespace pattern
{
class MinorHeuristic : public MinorMatcher {
  public:
    MinorHeuristic(std::unique_ptr<SubgraphMatcher> subgraphMatcher) : subgraphMatcher_(std::move(subgraphMatcher)){};
    inline std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& H) override {
        if (H.size() > G.size()) return std::nullopt;
        std::vector<vertex> mapping(G.size());
        std::iota(mapping.begin(), mapping.end(), 0);

        auto matching = minorRecursion(G, H, mapping, 0, 0);
        if (matching) return getResult(mapping_, matching.value());
        return std::nullopt;
    }

  protected:
    std::unique_ptr<SubgraphMatcher> subgraphMatcher_;
    std::vector<vertex> mapping_;

    virtual std::optional<std::vector<vertex>> minorRecursion(const core::Graph& G, const core::Graph& H,
                                                              const std::vector<vertex>& mapping, int depth,
                                                              int lastSkippedEdge) = 0;
    static core::Graph contractEdge(const core::Graph& G, vertex u, vertex v) {
        core::Graph Q = core::Graph(G);
        Q.contract_edge(u, v);
        return Q;
    }

    inline static std::vector<vertex> updateMapping(const std::vector<vertex>& mapping, vertex u, vertex v) {
        auto newMapping = std::vector<vertex>(mapping);
        for (vertex& vertex : newMapping) {
            if (vertex == v) vertex = u;
            if (vertex > v) vertex--;
        }
        return newMapping;
    }
    inline static std::vector<vertex> getResult(const std::vector<vertex>& mapping,
                                                const std::vector<vertex>& contractedResult) {
        auto result = std::vector<vertex>(mapping);
        for (vertex& v : result) {
            v = contractedResult[v];
        }
        return result;
    }
};
} // namespace pattern