#pragma once
#include "minor_matcher.h"
#include "subgraph_matcher.h"
#include <memory>
#include <numeric>
#include <map>

namespace pattern
{
class MinorExact : public MinorMatcher {
  public:
    MinorExact(std::unique_ptr<SubgraphMatcher> subgraphMatcher, bool directed = false)
        : subgraphMatcher_(std::move(subgraphMatcher)), directed_(directed){};
    virtual std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& H) = 0;

  protected:
    std::unique_ptr<SubgraphMatcher> subgraphMatcher_;
    std::map<std::tuple<vertex, vertex>, bool> omittableEdges_;
    std::vector<vertex> mapping_;
    bool directed_ = false;

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
    inline std::map<std::tuple<vertex, vertex>, bool> findOmittableEdges(const core::Graph& G) {
        std::map<std::tuple<vertex, vertex>, bool> omittableEdges;
        auto maxDeegree = directed_ ? 1 : 2;
        if (!G.connected() || G.size() == 0) return omittableEdges;
        for (vertex v = 0; v < G.size(); ++v) {
            if (G.degree_in(v) == maxDeegree && G.degree_out(v) == maxDeegree)
                omittableEdges.insert({std::tie(v, G.neighbours(v)[0]), false});
        }
        return omittableEdges;
    }
};
} // namespace pattern