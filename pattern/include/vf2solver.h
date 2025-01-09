#pragma once
#include "pattern.h"
#include <argraph.h>
#include "argedit.h"
#include "match.h"

#include <optional>
#include <vector>
#include <cstdint>

namespace pattern
{
class Vf2Solver {
  protected:
    inline Graph convertGraph(const core::Graph& G) {
        ARGEdit ed;
        for (vertex v = 0; v < G.size(); v++) {
            ed.InsertNode(&v);
        }
        for (auto [u, v] : G.edges()) {
            ed.InsertEdge(u, v, nullptr);
        }
        return Graph(&ed);
    }

    inline static std::optional<std::vector<vertex>> processMatching(State& state, const core::Graph& bigGraph,
                                                                     const core::Graph& smallGraph) {
        int n;
        std::vector<node_id> big_nodes(smallGraph.size()), small_nodes(smallGraph.size());

        if (vf2::match(&state, &n, big_nodes.data(), small_nodes.data())) {

            std::vector<vertex> result = std::vector<vertex>(bigGraph.size(), SIZE_MAX);
            for (int i = 0; i < smallGraph.size(); i++) {
                result[small_nodes[i]] = big_nodes[i];
            }
            return result;
        }
        return std::nullopt;
    };
};
} // namespace pattern
