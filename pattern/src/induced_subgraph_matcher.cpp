#include "induced_subgraph_matcher.h"

#include <numeric>

namespace pattern
{
bool InducedSubgraphMatcher::match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    std::unordered_map<vertex, vertex> small_big_mapping = std::unordered_map<vertex, vertex>();
    std::unordered_map<vertex, vertex> big_small_mapping = std::unordered_map<vertex, vertex>();

    auto vertex_indices = std::vector<vertex>(smallGraph.size());
    std::iota(vertex_indices.begin(), vertex_indices.end(), 0);

    for (auto vertex : vertex_indices) {
        if (induced_sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping,
                                              vertex_indices[0]))
            return true;
    }
    return false;
}

bool InducedSubgraphMatcher::induced_sub_isomorphism_recursion(const core::Graph& bigGraph,
                                                               const core::Graph& smallGraph,
                                                               std::unordered_map<vertex, vertex>& small_big_mapping,
                                                               std::unordered_map<vertex, vertex>& big_small_mapping,
                                                               vertex v) {

    if (small_big_mapping.size() == smallGraph.size()) {
        return true;
    }

    for (vertex big_v = 0; big_v < bigGraph.size(); big_v++) {

        if (!can_match_induced_isomorphism(bigGraph, smallGraph, big_small_mapping, small_big_mapping, v, big_v))
            continue;
        big_small_mapping.insert({big_v, v});
        small_big_mapping.insert({v, big_v});

        if (small_big_mapping.size() == smallGraph.size()) {
            return true;
        }

        bool remaining_neighbours = false;
        for (auto neighbour : smallGraph.get_neighbours(v)) {
            if (small_big_mapping.contains(neighbour) == false) {
                remaining_neighbours = true;
                if (induced_sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping,
                                                      neighbour))
                    return true;
            }
        }

        if (remaining_neighbours == false) {
            vertex first_unmapped = find_first_unmapped(smallGraph, small_big_mapping);
            if (induced_sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping,
                                                  first_unmapped))
                return true;
        }
        small_big_mapping.erase(v);
        big_small_mapping.erase(big_v);
    }
    return false;
}

bool InducedSubgraphMatcher::can_match_induced_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph,
                                                           const std::unordered_map<vertex, vertex>& mapping_big_small,
                                                           const std::unordered_map<vertex, vertex>& mapping_small_big,
                                                           vertex v, vertex big_v) {
    if (mapping_big_small.contains(big_v)) return false;
    if (mapping_small_big.contains(v)) return false;

    for (auto neighbour : smallGraph.get_neighbours(v)) {
        if (mapping_small_big.contains(neighbour)) {
            if (bigGraph.has_edge(big_v, mapping_small_big.at(neighbour)) == false) return false;
        }
    }

    for (auto neigbhour : bigGraph.get_neighbours(big_v)) {
        if (mapping_big_small.contains(neigbhour)) {
            if (smallGraph.has_edge(mapping_big_small.at(neigbhour), v) == false) return false;
        }
    }

    for (auto(pair) : mapping_small_big) {
        if (smallGraph.has_edge(pair.first, v) && bigGraph.has_edge(pair.second, big_v) == false) return false;
        if (bigGraph.has_edge(pair.second, big_v) && smallGraph.has_edge(pair.first, v)) return false;
    }
    return true;
}

vertex InducedSubgraphMatcher::find_first_unmapped(const core::Graph& G, std::unordered_map<vertex, vertex> map) {
    for (vertex i = 0; i < G.size(); i++) {
        if (!map.contains(i)) return i;
    }
    return -1;
}

} // namespace pattern