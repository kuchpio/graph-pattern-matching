#include "subgraph_matcher.h"

#include <numeric>

namespace pattern
{
bool SubgraphMatcher::match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    std::unordered_map<int, int> small_big_mapping = std::unordered_map<int, int>();
    std::unordered_map<int, int> big_small_mapping = std::unordered_map<int, int>();

    auto vertex_indices = std::vector<int>(smallGraph.size());
    std::iota(vertex_indices.begin(), vertex_indices.end(), 0);

    for (auto vertex : vertex_indices) {
        if (sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping, vertex_indices[0]))
            return true;
    }
    return false;
}

bool SubgraphMatcher::sub_isomorphism_recursion(const core::Graph& bigGraph, const core::Graph& smallGraph,
                                                std::unordered_map<int, int>& small_big_mapping,
                                                std::unordered_map<int, int>& big_small_mapping, int v) {

    if (small_big_mapping.size() == smallGraph.size()) {
        return true;
    }

    for (std::size_t big_v = 0; big_v < bigGraph.size(); big_v++) {

        if (!can_match_isomorphism(bigGraph, smallGraph, big_small_mapping, small_big_mapping, v, big_v)) continue;
        big_small_mapping.insert({big_v, v});
        small_big_mapping.insert({v, big_v});

        if (small_big_mapping.size() == smallGraph.size()) {
            return true;
        }

        bool remaining_neighbours = false;
        for (auto neighbour : smallGraph.get_neighbours(v)) {
            if (small_big_mapping.contains(neighbour) == false) {
                remaining_neighbours = true;
                if (sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping, neighbour))
                    return true;
            }
        }

        if (remaining_neighbours == false) {
            int first_unmapped = find_first_unmapped(smallGraph, small_big_mapping);
            if (sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping, first_unmapped))
                return true;
        }
        small_big_mapping.erase(v);
        big_small_mapping.erase(big_v);
    }
    return false;
}

bool SubgraphMatcher::can_match_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph,
                                            const std::unordered_map<int, int>& mapping_big_small,
                                            const std::unordered_map<int, int>& mapping_small_big, int v, int big_v) {
    if (mapping_big_small.contains(big_v)) return false;
    if (mapping_small_big.contains(v)) return false;

    for (auto neighbour : smallGraph.get_neighbours(v)) {
        if (mapping_small_big.contains(neighbour)) {
            if (bigGraph.has_edge(big_v, mapping_small_big.at(neighbour)) == false) return false;
        }
    }
    for (auto(pair) : mapping_small_big) {
        if (smallGraph.has_edge(pair.first, v) && bigGraph.has_edge(pair.second, big_v) == false) return false;
    }
    return true;
}

int SubgraphMatcher::find_first_unmapped(const core::Graph& G, std::unordered_map<int, int> map) {
    for (int i = 0; i < G.size(); i++) {
        if (!map.contains(i)) return i;
    }
    return -1;
}

} // namespace pattern