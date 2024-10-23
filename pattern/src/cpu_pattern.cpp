#include "core.h"
#include "pattern.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace pattern
{
bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return bigGraph.size() >= smallGraph.size();
}

bool sub_induced_isomorpshim(const core::Graph& bigGraph, const core::Graph& smallGraph) {

    if (bigGraph.size() == smallGraph.size()) return isomorphism(bigGraph, smallGraph);
    auto removed_vertices = std::vector<int>(bigGraph.size() - smallGraph.size());

    std::iota(removed_vertices.begin(), removed_vertices.end(), 0);

    do {
        core::Graph Q = core::Graph(bigGraph);
        Q.remove_vertices(removed_vertices);
        if (isomorphism(Q, smallGraph)) return true;

    } while (std::prev_permutation(removed_vertices.begin(), removed_vertices.end()));

    return false;
}

bool can_match_induced_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph,
                                   const std::unordered_map<int, int>& mapping_small_big,
                                   const std::unordered_map<int, int>& mapping_big_small, int v, int big_v) {
    if (mapping_small_big.contains(v)) return true;

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
} // namespace pattern
