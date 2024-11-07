#include "minor_matcher.h"
#include "core.h"
#include "isomorphism_matcher.h"
#include <optional>

namespace pattern
{
bool MinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return minor_recursion(G, H, 0, std::nullopt);
};

bool MinorMatcher::minor_recursion(const core::Graph& G, const core::Graph& H, std::size_t v,
                                   std::optional<int> last_neighbour_index) {
    if (H.size() > G.size()) return false;

    if (this->isomorphismMatcher.match(G, H)) return true;

    if (v >= G.size()) return false;

    int start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        if (minor_recursion(G_after_removal, H, v + 1, std::nullopt)) return true;
    }

    for (int neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size(); neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];

        auto G_after_edge_removal = remove_edge(G, v, neighbour);

        if (minor_recursion(G_after_edge_removal, H, v, neighbour_index)) return true;
        auto G_after_edge_contraction = contract_edge(G, v, neighbour);
        if (minor_recursion(G_after_edge_contraction, H, v, neighbour_index)) return true;
    }

    return minor_recursion(G, H, v + 1, std::nullopt);
}

core::Graph MinorMatcher::remove_vertex(const core::Graph& G, int v) {
    core::Graph Q = core::Graph(G);
    Q.remove_vertex(v);
    return Q;
}

core::Graph MinorMatcher::remove_edge(const core::Graph& G, int u, int v) {
    core::Graph Q = core::Graph(G);
    Q.remove_edge(u, v);
    return Q;
}

core::Graph MinorMatcher::contract_edge(const core::Graph& G, int u, int v) {
    core::Graph Q = core::Graph(G);
    Q.contract_edge(u, v);
    return Q;
}

} // namespace pattern