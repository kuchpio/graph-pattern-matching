#include "native_minor_matcher.h"
#include "core.h"
#include <optional>
#include <vector>

namespace pattern
{
bool NativeMinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return minor_recursion(G, H, 0, std::nullopt);
};

std::optional<std::vector<vertex>> NativeMinorMatcher::matching(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return std::nullopt;

    return minorRecursion(G, H, 0, std::nullopt);
};

bool NativeMinorMatcher::minor_recursion(const core::Graph& G, const core::Graph& H, vertex v,
                                         std::optional<vertex> last_neighbour_index) {
    if (H.size() > G.size()) return false;

    if (this->isomorphismMatcher.match(G, H)) return true;

    if (v >= G.size()) return false;

    vertex start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        if (minor_recursion(G_after_removal, H, v + 1, std::nullopt)) return true;
    }

    for (vertex neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size();
         neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];

        auto G_after_edge_removal = remove_edge(G, v, neighbour);

        if (minor_recursion(G_after_edge_removal, H, v, neighbour_index)) return true;
        auto G_after_edge_contraction = contract_edge(G, v, neighbour);
        if (minor_recursion(G_after_edge_contraction, H, v, neighbour_index)) return true;
    }

    return minor_recursion(G, H, v + 1, std::nullopt);
}

std::optional<std::vector<vertex>> NativeMinorMatcher::minorRecursion(const core::Graph& G, const core::Graph& H,
                                                                      vertex v,
                                                                      std::optional<vertex> last_neighbour_index) {
    if (H.size() > G.size()) return std::nullopt;

    auto matching = this->isomorphismMatcher.matching(G, H);
    if (matching.has_value()) return matching;

    if (v >= G.size()) return std::nullopt;

    vertex start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        auto matching = minorRecursion(G_after_removal, H, v + 1, std::nullopt);
        if (matching.has_value()) return *matching;
    }

    for (vertex neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size();
         neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];

        auto G_after_edge_removal = remove_edge(G, v, neighbour);

        auto matching = minorRecursion(G_after_edge_removal, H, v + 1, std::nullopt);
        if (matching.has_value()) return *matching;

        auto G_after_edge_contraction = contract_edge(G, v, neighbour);
        matching = minorRecursion(G_after_edge_contraction, H, v, neighbour_index);
        if (matching.has_value()) return matching;
    }

    return minorRecursion(G, H, v + 1, std::nullopt);
}

core::Graph NativeMinorMatcher::remove_vertex(const core::Graph& G, vertex v) {
    core::Graph Q = core::Graph(G);
    Q.remove_vertex(v);
    return Q;
}

core::Graph NativeMinorMatcher::remove_edge(const core::Graph& G, vertex u, vertex v) {
    core::Graph Q = core::Graph(G);
    Q.remove_edge(u, v);
    return Q;
}

core::Graph NativeMinorMatcher::contract_edge(const core::Graph& G, vertex u, vertex v) {
    core::Graph Q = core::Graph(G);
    Q.contract_edge(u, v);
    return Q;
}

} // namespace pattern