#include "minor_matcher.h"
#include "core.h"
#include "isomorphism_matcher.h"
#include "pattern.h"

namespace pattern
{
bool MinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return minor_recursion(G, H, 0);
};

bool MinorMatcher::minor_recursion(const core::Graph& G, const core::Graph& H, int v) {
    auto isomorphismMatcher = IsomorphismMatcher();
    if (isomorphismMatcher.match(G, H)) return true;

    if (v > G.size()) return false;

    if (minor_recursion(remove_vertex(G, v), H, v + 1)) return true;
    for (auto neighbour : G.get_neighbours(v)) {
        if (minor_recursion(remove_edge(G, v, neighbour), H, v + 1)) return true;
        if (minor_recursion(extract_edge(G, v, neighbour), H, v + 1)) return true;
    }
    return minor_recursion(G, H, v + 1);
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

core::Graph MinorMatcher::extract_edge(const core::Graph& G, int u, int v) {
    core::Graph Q = core::Graph(G);
    Q.extract_edge(u, v);
    return Q;
}

} // namespace pattern