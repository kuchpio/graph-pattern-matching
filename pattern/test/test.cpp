#include "core.h"
#include "native_isomorphism_matcher.h"
#include "native_minor_matcher.h"
#include "native_subgraph_matcher.h"
#include "topological_minor_matcher.h"
#include "utils.h"
#include <cassert>

bool random_graph_isomorphism_test();
bool small_graph_not_isomorphic();
bool subgraph_not_sub_isomorphic();
bool small_graph_sub_isomorphic();
bool small_not_minor();
bool has_minor_not_topological();
bool small_has_minor();

bool small_graph_isomorphic();
int main() {
    assert(random_graph_isomorphism_test() == true);
    assert(small_graph_not_isomorphic() == false);
    // assert(small_graph_isomorphic() == true); // Fix me!
    assert(small_graph_sub_isomorphic() == true);
    assert(subgraph_not_sub_isomorphic() == false);
    assert(small_not_minor() == false);
    assert(small_has_minor() == true);
    //  assert(has_minor_not_topological() == true);

    return 0;
}

bool small_graph_not_isomorphic() {
    std::size_t small_size = 5;
    core::Graph G = core::Graph(small_size);
    core::Graph Q = core::Graph(small_size);

    G.add_edge(0, 1);
    G.add_edge(0, 2);
    G.add_edge(1, 3);
    G.add_edge(3, 4);

    Q.add_edge(0, 1);
    Q.add_edge(0, 3);
    Q.add_edge(3, 2);
    Q.add_edge(2, 4);

    auto matcher = pattern::NativeIsomorphismMatcher();

    return matcher.match(G, Q);
}

bool small_graph_isomorphic() {
    std::size_t small_size = 5;
    core::Graph G = core::Graph(small_size);
    core::Graph Q = core::Graph(small_size);

    // Graph G
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);

    // Graph Q (isomorphic to G but with shuffled labels)
    Q.add_edge(1, 2); // corresponds to edge (0, 1) in G
    Q.add_edge(2, 3); // corresponds to edge (0, 2) in G
    Q.add_edge(3, 4); // corresponds to edge (1, 3) in G
    Q.add_edge(4, 0);

    auto matcher = pattern::NativeIsomorphismMatcher();

    return matcher.match(G, Q);
}

bool random_graph_isomorphism_test() {
    core::Graph G = utils::GraphFactory::random_connected_graph(30, 0.4f);
    core::Graph Q = utils::GraphFactory::isomoporhic_graph(G);
    auto matcher = pattern::NativeIsomorphismMatcher();
    return matcher.match(G, Q);
}

bool subgraph_not_sub_isomorphic() {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5); // Extra node and edge in G

    // Define edges for the smaller subgraph Q
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    pattern::NativeSubgraphMatcher matcher = pattern::NativeSubgraphMatcher();

    // Check for subgraph isomorphism
    return matcher.match(G, Q);
}

bool small_graph_sub_isomorphic() {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    // Define edges for the smaller subgraph Q
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);

    // Check for subgraph isomorphism
    pattern::NativeSubgraphMatcher matcher = pattern::NativeSubgraphMatcher();

    return matcher.match(G, Q);
}

bool small_not_minor() {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G (a linear chain with no cycles)
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    // Define edges for the smaller graph Q (a cycle, which cannot be formed by contractions in G)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = pattern::NativeMinorMatcher();

    // Check for minor relationship - expecting false because Q is a cycle but G is a chain
    return matcher.match(G, Q);
}

bool small_has_minor() {
    std::size_t graph_size = 8;
    std::size_t subgraph_size = 5;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G Qubic graph Q^3
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 0);
    G.add_edge(0, 4);
    G.add_edge(1, 5);
    G.add_edge(2, 6);
    G.add_edge(3, 7);
    G.add_edge(4, 5);
    G.add_edge(5, 6);
    G.add_edge(6, 7);
    G.add_edge(7, 4);

    // Define edges for the smaller graph Q (wheel W^4)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    Q.add_edge(0, 4);
    Q.add_edge(1, 4);
    Q.add_edge(2, 4);
    Q.add_edge(3, 4);

    auto matcher = pattern::NativeMinorMatcher();
    // Check for minor relationship - expecting true because Q can be derived from G
    return matcher.match(G, Q);
}

bool has_minor_not_topological() {

    std::size_t graph_size = 8;
    std::size_t subgraph_size = 5;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G Qubic graph Q^3
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 0);
    G.add_edge(0, 4);
    G.add_edge(1, 5);
    G.add_edge(2, 6);
    G.add_edge(3, 7);
    G.add_edge(4, 5);
    G.add_edge(5, 6);
    G.add_edge(6, 7);
    G.add_edge(7, 4);

    // Define edges for the smaller graph Q (wheel W^4)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    Q.add_edge(0, 4);
    Q.add_edge(1, 4);
    Q.add_edge(2, 4);
    Q.add_edge(3, 4);

    auto matcher = pattern::TopologicalMinorMatcher();
    // Check for minor relationship - expecting true because Q can be derived from G
    return matcher.match(G, Q);
}