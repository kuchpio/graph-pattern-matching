#include "core.h"
#include "isomorphism_matcher.h"
#include "minor_matcher.h"
#include "pattern.h"
#include "subgraph_matcher.h"
#include <cassert>
bool random_graph_isomorphism_test();
bool small_graph_not_isomorphic();
bool subgraph_not_sub_isomorphic();
bool small_graph_sub_isomorphic();
bool small_not_minor();
bool small_has_minor();

bool small_graph_isomorphic();
int main() {
    assert(random_graph_isomorphism_test() == true);
    assert(small_graph_not_isomorphic() == false);
    assert(small_graph_isomorphic() == true);
    assert(small_graph_sub_isomorphic() == true);
    assert(subgraph_not_sub_isomorphic() == false);
    assert(small_not_minor() == false);
    assert(small_has_minor() == true);

    return 0;
}

bool small_graph_not_isomorphic() {
    int small_size = 5;
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

    auto matcher = pattern::IsomorphismMatcher();

    return matcher.match(G, Q);
}

bool small_graph_isomorphic() {
    int small_size = 5;
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

    auto matcher = pattern::IsomorphismMatcher();

    return matcher.match(G, Q);
}

bool random_graph_isomorphism_test() {
    core::Graph G = utils::GraphFactory::random_graph(30, 0.4f);
    core::Graph Q = utils::GraphFactory::isomoporhic_graph(G);
    auto matcher = pattern::IsomorphismMatcher();
    return matcher.match(G, Q);
}

bool subgraph_not_sub_isomorphic() {
    int graph_size = 6;
    int subgraph_size = 4;

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

    pattern::SubgraphMatcher matcher = pattern::SubgraphMatcher();

    // Check for subgraph isomorphism
    return matcher.match(G, Q);
}

bool small_graph_sub_isomorphic() {
    int graph_size = 6;
    int subgraph_size = 4;

    pattern::SubgraphMatcher matcher = pattern::SubgraphMatcher();

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
    return matcher.match(G, Q);
}

bool small_not_minor() {
    int graph_size = 6;
    int subgraph_size = 4;

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

    auto matcher = pattern::MinorMatcher();

    // Check for minor relationship - expecting false because Q is a cycle but G is a chain
    return matcher.match(G, Q) == false;
}

bool small_has_minor() {
    int graph_size = 6;
    int subgraph_size = 3;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);
    G.add_edge(1, 3); // Extra edge to create a potential contraction

    // Define edges for the smaller graph Q
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);

    auto matcher = pattern::MinorMatcher();
    // Check for minor relationship - expecting true because Q can be derived from G
    return matcher.match(G, Q);
}