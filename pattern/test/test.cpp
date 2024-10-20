#include "core.h"
#include "pattern.h"
#include <cassert>
bool random_graph_isomorphism_test();
bool small_graph_not_isomorphic();

bool small_graph_isomorphic();
int main() {
    assert(random_graph_isomorphism_test() == true);
    assert(small_graph_not_isomorphic() == false);
    assert(small_graph_isomorphic() == true);
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

    return pattern::connected_isomorphism(G, Q);
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

    return pattern::connected_isomorphism(G, Q);
}

bool random_graph_isomorphism_test() {
    core::Graph G = utils::GraphFactory::random_graph(30, 0.4f);
    core::Graph Q = utils::GraphFactory::isomoporhic_graph(G);
    return pattern::isomorphism(G, Q);
}
