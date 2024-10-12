#include "core.h"
#include "pattern.h"
#include "utils.h"
#include <cassert>
bool random_graph_isomorphism_test();
bool small_graph_not_isomorphic();
int main() {
    assert(random_graph_isomorphism_test() == true);
    assert(small_graph_not_isomorphic() == false);
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
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 4);

    bool is_isomorphic = pattern::is_isomorphism(G, Q);

    return is_isomorphic;
}

bool random_graph_isomorphism_test() {
    core::Graph G = utils::GraphFactory::random_graph(10, 0.4f);
    core::Graph Q = utils::GraphFactory::isomoporhic_graph(G);
    return pattern::is_isomorphism(G, Q);
}
