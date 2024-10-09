#include "pattern.h"
#include "utils.h"

int main() {
}

bool random_graph_isomorphism_test() {

    core::Graph G = utils::GraphFactory::random_graph(100, 0.4f);
    core::Graph Q = utils::GraphFactory::isomoporhic_graph(G);

    return true;
}
