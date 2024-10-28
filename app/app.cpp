#include <iostream>

#include "image.h"
#include "isomorphism_matcher.h"
#include "minor_matcher.h"
#include "pattern.h"
#include "subgraph_matcher.h"
#include "utils.h"
#include "cudaTest.h"

int main() {
    /*
    auto bigGraph = image::grapherize(8);
    auto smallGraph = image::grapherize(6);

    if (pattern::match(bigGraph, smallGraph)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }
    */

    // runCudaTest();

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

    if (matcher.match(G, Q)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }

    return 0;
}
