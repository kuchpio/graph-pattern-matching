#include <cstddef>
#include <iostream>

#include "image.h"
#include "isomorphism_matcher.h"
#include "miner_minor_matcher.hpp"
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

    vertex graph_size = 8;
    vertex subgraph_size = 5;

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

    auto matcher = pattern::MinerMinorMatcher();

    // auto matcher = pattern::MinorMatcher();

    if (matcher.match(G, Q)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }

    return 0;
}
