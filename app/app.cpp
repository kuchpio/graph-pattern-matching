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
    core::Graph G = utils::GraphFactory::random_connected_graph(graph_size);
    core::Graph Q = utils::GraphFactory::random_minor(G, subgraph_size);

    auto matcher = pattern::MinerMinorMatcher();

    // auto matcher = pattern::MinorMatcher();

    if (matcher.match(G, Q)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }

    return 0;
}
