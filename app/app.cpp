#include <cstddef>
#include <iostream>

#include "image.h"
#include "miner_minor_matcher.hpp"
#include "minor_matcher.h"
#include "pattern.h"
#include "subgraph_matcher.h"
#include "utils.h"
#include "cudaTest.h"
#include "vf2_induced_subgraph_solver.hpp"
#include "vf2_subgraph_solver.hpp"

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

    core::Graph G = utils::GraphFactory::random_connected_graph(30, 0.4f);
    core::Graph Q = utils::GraphFactory::isomoporhic_graph(G);
    auto matcher = pattern::Vf2InducedSubgraphSolver();

    // auto matcher = pattern::MinorMatcher();

    if (matcher.match(G, Q)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }

    return 0;
}
