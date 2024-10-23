#include <iostream>

#include "image.h"
#include "isomorphism_matcher.h"
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
    Q.add_edge(1, 4);

    auto matcher = pattern::IsomorphismMatcher();

    if (matcher.match(G, Q)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }

    return 0;
}
