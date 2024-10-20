#include <iostream>

#include "image.h"
#include "pattern.h"
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

    auto G = utils::GraphFactory::random_graph(10, 0.4f);
    auto Q = utils::GraphFactory::isomoporhic_graph(G);

    if (pattern::isomorphism(G, Q)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }

    return 0;
}
