#include <iostream>

#include "image.h"
#include "pattern.h"
#include "cudaTest.h"
#include "render.h"

int main() {
    auto bigGraph = image::grapherize(8);
    auto smallGraph = image::grapherize(6);

    if (pattern::match(bigGraph, smallGraph)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }

    runCudaTest();

    testRender();

    return 0;
}
