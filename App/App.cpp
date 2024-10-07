#include <iostream>

#include "Grapherizer.h"
#include "Matcher.h"
#include "cudaTest.h"

int main()
{
	auto bigGraph = grz::grapherize(8);
	auto smallGraph = grz::grapherize(6);

	if (mtr::match(bigGraph, smallGraph)) {
		std::cout << "Match found." << std::endl;
	} else {
		std::cout << "Match not found." << std::endl;
	}

	runCudaTest();

	return 0;
}
