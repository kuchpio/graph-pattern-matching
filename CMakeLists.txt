cmake_minimum_required (VERSION 3.9)

project("graph-pattern-matching")
enable_language("CUDA")

include(CTest)

add_subdirectory("App")

add_subdirectory("Core")

add_subdirectory("Grapherizer")
add_subdirectory("Grapherizer/Test")

add_subdirectory("Matcher")
add_subdirectory("Matcher/Test")
