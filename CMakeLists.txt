cmake_minimum_required (VERSION 3.9)

project("graph-pattern-matching")
enable_language("CUDA")

include(CTest)

add_subdirectory("app")

add_subdirectory("core")

add_subdirectory("image")
add_subdirectory("image/test")

add_subdirectory("pattern")
add_subdirectory("pattern/test")
