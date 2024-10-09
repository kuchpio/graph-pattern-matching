cmake_minimum_required (VERSION 3.9)

set(CMAKE_CXX_STANDARD 23)

project("graph-pattern-matching")
enable_language("CUDA")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

add_subdirectory("app")

add_subdirectory("core")

add_subdirectory("image")
add_subdirectory("image/test")

add_subdirectory("pattern")
add_subdirectory("pattern/test")

add_subdirectory("utils")
