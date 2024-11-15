cmake_minimum_required (VERSION 3.16 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)

project("graph-pattern-matching")
enable_language("CUDA")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_subdirectory("external")

add_subdirectory("core")
add_subdirectory("core/test")

add_subdirectory("utils")

add_subdirectory("image")
add_subdirectory("image/test")

add_subdirectory("pattern")
add_subdirectory("pattern/test")

add_subdirectory("app")

set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/app" PROPERTY VS_STARTUP_PROJECT "app")
