cmake_minimum_required (VERSION 3.19 FATAL_ERROR)

project("graph-pattern-matching")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language("CUDA")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(CUDA_ENABLED)
endif()

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

add_subdirectory("plugin-sample")

add_subdirectory("app")

set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/app" PROPERTY VS_STARTUP_PROJECT "app")
