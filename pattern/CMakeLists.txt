add_library(pattern STATIC "pattern.cpp" "include/pattern.h")
target_include_directories(pattern PUBLIC "include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(pattern core)
