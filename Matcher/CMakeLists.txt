add_library(Matcher STATIC "Matcher.cpp" "include/Matcher.h")
target_include_directories(Matcher PUBLIC "include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Matcher Core)
