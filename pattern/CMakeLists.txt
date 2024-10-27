add_library(pattern STATIC 
            "src/pattern.cpp" 
            "src/cpu_pattern.cpp" 
            "src/subgraph_matcher.cpp"
            "src/isomorphism_matcher.cpp"
            "src/induced_subgraph_matcher.cpp"
            "src/minor_matcher.cpp"
            "src/induced_minor_matcher.cpp"
            )
target_include_directories(pattern PUBLIC "include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(pattern core utils)
