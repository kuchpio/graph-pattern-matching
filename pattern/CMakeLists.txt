
add_library(pattern STATIC 
            "src/isomorphism_matcher.cpp"
            "src/subgraph_matcher.cpp"  
            "src/induced_subgraph_matcher.cpp"
            "src/minor_matcher.cpp"
            "src/induced_minor_matcher.cpp"
            "src/topological_minor_matcher.cpp"
            "src/topological_induced_minor_matcher.cpp"
            "src/minorMinerSolvers/miner_minor_solver.cpp"
            "src/vf2libSolvers/vf2_subisomorphism_solver.cpp"
            "src/vf2libSolvers/vf2_induced_subgraph_solver.cpp"
            )
target_include_directories(pattern PUBLIC "include" "include/vf2libSolvers" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(pattern core utils minorminer vf2lib)
