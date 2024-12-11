
add_library(pattern STATIC 
            "src/isomorphismSolvers/native_isomorphism_matcher.cpp"
            "src/subgraphSolvers/native_subgraph_matcher.cpp"  
            "src/subgraphSolvers/induced_subgraph_matcher.cpp"
            "src/minorSolvers/minor_matcher.cpp"
            "src/minorSolvers/induced_minor_matcher.cpp"
            "src/minorSolvers/topological_minor_matcher.cpp"
            "src/minorSolvers/topological_induced_minor_matcher.cpp"
            "src/minorSolvers/miner_minor_solver.cpp"
            "src/subgraphSolvers/vf2_subisomorphism_solver.cpp"
            "src/subgraphSolvers/vf2_induced_subgraph_solver.cpp"
            "src/isomorphismSolvers/vf2_isomorphism_solver.cpp"
            "src/subgraphSolvers/cuda_subgraph_matcher.cu"
            )
target_include_directories(pattern PUBLIC "include" "include/minorSolvers" "include/subgraphSolvers" 
                            "include/isomorphismSolvers"  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(pattern PROPERTIES CUDA_ARCHITECTURES "61" CXX_STANDARD 20)

target_link_libraries(pattern core utils minorminer vf2lib)
target_compile_options(pattern PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G -lineInfo>)

