add_library(pattern STATIC 
            "src/isomorphismSolvers/native_isomorphism_matcher.cpp"
            "src/minorSolvers/miner_minor_solver.cpp"
            "src/subgraphSolvers/cuda_subgraph_matcher.cu"
            "src/minorSolvers/topological_minor_exact.cpp"
            "src/minorSolvers/induced_minor_exact_matcher.cpp"
            "src/minorSolvers/minor_exact_matcher.cpp"
            )
target_include_directories(pattern PUBLIC "include" "include/minorSolvers" "include/subgraphSolvers" 
                            "include/isomorphismSolvers" "${PROJECT_BINARY_DIR}/config" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(pattern PROPERTIES CUDA_ARCHITECTURES "61")

target_link_libraries(pattern core utils minorminer vf2lib)
target_compile_options(pattern PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G -lineInfo>)

