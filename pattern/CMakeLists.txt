set(NT_WRAPPERS_DIR "${CMAKE_SOURCE_DIR}/external/nauty/nt_wrappers")
#include_directories(${NT_WRAPPERS_DIR})

add_library(pattern STATIC 
            "src/subgraph_matcher.cpp"
            "src/isomorphism_matcher.cpp"
            "src/induced_subgraph_matcher.cpp"
            "src/minor_matcher.cpp"
            "src/induced_minor_matcher.cpp"
            "src/topological_minor_matcher.cpp"
            "src/topological_induced_minor_matcher.cpp"
            "src/minorMinerSolvers/miner_minor_solver.cpp"
            "src/nautySolvers/nauty_isomorphism_solver.cpp"
            )
target_include_directories(pattern PUBLIC "include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${NT_WRAPPERS_DIR})
target_link_libraries(pattern core utils minorminer pthread ${NT_WRAPPERS_DIR}/nt_wrappers.a)
