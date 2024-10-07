add_executable (App "App.cpp" "cudaTest.cu" "cudaTest.h")
set_target_properties(App PROPERTIES CUDA_ARCHITECTURES "50" CXX_STANDARD 17)

target_link_libraries(App Core Grapherizer Matcher)
target_include_directories(App PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(App PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
