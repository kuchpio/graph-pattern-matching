add_executable (app "app.cpp" "cudaTest.cu" "cudaTest.h")
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "50" CXX_STANDARD 17)

target_link_libraries(app core image pattern)
target_include_directories(app PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
