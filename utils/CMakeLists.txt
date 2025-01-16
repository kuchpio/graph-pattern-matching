add_library(utils STATIC "utils.cpp" "generateSamples.cpp")
target_include_directories(utils PUBLIC "include")
target_link_libraries(utils core pattern)