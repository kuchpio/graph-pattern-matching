add_library(utils STATIC "utils.cpp" "include/utils.h")
target_include_directories(utils PUBLIC "include")
target_link_libraries(utils core)