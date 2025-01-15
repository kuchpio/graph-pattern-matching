add_library(core STATIC "src/core.cpp" "include/core.h" "src/graph6Serializer.cpp" "include/graph6Serializer.h")
target_include_directories(core PUBLIC "include")
