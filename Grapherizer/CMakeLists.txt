add_library(Grapherizer STATIC "Grapherizer.cpp" "include/Grapherizer.h")
target_include_directories(Grapherizer PUBLIC "include")
target_link_libraries(Grapherizer Core)
