add_library(Matcher STATIC "Matcher.cpp" "include/Matcher.h")
target_include_directories(Matcher PUBLIC "include")
target_link_libraries(Matcher Core)
