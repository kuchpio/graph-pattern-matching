add_library(image STATIC "image.cpp" "include/image.h")
target_include_directories(image PUBLIC "include")
target_link_libraries(image core)
