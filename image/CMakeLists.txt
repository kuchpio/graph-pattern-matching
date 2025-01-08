add_library(image STATIC "image.cpp" "include/image.h")
target_include_directories(image PUBLIC "include")
target_link_libraries(image core nlohmann_json::nlohmann_json)
target_compile_definitions(image PUBLIC EDGE_DETECTION_DIR="${CMAKE_SOURCE_DIR}/image/edge_detection")