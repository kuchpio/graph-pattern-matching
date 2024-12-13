add_library(image STATIC "image.cpp" "include/image.h")
target_include_directories(image PUBLIC "include")
target_link_libraries(image core nlohmann_json::nlohmann_json)
add_definitions(-DEDGE_DETECTION_DIR="${CMAKE_SOURCE_DIR}/image/edge_detection")