add_library(image STATIC "image.cpp" "include/image.h")
target_include_directories(image PUBLIC "include")
target_link_libraries(image core nlohmann_json::nlohmann_json)

add_custom_target(copy-edge-detection-model ALL 
	COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/edge_detection" "${CMAKE_CURRENT_BINARY_DIR}/edge_detection" 
	COMMENT "Copying edge detection model"
)
add_dependencies(image copy-edge-detection-model)
