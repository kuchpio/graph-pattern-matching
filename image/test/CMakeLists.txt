add_executable("image-test" "test.cpp")
target_link_libraries("image-test" image)
add_test(NAME "Image-to-graph-conversion" COMMAND "image-test")