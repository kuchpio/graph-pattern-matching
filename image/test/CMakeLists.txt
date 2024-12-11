add_executable("image-test" "test.cpp")
target_link_libraries("image-test" PUBLIC image GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests("image-test")