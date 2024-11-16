add_executable("core-test" "graph6SerializerTests.cpp")
target_link_libraries("core-test" PUBLIC core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests("core-test")
