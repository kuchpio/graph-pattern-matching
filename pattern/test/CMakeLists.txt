add_executable("pattern-test"  "isomorphismMatcherTests.cpp")
target_link_libraries("pattern-test" PUBLIC pattern GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests("pattern-test")

add_test(NAME "Graph-pattern-matching" COMMAND "pattern-test")
