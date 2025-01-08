add_executable("pattern-test"  
    "isomorphismMatcherTests.cpp" 
    "subgraphMatcherTests.cpp" 
    "minorMatcherTests.cpp"
    "cudaTests.cpp"
)
target_link_libraries("pattern-test" PUBLIC pattern GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests("pattern-test")

