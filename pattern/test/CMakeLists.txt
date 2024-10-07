add_executable("pattern-test" "test.cpp")
target_link_libraries("pattern-test" pattern)
add_test(NAME "Graph-pattern-matching" COMMAND "pattern-test")
