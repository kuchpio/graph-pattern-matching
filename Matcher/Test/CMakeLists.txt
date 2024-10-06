add_executable(MatcherTest "Test.cpp")
target_link_libraries(MatcherTest Matcher)
add_test(NAME MatcherTest COMMAND MatcherTest)
