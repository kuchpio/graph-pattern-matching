add_executable(GrapherizerTest "Test.cpp")
target_link_libraries(GrapherizerTest Grapherizer)
add_test(NAME GrapherizerTest COMMAND GrapherizerTest)