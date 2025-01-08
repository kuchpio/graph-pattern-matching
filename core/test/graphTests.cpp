#include "core.h"
#include "gtest/gtest.h"

namespace core
{

TEST(GraphConstructor, FromSize) {
    Graph graph(5);

    EXPECT_EQ(graph.size(), 5) << "Graph size should be 5.";
    for (vertex v = 0; v < 5; v++) {
        EXPECT_EQ(graph.degree_in(v), 0) << "Vertex " << v << " should have no incoming edges.";
    }
}

TEST(GraphConstructor, FromEdges) {
    std::vector<std::tuple<vertex, vertex>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}};
    Graph graph(edges);

    EXPECT_EQ(graph.size(), 5) << "Graph size should be 5.";
    EXPECT_EQ(graph.edge_count(), 5) << "Graph should have 5 edges.";

    for (const auto& [u, v] : edges) {
        EXPECT_TRUE(graph.has_edge(u, v)) << "Edge between " << u << " and " << v << " should exist.";
    }
}

TEST(GraphOperations, AddVertex) {
    Graph graph(3);

    auto new_vertex = graph.add_vertex();
    EXPECT_EQ(graph.size(), 4) << "Graph size should be updated to 4.";
    EXPECT_EQ(new_vertex, 3) << "New vertex ID should be 3.";
}

TEST(GraphOperations, AddEdge) {
    Graph graph(3);

    graph.add_edge(0, 1);
    EXPECT_TRUE(graph.has_edge(0, 1)) << "Edge between 0 and 1 should exist.";
}

TEST(GraphOperations, RemoveEdge) {
    Graph graph(3);
    graph.add_edge(0, 1);

    EXPECT_TRUE(graph.remove_edge(0, 1)) << "Edge between 0 and 1 should be removed.";
    EXPECT_FALSE(graph.has_edge(0, 1)) << "Edge between 0 and 1 should no longer exist.";
}

TEST(GraphOperations, RemoveVertex) {
    Graph graph(4);
    graph.add_edge(0, 1);
    graph.add_edge(2, 3);

    EXPECT_TRUE(graph.remove_vertex(1)) << "Vertex 1 should be removed.";
    EXPECT_EQ(graph.size(), 3) << "Graph size should be updated to 3.";
    EXPECT_FALSE(graph.has_edge(0, 1)) << "Edge between 0 and 1 should no longer exist.";
}

TEST(GraphOperations, ContractEdge) {
    Graph graph(5);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);

    EXPECT_TRUE(graph.contract_edge(0, 1)) << "Edge contraction between 0 and 1 should succeed.";
    EXPECT_TRUE(graph.has_edge(0, 1)) << "Vertex 0 should now be connected to vertex 1.";
}

TEST(GraphProperties, DegreeIn) {
    Graph graph(3);
    graph.add_edge(0, 1);
    graph.add_edge(2, 1);

    EXPECT_EQ(graph.degree_in(1), 2) << "Vertex 1 should have an in-degree of 2.";
}

TEST(GraphProperties, DegreeOut) {
    Graph graph(3);
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);

    EXPECT_EQ(graph.degree_out(0), 2) << "Vertex 0 should have an out-degree of 2.";
}

TEST(GraphEquality, EqualGraphs) {
    Graph graph1(3);
    graph1.add_edge(0, 1);
    graph1.add_edge(1, 2);

    Graph graph2(3);
    graph2.add_edge(0, 1);
    graph2.add_edge(1, 2);

    EXPECT_TRUE(graph1 == graph2) << "The graphs should be equal.";
}

TEST(GraphEquality, UnequalGraphs) {
    Graph graph1(3);
    graph1.add_edge(0, 1);

    Graph graph2(3);
    graph2.add_edge(1, 2);

    EXPECT_FALSE(graph1 == graph2) << "The graphs should not be equal.";
}

TEST(GraphSubgraph, IsSubgraph) {
    Graph graph1(5);
    graph1.add_edge(0, 1);
    graph1.add_edge(1, 2);
    graph1.add_edge(2, 3);
    graph1.add_edge(3, 4);

    Graph graph2(3);
    graph2.add_edge(0, 1);
    graph2.add_edge(1, 2);

    EXPECT_TRUE(graph1.has_subgraph(graph2)) << "Graph2 should be a subgraph of Graph1.";
}

TEST(GraphReordering, ApplyMapping) {
    Graph graph(3);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);

    std::vector<vertex> mapping = {2, 0, 1};
    auto reordered_graph = graph.applyMapping(mapping);

    EXPECT_TRUE(reordered_graph.has_edge(2, 0)) << "Edge between 2 and 0 should exist in the reordered graph.";
    EXPECT_TRUE(reordered_graph.has_edge(0, 1)) << "Edge between 0 and 1 should exist in the reordered graph.";
}

} // namespace core
