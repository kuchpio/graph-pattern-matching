#include "core.h"
#include "graph6Serializer.h"
#include "gtest/gtest.h"

namespace core
{

TEST(Graph6Deserialize, Path_5) {
    const bool expectedAdjecencyMatrix[] = {0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};

    auto graph = Graph6Serializer::Deserialize("DQc");

    EXPECT_EQ(graph.size(), 5) << "Incorrect size";
    for (vertex u = 0; u < 5; u++) {
        for (vertex v = 0; v < 5; v++) {
            EXPECT_EQ(graph.has_edge(u, v), expectedAdjecencyMatrix[u * 5 + v])
                << "The edge between " << u << " and " << v << " should "
                << (expectedAdjecencyMatrix[u * 5 + v] ? "not " : "") << " exist.";
        }
    }
}

TEST(Graph6Deserialize, Cycle_5) {
    const bool expectedAdjecencyMatrix[] = {0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0};

    auto graph = Graph6Serializer::Deserialize("DdW");

    EXPECT_EQ(graph.size(), 5) << "Incorrect size";
    for (vertex u = 0; u < 5; u++) {
        for (vertex v = 0; v < 5; v++) {
            EXPECT_EQ(graph.has_edge(u, v), expectedAdjecencyMatrix[u * 5 + v])
                << "The edge between " << u << " and " << v << " should "
                << (expectedAdjecencyMatrix[u * 5 + v] ? "not " : "") << " exist.";
        }
    }
}

TEST(Graph6Deserialize, Star_5) {
    const bool expectedAdjecencyMatrix[] = {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};

    auto graph = Graph6Serializer::Deserialize("DXG");

    EXPECT_EQ(graph.size(), 5) << "Incorrect size";
    for (vertex u = 0; u < 5; u++) {
        for (vertex v = 0; v < 5; v++) {
            EXPECT_EQ(graph.has_edge(u, v), expectedAdjecencyMatrix[u * 5 + v])
                << "The edge between " << u << " and " << v << " should "
                << (expectedAdjecencyMatrix[u * 5 + v] ? "not " : "") << " exist.";
        }
    }
}

TEST(Graph6Deserialize, Full_5) {
    const bool expectedAdjecencyMatrix[] = {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};

    auto graph = Graph6Serializer::Deserialize("D~{");

    EXPECT_EQ(graph.size(), 5) << "Incorrect size";
    for (vertex u = 0; u < 5; u++) {
        for (vertex v = 0; v < 5; v++) {
            EXPECT_EQ(graph.has_edge(u, v), expectedAdjecencyMatrix[u * 5 + v])
                << "The edge between " << u << " and " << v << " should "
                << (expectedAdjecencyMatrix[u * 5 + v] ? "not " : "") << " exist.";
        }
    }
}

TEST(Graph6Deserialize, Header) {
    const bool expectedAdjecencyMatrix[] = {0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};

    auto graph = Graph6Serializer::Deserialize(">>graph6<<DQc");

    EXPECT_EQ(graph.size(), 5) << "Incorrect size";
    for (vertex u = 0; u < 5; u++) {
        for (vertex v = 0; v < 5; v++) {
            EXPECT_EQ(graph.has_edge(u, v), expectedAdjecencyMatrix[u * 5 + v])
                << "The edge between " << u << " and " << v << " should "
                << (expectedAdjecencyMatrix[u * 5 + v] ? "not " : "") << " exist.";
        }
    }
}

TEST(Graph6Deserialize, NoContent) {
    EXPECT_THROW(Graph6Serializer::Deserialize(""), graph6FormatError);
}

TEST(Graph6Deserialize, InvalidSizeEncoding) {
    EXPECT_THROW(Graph6Serializer::Deserialize("~x"), graph6FormatError);
}

TEST(Graph6Deserialize, InvalidCharacter) {
    EXPECT_THROW(Graph6Serializer::Deserialize("D<Qc"), graph6InvalidCharacterError);
}

TEST(Graph6Deserialize, EncodingTooShort) {
    EXPECT_THROW(Graph6Serializer::Deserialize("DX"), graph6FormatError);
}

TEST(Graph6Deserialize, EncodingTooLong) {
    EXPECT_THROW(Graph6Serializer::Deserialize("DXXX"), graph6FormatError);
}

TEST(Graph6Deserialize, InvalidEncodingPadding) {
    EXPECT_THROW(Graph6Serializer::Deserialize("DX~"), graph6FormatError);
}

TEST(Graph6Serialize, Path_5) {
    auto graph = core::Graph(5);
    graph.add_edge(0, 2);
    graph.add_edge(2, 0);
    graph.add_edge(0, 4);
    graph.add_edge(4, 0);
    graph.add_edge(1, 3);
    graph.add_edge(3, 1);
    graph.add_edge(3, 4);
    graph.add_edge(4, 3);

    auto graph6 = Graph6Serializer::Serialize(graph);

    EXPECT_EQ(graph6, "DQc");
}

TEST(Graph6Serialize, Cycle_5) {
    auto graph = core::Graph(5);
    graph.add_edge(0, 3);
    graph.add_edge(3, 0);
    graph.add_edge(3, 2);
    graph.add_edge(2, 3);
    graph.add_edge(2, 4);
    graph.add_edge(4, 2);
    graph.add_edge(4, 1);
    graph.add_edge(1, 4);
    graph.add_edge(1, 0);
    graph.add_edge(0, 1);

    auto graph6 = Graph6Serializer::Serialize(graph);

    EXPECT_EQ(graph6, "DdW");
}

TEST(Graph6Serialize, Star_5) {
    auto graph = core::Graph(5);
    graph.add_edge(2, 0);
    graph.add_edge(0, 2);
    graph.add_edge(2, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 3);
    graph.add_edge(3, 2);
    graph.add_edge(2, 4);
    graph.add_edge(4, 2);

    auto graph6 = Graph6Serializer::Serialize(graph);

    EXPECT_EQ(graph6, "DXG");
}

TEST(Graph6Serialize, Full_5) {
    auto graph = core::Graph(5);
    graph.add_edge(0, 1);
    graph.add_edge(1, 0);
    graph.add_edge(0, 2);
    graph.add_edge(2, 0);
    graph.add_edge(0, 3);
    graph.add_edge(3, 0);
    graph.add_edge(0, 4);
    graph.add_edge(4, 0);
    graph.add_edge(1, 2);
    graph.add_edge(2, 1);
    graph.add_edge(1, 3);
    graph.add_edge(3, 1);
    graph.add_edge(1, 4);
    graph.add_edge(4, 1);
    graph.add_edge(2, 3);
    graph.add_edge(3, 2);
    graph.add_edge(2, 4);
    graph.add_edge(4, 2);
    graph.add_edge(3, 4);
    graph.add_edge(4, 3);

    auto graph6 = Graph6Serializer::Serialize(graph);

    EXPECT_EQ(graph6, "D~{");
}

} // namespace core
