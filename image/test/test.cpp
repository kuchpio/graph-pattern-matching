#include "image.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace image
{

class ImageToGraphTest : public ::testing::Test {
  protected:
    std::filesystem::path modulePath;
    std::string testImagePath;

    void SetUp() override {
        modulePath = "../";
        testImagePath = modulePath.string() + "edge_detection/image.png";
    }
};

TEST_F(ImageToGraphTest, BasicGraphTriangulationTest) {
    int vertexCount = 100;
    ASSERT_TRUE(std::filesystem::exists(testImagePath)) << "Test image does not exist";

    auto [graph, positions] = grapherize(modulePath, testImagePath, vertexCount, false);

    EXPECT_EQ(graph.size(), vertexCount);

    EXPECT_EQ(positions.size(), vertexCount);

    for (const auto& [x, y] : positions) {
        EXPECT_GE(x, 0.0);
        EXPECT_GE(y, 0.0);
    }
}

TEST_F(ImageToGraphTest, BasicGraphForGraphTest) {
    int vertexCount = 100;
    ASSERT_TRUE(std::filesystem::exists(testImagePath)) << "Test image does not exist";

    auto [graph, positions] = grapherize(modulePath, testImagePath, vertexCount, true);

    EXPECT_EQ(graph.size(), vertexCount);

    EXPECT_EQ(positions.size(), vertexCount);

    for (const auto& [x, y] : positions) {
        EXPECT_GE(x, 0.0);
        EXPECT_GE(y, 0.0);
    }
}

TEST_F(ImageToGraphTest, InvalidImagePathTest) {
    std::string invalidPath = "/invalid/path/to/image.png";
    int vertexCount = 100;

    EXPECT_THROW(grapherize(modulePath, invalidPath, vertexCount), std::runtime_error);
}

TEST_F(ImageToGraphTest, ZeroVertexCountTest) {
    ASSERT_TRUE(std::filesystem::exists(testImagePath)) << "Test image does not exist";

    int vertexCount = 0;
    EXPECT_THROW(grapherize(modulePath, testImagePath, vertexCount), std::runtime_error);
}

} // namespace image
