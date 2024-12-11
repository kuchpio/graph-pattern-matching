#include "image.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifndef EDGE_DETECTION_DIR
#define EDGE_DETECTION_DIR "./"
#endif

namespace image
{

class ImageToGraphTest : public ::testing::Test {
  protected:
    std::string testImagePath;

    void SetUp() override {
        testImagePath = std::string(EDGE_DETECTION_DIR) + "/beagle.png";

        std::ofstream imageFile(testImagePath, std::ios::binary);
        if (!imageFile) {
            throw std::runtime_error("Failed to create test image file");
        }

        imageFile.close();
    }

    void TearDown() override {
        std::filesystem::remove(testImagePath);
    }
};

TEST_F(ImageToGraphTest, BasicGraphTest) {
    int vertexCount = 100;
    ASSERT_TRUE(std::filesystem::exists(testImagePath)) << "Test image does not exist";

    auto [graph, positions] = grapherize(testImagePath, vertexCount);

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

    EXPECT_THROW(grapherize(invalidPath, vertexCount), std::runtime_error);
}

TEST_F(ImageToGraphTest, ZeroVertexCountTest) {
    ASSERT_TRUE(std::filesystem::exists(testImagePath)) << "Test image does not exist";

    int vertexCount = 0;
    EXPECT_THROW(grapherize(testImagePath, vertexCount), std::runtime_error);

}

} // namespace image
