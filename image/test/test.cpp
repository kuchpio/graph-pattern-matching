#include "image.h"

#include "gtest/gtest.h"

namespace image
{

TEST(ImageToGraphConversion, BasicTest) {
    auto path = "TODO";

    auto [graph, positions] = grapherize(path, 100);

    EXPECT_EQ(graph.size(), 100);
}

} // namespace image
