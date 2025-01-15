#pragma once

#include <vector>
#include <filesystem>

#include "core.h"

namespace image
{

std::pair<core::Graph, std::vector<std::pair<float, float>>> grapherize(std::filesystem::path modulePath,
                                                                        const std::string& imagePath, int vertexCount,
                                                                        bool graph = true);

}
