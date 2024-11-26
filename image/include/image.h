#pragma once

#include <vector>

#include "core.h"

namespace image
{

std::pair<core::Graph, std::vector<std::pair<float, float>>> grapherize(const std::string& imagePath, int vertexCount);

}
