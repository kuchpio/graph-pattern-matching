#include "core.h"

namespace core
{
Graph::Graph(int size) {
    _size = size;
}

int Graph::size() const {
    return _size;
}
} // namespace core
