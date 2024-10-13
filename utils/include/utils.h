#include "core.h"
#include <vector>

namespace utils
{
class GraphFactory {
  private:
  public:
    static core::Graph isomoporhic_graph(const core::Graph& G);
    static core::Graph random_graph(int n, float edge_propability);
    static std::vector<core::Graph> divide_into_components(const core::Graph& G);
};

} // namespace utils