#include "core.h"

namespace utils
{
class GraphFactory {
  private:
  public:
    static core::Graph isomoporhic_graph(const core::Graph& G);
    static core::Graph random_graph(int n, float edge_propability);
};

} // namespace utils