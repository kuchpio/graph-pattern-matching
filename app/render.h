#pragma once

#include "core.h"

class GraphRenderer {
  private:
    const float VERTEX_SIZE = 8.0f;
    const float EDGE_WIDTH = 2.0f;

    unsigned int _vertexArrayObject;
    unsigned int _vertexBuffer;
    unsigned int _edgesBuffer;
    unsigned int _vertexCount;
    unsigned int _edgesCount;

  public:
    GraphRenderer();
    void setVertexPositions(const float *positions2D, unsigned int vertexCount);
    void setEdges(const core::Graph& graph);
    void render(int width, int height) const;
    ~GraphRenderer();
};

void testRender();
