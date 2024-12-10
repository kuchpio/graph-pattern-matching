#pragma once

#include "wx/glcanvas.h"

class GraphCanvas : public wxGLCanvas {
    const char* nodeVertexShaderPath = "node.vsh";
    const char* nodeFragmentShaderPath = "node.fsh";
    const char* edgeVertexShaderPath = "edge.vsh";
    const char* edgeFragmentShaderPath = "edge.fsh";

    wxGLContext* openGLContext;
    bool isOpenGLInitialized = false, isOpenGLInitializationAttempted = false;
    wxSize viewPortSize;

    unsigned int vertexArrayObject = 0;
    unsigned int vertexBuffer = 0;
    unsigned int vertexStateBuffer = 0;
    unsigned int edgesBuffer = 0;
    unsigned int nodeShaderProgram = 0;
    unsigned int edgeShaderProgram = 0;
    unsigned int settingsUniformBufferObject = 0;

    unsigned int vertexCount = 0;
    unsigned int edgesCount = 0;

    bool InitializeOpenGLFunctions();
    bool InitializeOpenGL();
    std::optional<unsigned int> InitializeShader(const char* vertexShaderPath,
                                                              const char* fragmentShaderPath);
    void SetNodeSize(float radius, float border) const;

    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);
    void UpdateCanvasSize() const;

  public:
    const float NODE_RADIUS = 20.0;
    const float NODE_BORDER = 3.0;
    const float EDGE_WIDTH = 2.0;

    GraphCanvas(wxWindow* parent, const wxGLAttributes& canvasAttrs);
    ~GraphCanvas();

    void SetVertexPositions(const float* positions2D, unsigned int vertexCount);
    void SetVertexStates(const unsigned int* states, unsigned int vertexCount);
    void SetEdges(const unsigned int* edges, unsigned int edgesCount);
    const std::pair<int, int> CanvasSize() const;
    void SetBoundingSize(float width, float height) const;
    void SetCenterPosition(float x, float y) const;
};
