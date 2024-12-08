#include "glad/glad.h"
#include "wx/msgdlg.h"
#include <optional>
#include <fstream>
#include <sstream>

#include "graphCanvas.h"

GraphCanvas::GraphCanvas(wxWindow* parent, const wxGLAttributes& canvasAttrs) : wxGLCanvas(parent, canvasAttrs) {
    wxGLContextAttrs ctxAttrs;
    ctxAttrs.PlatformDefaults().CoreProfile().OGLVersion(3, 3).EndList();
    openGLContext = new wxGLContext(this, nullptr, &ctxAttrs);

    if (!openGLContext->IsOK()) {
        wxMessageBox("This application needs an OpenGL 3.3 capable driver.", "OpenGL version error",
                     wxOK | wxICON_INFORMATION, this);
        delete openGLContext;
        openGLContext = nullptr;
    }

    Bind(wxEVT_PAINT, &GraphCanvas::OnPaint, this);
    Bind(wxEVT_SIZE, &GraphCanvas::OnSize, this);
}

GraphCanvas::~GraphCanvas() {
    delete openGLContext;
}

void GraphCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)) {
    if (!isOpenGLInitialized) return;
    SetCurrent(*openGLContext);

    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(edgeShaderProgram);
    glBindVertexArray(vertexArrayObject);
    glDrawElements(GL_LINES, edgesCount * 2, GL_UNSIGNED_INT, 0);

    glUseProgram(nodeShaderProgram);
    glBindVertexArray(vertexArrayObject);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, vertexCount);

    glBindVertexArray(0);

    SwapBuffers();
}

void GraphCanvas::OnSize(wxSizeEvent& event) {
    if (!isOpenGLInitializationAttempted) isOpenGLInitialized = InitializeOpenGL();

    if (!isOpenGLInitialized) return;
    SetCurrent(*openGLContext);

    auto viewPortSize = event.GetSize() * GetContentScaleFactor();
    glViewport(0, 0, viewPortSize.x, viewPortSize.y);
    SetCanvasSize(viewPortSize.GetWidth(), viewPortSize.GetHeight());
}

bool GraphCanvas::InitializeOpenGL() {
    isOpenGLInitializationAttempted = true;
    if (!openGLContext) return false;

    SetCurrent(*openGLContext);

    if (!InitializeOpenGLFunctions()) {
        wxMessageBox("Error: Could not initialize OpenGL function pointers.", "OpenGL initialization error",
                     wxOK | wxICON_INFORMATION, this);
        return false;
    }

    wxLogDebug("OpenGL version: %s", reinterpret_cast<const char*>(glGetString(GL_VERSION)));
    wxLogDebug("OpenGL vendor: %s", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));

    if (auto result = InitializeShader(nodeVertexShaderPath, nodeFragmentShaderPath)) {
        nodeShaderProgram = result.value();
    } else {
        wxMessageBox("Error: Could not initialize OpenGL shaders.", "OpenGL initialization error",
                     wxOK | wxICON_INFORMATION, this);
        return false;
    }

    if (auto result = InitializeShader(edgeVertexShaderPath, edgeFragmentShaderPath)) {
        edgeShaderProgram = result.value();
    } else {
        wxMessageBox("Error: Could not initialize OpenGL shaders.", "OpenGL initialization error",
                     wxOK | wxICON_INFORMATION, this);
        return false;
    }

    glGenVertexArrays(1, &vertexArrayObject);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &vertexStateBuffer);
    glGenBuffers(1, &edgesBuffer);
    glGenBuffers(1, &settingsUniformBufferObject);

    glBindVertexArray(vertexArrayObject);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);

    glBindBuffer(GL_ARRAY_BUFFER, vertexStateBuffer);
    glVertexAttribIPointer(2, 1, GL_UNSIGNED_INT, 0, (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgesBuffer);

    unsigned int uniformBlockIndexNode = glGetUniformBlockIndex(nodeShaderProgram, "settings");
    unsigned int uniformBlockIndexEdge = glGetUniformBlockIndex(edgeShaderProgram, "settings");
    glUniformBlockBinding(nodeShaderProgram, uniformBlockIndexNode, 0);
    glUniformBlockBinding(edgeShaderProgram, uniformBlockIndexEdge, 0);
    glBindBuffer(GL_UNIFORM_BUFFER, settingsUniformBufferObject);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(float) * (2 + 2 + 2 + 1 + 1), NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, settingsUniformBufferObject, 0, sizeof(float) * (2 + 2 + 2 + 1 + 1));

    SetNodeSize(20.0, 3.0);
    glLineWidth(2.0);

    glBindVertexArray(0);

    return true;
}

bool GraphCanvas::InitializeOpenGLFunctions() {
    auto gladVersion = gladLoadGL();

    if (0 == gladVersion) {
        wxLogError("OpenGL glad initialization failed");
        return false;
    }

    wxLogDebug("Status: Using Glad");

    return true;
}

std::optional<unsigned int> GraphCanvas::InitializeShader(const char* vertexShaderPath, const char* fragmentShaderPath) {

    std::string shaderCode;
    std::ifstream shaderFile;
    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    // Read and compile vertex shader
    try {
        shaderFile.open(vertexShaderPath);
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();
        shaderCode = shaderStream.str();
    } catch (std::ifstream::failure e) {
        wxLogDebug("Could not read vertex shader source code from %s", vertexShaderPath);
        return {};
    }
    const char* vertexShaderSource = shaderCode.c_str();

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        wxLogDebug("Vertex shader compilation failed: %s", infoLog);
        return {};
    }

    // Read and compile fragment shader
    try {
        shaderFile.open(fragmentShaderPath);
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();
        shaderCode = shaderStream.str();
    } catch (std::ifstream::failure e) {
        wxLogDebug("Could not read fragment shader source code from %s", fragmentShaderPath);
        return {};
    }
    const char* fragmentShaderSource = shaderCode.c_str();

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        wxLogDebug("Fragment shader compilation failed: %s", infoLog);
        return {};
    }

    // Link shaders
    auto shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        wxLogDebug("Shader Program Linking Failed: %s", infoLog);
        return {};
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

void GraphCanvas::SetVertexPositions(const float* positions2D, unsigned int vertexCount) {
    if (!isOpenGLInitialized) return;
    SetCurrent(*openGLContext);

    float minX = FLT_MAX, minY = FLT_MAX, maxX = FLT_MIN, maxY = FLT_MIN;
    for (unsigned int i = 0; i < vertexCount; i++) {
        float x = positions2D[2 * i];
        float y = positions2D[2 * i + 1];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    float boundingSize = std::max(maxX - minX, maxY - minY);
    SetBoundingSize(boundingSize, boundingSize);
    SetCenterPosition((minX + maxX) / 2, (minY + maxY) / 2);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCount * 2, positions2D, GL_DYNAMIC_DRAW);

    this->vertexCount = vertexCount;
}

void GraphCanvas::SetVertexStates(const unsigned int* states, unsigned int vertexCount) {
    if (!isOpenGLInitialized) return;
    SetCurrent(*openGLContext);

    glBindBuffer(GL_ARRAY_BUFFER, vertexStateBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(unsigned int) * vertexCount, states, GL_STATIC_DRAW);
}

void GraphCanvas::SetEdges(const unsigned int* edges, unsigned int edgesCount) {
    if (!isOpenGLInitialized) return;
    SetCurrent(*openGLContext);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgesBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * edgesCount * 2, edges, GL_STATIC_DRAW);

    this->edgesCount = edgesCount;
}

void GraphCanvas::SetCanvasSize(int width, int height) const {
    float canvasSize[] = { (float)width, (float)height };
    glBindBuffer(GL_UNIFORM_BUFFER, settingsUniformBufferObject);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(float) * 2, canvasSize);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void GraphCanvas::SetBoundingSize(float width, float height) const {
    float boundingSize[] = { width, height };
    glBindBuffer(GL_UNIFORM_BUFFER, settingsUniformBufferObject);
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float) * 2, sizeof(float) * 2, boundingSize);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void GraphCanvas::SetCenterPosition(float x, float y) const {
    float centerPosition[] = { x, y };
    glBindBuffer(GL_UNIFORM_BUFFER, settingsUniformBufferObject);
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float) * 4, sizeof(float) * 2, centerPosition);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void GraphCanvas::SetNodeSize(float radius, float border) const {
    float nodeSize[] = { radius, border };
    glBindBuffer(GL_UNIFORM_BUFFER, settingsUniformBufferObject);
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(float) * 6, sizeof(float) * 2, nodeSize);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}
