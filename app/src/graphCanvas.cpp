#include "glad/glad.h"
#include "wx/msgdlg.h"
#include <optional>

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
    if (!isOpenGLInitializationAttempted) isOpenGLInitialized = InitializeOpenGL();
    if (!isOpenGLInitialized) return;

    SetCurrent(*openGLContext);
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(edgeShaderProgram);
    glUniform2f(glGetUniformLocation(edgeShaderProgram, "canvasSize"), canvasWidth, canvasHeight);
    glUniform2f(glGetUniformLocation(edgeShaderProgram, "boundingSize"), boundingBoxWidth, boundingBoxHeight);
    glUniform2f(glGetUniformLocation(edgeShaderProgram, "centerPos"), centerX, centerY);
    glBindVertexArray(vertexArrayObject);
    glDrawElements(GL_LINES, edgesCount * 2, GL_UNSIGNED_INT, 0);

    glUseProgram(nodeShaderProgram);
    glUniform2f(glGetUniformLocation(nodeShaderProgram, "canvasSize"), canvasWidth, canvasHeight);
    glUniform2f(glGetUniformLocation(nodeShaderProgram, "boundingSize"), boundingBoxWidth, boundingBoxHeight);
    glUniform2f(glGetUniformLocation(nodeShaderProgram, "centerPos"), centerX, centerY);
    glBindVertexArray(vertexArrayObject);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, vertexCount);

    glBindVertexArray(0);

    SwapBuffers();
}

void GraphCanvas::OnSize(wxSizeEvent& event) {
    auto viewPortSize = event.GetSize() * GetContentScaleFactor();
    canvasWidth = viewPortSize.GetWidth();
    canvasHeight = viewPortSize.GetHeight();

    if (!isOpenGLInitialized) return;

    SetCurrent(*openGLContext);
    glViewport(0, 0, viewPortSize.x, viewPortSize.y);
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

    if (auto result = InitializeShader(nodeVertexShaderSource, nodeFragmentShaderSource)) {
        nodeShaderProgram = result.value();
    } else {
        wxMessageBox("Error: Could not initialize OpenGL shaders.", "OpenGL initialization error",
                     wxOK | wxICON_INFORMATION, this);
        return false;
    }

    if (auto result = InitializeShader(edgeVertexShaderSource, edgeFragmentShaderSource)) {
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

    glLineWidth(EDGE_WIDTH);

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

std::optional<unsigned int> GraphCanvas::InitializeShader(const char* vertexShaderSource, const char* fragmentShaderSource) {
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        wxLogDebug("Vertex Shader Compilation Failed: %s", infoLog);
        return {};
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        wxLogDebug("Fragment Shader Compilation Failed: %s", infoLog);
        return {};
    }

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

    centerX = (minX + maxX) / 2;
    centerY = (minY + maxY) / 2;
    boundingBoxWidth = maxX - minX;
    boundingBoxHeight = maxY - minY;
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

char* GraphCanvas::nodeVertexShaderSource = R"(
	#version 330 core

	layout (location = 1) in vec2 nodePos;
    layout (location = 2) in uint nodeState;
    out vec2 quadCoord;
    flat out uint nodeColorIndex;

	uniform vec2 canvasSize;
	uniform vec2 boundingSize;
	uniform vec2 centerPos;
	uniform float radius = 20.0;

	void main()
	{
		const vec2 quadCoordArray[6] = vec2[6] (
			vec2(1.0, -1.0),
			vec2(1.0, 1.0),
			vec2(-1.0, 1.0),
			vec2(-1.0, 1.0),
			vec2(-1.0, -1.0),
			vec2(1.0, -1.0)
		);

		quadCoord = quadCoordArray[gl_VertexID % 6];
        nodeColorIndex = nodeState;
        vec2 radiusScaled = vec2(radius, radius) / canvasSize;
        vec2 quadOffset = quadCoord * radiusScaled;
		vec2 vertexOffset = 2 * (nodePos - centerPos) / (boundingSize * (1 + 2 * radiusScaled));

		gl_Position = vec4(vertexOffset.x + quadOffset.x, vertexOffset.y + quadOffset.y, 0.0, 1.0);
	}
)";

char* GraphCanvas::nodeFragmentShaderSource = R"(
	#version 330 core

	in vec2 quadCoord;
    flat in uint nodeColorIndex;
	out vec4 FragColor;

	uniform float radius = 20.0;
	uniform float border = 3.0;

	void main()
	{
		const vec4 nodeColorArray[2] = vec4[2] (
			vec4(0.7, 0.7, 0.7, 1.0),
			vec4(0.4, 0.4, 1.0, 1.0)
		);

		float borderThreshold = (1.0 - border / radius) * (1.0 - border / radius);
		float d = dot(quadCoord, quadCoord);
		if (d <= 1.0) {
			FragColor = d < borderThreshold ? nodeColorArray[nodeColorIndex] : vec4(0.0, 0.0, 0.0, 1.0);
		} else {
			discard;
		}
	}
)";

char* GraphCanvas::edgeVertexShaderSource = R"(
	#version 330 core

	layout (location = 0) in vec2 nodePos;

	uniform vec2 canvasSize;
	uniform vec2 boundingSize;
	uniform vec2 centerPos;
	uniform float radius = 20.0;

	void main()
	{
        vec2 radiusScaled = vec2(radius, radius) / canvasSize;
		vec2 vertexOffset = 2 * (nodePos - centerPos) / (boundingSize * (1 + 2 * radiusScaled));

		gl_Position = vec4(vertexOffset.x, vertexOffset.y, 0.0, 1.0);
	}
)";

char* GraphCanvas::edgeFragmentShaderSource = R"(
	#version 330 core

	out vec4 FragColor;

	void main()
	{
		FragColor = vec4(0.0, 0.0, 0.0, 1.0);
	}
)";
