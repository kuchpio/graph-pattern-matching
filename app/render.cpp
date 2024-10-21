#include <cstdio>
#include <vector>

#define GLAD_GL_IMPLEMENTATION
#include "glad/gl.h"
#include "GLFW/glfw3.h"

#include "render.h"

GraphRenderer::GraphRenderer() : _vertexCount(0), _edgesCount(0) {
    glGenVertexArrays(1, &_vertexArrayObject);
    glGenBuffers(1, &_vertexBuffer);
    glGenBuffers(1, &_edgesBuffer);

    glPointSize(VERTEX_SIZE);
    glLineWidth(EDGE_WIDTH);
}

void GraphRenderer::setVertexPositions(const float* positions2D, unsigned int vertexCount) {
    glBindVertexArray(_vertexArrayObject);
    glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCount * 2, positions2D, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    _vertexCount = vertexCount;
}

void GraphRenderer::setEdges(const core::Graph& graph) {
    std::vector<unsigned int> edges;
    
    // TODO: Get edges info from graph
    for (int i = 0; i < graph.size() - 1; i++) {
        edges.push_back(i);
        edges.push_back(i + 1);
    }

    edges.push_back(graph.size() - 1);
    edges.push_back(0);

    glBindVertexArray(_vertexArrayObject);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _edgesBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * edges.size(), edges.data(), GL_STATIC_DRAW);

    _edgesCount = edges.size();
}

void GraphRenderer::render(int width, int height) const {
	glViewport(0, 0, width, height);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindVertexArray(_vertexArrayObject);

	glColor3f(1.0f, 1.0f, 1.0f);
	glDrawElements(GL_LINES, _edgesCount, GL_UNSIGNED_INT, 0);

	glColor3f(1.0f, 0.0f, 0.0f);
	glDrawArrays(GL_POINTS, 0, _vertexCount);

	glBindVertexArray(0);
}

GraphRenderer::~GraphRenderer() {
    glDeleteBuffers(1, &_edgesBuffer);
    glDeleteBuffers(1, &_vertexBuffer);
    glDeleteBuffers(1, &_vertexArrayObject);
}

static void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void testRender() {
    const float vertices[] = {
        -0.5f, 0.3f, 
        0.4f, 0.1f, 
        -0.7f, 0.9f, 
        0.0f, 0.3f, 
        0.2f, -0.5f
    };
    auto graph = core::Graph(5);

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) return;

    auto window = glfwCreateWindow(640, 480, "Hello GLFW!", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return;
    }

    glfwSetKeyCallback(window, key_callback);

    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);

    {
        GraphRenderer graphRenderer;
        graphRenderer.setEdges(graph);
        graphRenderer.setVertexPositions(vertices, graph.size());

        while (!glfwWindowShouldClose(window)) {
            int width, height;

            glfwGetFramebufferSize(window, &width, &height);

            graphRenderer.render(width, height);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    glfwDestroyWindow(window);

    glfwTerminate();
}
