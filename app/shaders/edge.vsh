#version 330 core

layout (location = 0) in vec2 nodePos;

layout (std140) uniform settings
{
	vec2 canvasSize;
	vec2 boundingSize;
	vec2 centerPos;
	float nodeRadius;
	float nodeBorder;
};

void main()
{
	vec2 radiusScaled = vec2(nodeRadius, nodeRadius) / canvasSize;
	vec2 vertexOffset = 2 * (nodePos - centerPos) / (boundingSize * (1 + 2 * radiusScaled));

	gl_Position = vec4(vertexOffset.x, vertexOffset.y, 0.0, 1.0);
}
