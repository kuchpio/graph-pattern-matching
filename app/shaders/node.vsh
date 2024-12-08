#version 330 core

layout (location = 1) in vec2 nodePos;
layout (location = 2) in uint nodeState;
out vec2 quadCoord;
flat out uint nodeColorIndex;

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
	vec2 radiusScaled = vec2(nodeRadius, nodeRadius) / canvasSize;
	vec2 quadOffset = quadCoord * radiusScaled;
	vec2 vertexOffset = 2 * (nodePos - centerPos) / (boundingSize * (1 + 2 * radiusScaled));

	gl_Position = vec4(vertexOffset.x + quadOffset.x, vertexOffset.y + quadOffset.y, 0.0, 1.0);
}
