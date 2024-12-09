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

	vec2 ratio = (canvasSize - 3 * nodeRadius) / boundingSize;
	vec2 quadCanvasPosition = (2 * (nodePos - centerPos) * min(ratio.x, ratio.y) + quadCoord * nodeRadius) / canvasSize;
	gl_Position = vec4(quadCanvasPosition.x, quadCanvasPosition.y, 0.0, 1.0);
}
