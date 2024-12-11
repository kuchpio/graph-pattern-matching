#version 330 core

layout (location = 1) in vec2 nodePos;
layout (location = 2) in uint nodeState;
layout (location = 3) in uint nodeLabel;
out vec2 quadCoord;
out vec4 nodeColor;
flat out uint nodeBorderIndex;

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
	const vec4 nodeColorArray[9] = vec4[9] (
		vec4(0.75, 0.75, 0.75, 1.0),
		vec4(0.76, 0.29, 0.30, 1.0),
		vec4(0.44, 0.59, 0.28, 1.0),
		vec4(0.22, 0.82, 0.80, 1.0),
		vec4(0.76, 0.68, 0.28, 1.0),
		vec4(0.63, 0.30, 0.61, 1.0),
		vec4(0.41, 0.50, 0.85, 1.0),
		vec4(0.75, 0.44, 0.20, 1.0),
		vec4(0.28, 0.76, 0.57, 1.0)
	);

	quadCoord = quadCoordArray[gl_VertexID % 6];
	nodeColor = nodeColorArray[nodeLabel == 0u ? 0u : (1u + nodeLabel % 8u)];
	nodeBorderIndex = nodeState;

	vec2 ratio = (canvasSize - 3 * nodeRadius) / boundingSize;
	vec2 quadCanvasPosition = (2 * (nodePos - centerPos) * min(ratio.x, ratio.y) + quadCoord * nodeRadius) / canvasSize;
	gl_Position = vec4(quadCanvasPosition.x, quadCanvasPosition.y, 0.0, 1.0);
}
