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

layout (std140) uniform colors
{
	vec4 nodeBorderArray[4];
	vec4 nodeColorArray[9];
	vec4 nodeUtilityColor;
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
	nodeColor = nodeColorArray[nodeLabel == 0u ? 0u : (1u + nodeLabel % 8u)];
	nodeBorderIndex = nodeState;

	vec2 ratio = (canvasSize - 3 * nodeRadius) / boundingSize;
	vec2 quadCanvasPosition = (2 * (nodePos - centerPos) * min(ratio.x, ratio.y) + quadCoord * nodeRadius) / canvasSize;
	gl_Position = vec4(quadCanvasPosition.x, quadCanvasPosition.y, 0.0, 1.0);
}
