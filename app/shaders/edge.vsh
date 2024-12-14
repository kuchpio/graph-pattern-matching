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

layout (std140) uniform colors
{
	vec4 nodeBorderArray[4];
	vec4 nodeColorArray[9];
};

void main()
{
	vec2 ratio = (canvasSize - 3 * nodeRadius) / boundingSize;
	vec2 nodeCanvasPosition = 2 * (nodePos - centerPos) * min(ratio.x, ratio.y) / canvasSize;
	gl_Position = vec4(nodeCanvasPosition.x, nodeCanvasPosition.y, 0.0, 1.0);
}
