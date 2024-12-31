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
	vec4 nodeUtilityColor;
};

void main()
{
	vec2 nodeCanvasPosition = vec2(1.0, -1.0) * 2 * (nodePos / canvasSize - 0.5);
	gl_Position = vec4(nodeCanvasPosition.x, nodeCanvasPosition.y, 0.0, 1.0);
}
