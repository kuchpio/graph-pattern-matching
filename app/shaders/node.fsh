#version 330 core

in vec2 quadCoord;
in vec4 nodeColor;
flat in uint nodeBorderIndex;
out vec4 FragColor;

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
	float borderToRadius = nodeBorder / nodeRadius;
	float borderInnerThreshold = (1.0 - 2 * borderToRadius) * (1.0 - 2 * borderToRadius);
	float borderOuterThreshold = nodeBorderIndex > 0u ? 1.0 : (1.0 - borderToRadius) * (1.0 - borderToRadius);
	float d = dot(quadCoord, quadCoord);
	if (d <= borderOuterThreshold) {
		FragColor = d < borderInnerThreshold ? nodeColor : nodeBorderArray[nodeBorderIndex];
	} else {
		discard;
	}
}
