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

void main()
{
	const vec4 nodeBorderArray[4] = vec4[4] (
		vec4(0.0, 0.0, 0.0, 1.0),
		vec4(0.0, 0.0, 0.6, 1.0),
		vec4(0.6, 0.0, 0.0, 1.0),
		vec4(0.3, 0.0, 0.3, 1.0)
	);

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
