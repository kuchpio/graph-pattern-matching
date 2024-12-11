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

	float borderThreshold = (1.0 - nodeBorder / nodeRadius) * (1.0 - nodeBorder / nodeRadius);
	float d = dot(quadCoord, quadCoord);
	if (d <= 1.0) {
		FragColor = d < borderThreshold ? nodeColor : nodeBorderArray[nodeBorderIndex];
	} else {
		discard;
	}
}
