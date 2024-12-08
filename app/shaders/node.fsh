#version 330 core

in vec2 quadCoord;
flat in uint nodeColorIndex;
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
	const vec4 nodeColorArray[2] = vec4[2] (
		vec4(0.7, 0.7, 0.7, 1.0),
		vec4(0.4, 0.4, 1.0, 1.0)
	);

	float borderThreshold = (1.0 - nodeBorder / nodeRadius) * (1.0 - nodeBorder / nodeRadius);
	float d = dot(quadCoord, quadCoord);
	if (d <= 1.0) {
		FragColor = d < borderThreshold ? nodeColorArray[nodeColorIndex] : vec4(0.0, 0.0, 0.0, 1.0);
	} else {
		discard;
	}
}
