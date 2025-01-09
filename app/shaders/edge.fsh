#version 330 core

out vec4 FragColor;

layout (std140) uniform colors
{
	vec4 nodeBorderArray[4];
	vec4 nodeColorArray[9];
	vec4 nodeUtilityColor;
};

void main()
{
	FragColor = nodeBorderArray[0];
}
