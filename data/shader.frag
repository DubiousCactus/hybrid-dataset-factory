#version 330

uniform vec3 Color;
uniform vec3 Light1;
uniform vec3 Light2;
uniform vec3 Light3;
uniform vec3 Light4;
uniform vec3 viewPos;
uniform bool UseTexture;
uniform sampler2D Texture;

in vec3 v_vert;
in vec3 v_norm;
in vec2 v_text;

out vec4 f_color;

void main() {
	float ambientStrength = 0.5;
	float specularStrength = 0.5;
	int shininess = 64;

	vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);
	vec3 lightDir = normalize(Light1 - v_vert);
	vec3 viewDir = normalize(viewPos - v_vert);
	vec3 reflectDir = reflect(-lightDir, normalize(v_norm));

	vec3 ambient = lightColor * ambientStrength;

	float diffuse = clamp(dot(normalize(v_norm), lightDir), 0.0, 1.0);

	float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
	vec3 specular = specularStrength * spec * lightColor;

	vec3 combined = (ambient + diffuse + specular);

	f_color = UseTexture ? vec4(texture(Texture, v_text).rgb * combined, 1.0)
		: vec4(Color * combined, 1.0);
}
