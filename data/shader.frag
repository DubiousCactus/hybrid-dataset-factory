#version 330

uniform vec3 Color;
uniform vec3 Light1;
uniform vec3 Light2;
uniform vec3 Light3;
uniform vec3 Light4;
uniform bool UseTexture;
uniform sampler2D Texture;

in vec3 v_vert;
in vec3 v_norm;
in vec2 v_text;

out vec4 f_color;

void main() {
	float diffuse = clamp(dot(normalize(Light1 - v_vert),
				normalize(v_norm)), 0.0, 1.0);
	/*lum = lum + clamp(dot(normalize(Light2 - v_vert),
				normalize(v_norm)), 0.0, 1.0);
	lum = lum + clamp(dot(normalize(Light3 - v_vert),
				normalize(v_norm)), 0.0, 1.0);
	lum = lum + clamp(dot(normalize(Light4 - v_vert),
				normalize(v_norm)), 0.0, 1.0);
	lum = lum * 0.8 + 0.2;*/
	float ambientFactor = 0.5;
	vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);
	vec3 ambient = lightColor * ambientFactor;
	vec3 diffuseAmbient = (ambient + diffuse);
	f_color = UseTexture ? vec4(texture(Texture, v_text).rgb * ambient, 1.0)
		: vec4(Color * diffuseAmbient, 1.0);
}
