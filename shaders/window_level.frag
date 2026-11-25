#version 430 core
// Window/level: choose which intensity range is spread from black to white.
// The sigmoid curve is the soft version used by DICOM viewers: it never clips completely.
layout(binding = 0) uniform sampler2D uImage;
uniform float uLevel;
uniform float uWidth;
uniform int uCurve;  // 0 = linear, 1 = sigmoid
layout(location = 0) out float outValue;

void main() {
    float value = texelFetch(uImage, ivec2(gl_FragCoord.xy), 0).r;
    float t = (value - uLevel) / max(uWidth, 1e-6);
    outValue = uCurve == 1 ? 1.0 / (1.0 + exp(-4.0 * t)) : clamp(t + 0.5, 0.0, 1.0);
}
