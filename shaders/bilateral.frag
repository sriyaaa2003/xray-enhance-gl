#version 430 core
// Bilateral filter: average neighbours that are close in space AND similar in brightness,
// so noise is smoothed while strong edges (for example bone outlines) are mostly left alone.
layout(binding = 0) uniform sampler2D uImage;
uniform int uRadius;
uniform float uSigmaSpace;
uniform float uSigmaRange;
layout(location = 0) out float outValue;

void main() {
    ivec2 last = textureSize(uImage, 0) - 1;
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float centre = texelFetch(uImage, pixel, 0).r;
    float sum = 0.0;
    float weights = 0.0;
    for (int dy = -uRadius; dy <= uRadius; ++dy) {
        for (int dx = -uRadius; dx <= uRadius; ++dx) {
            float value = texelFetch(uImage, clamp(pixel + ivec2(dx, dy), ivec2(0), last), 0).r;
            float difference = value - centre;
            float w = exp(-float(dx * dx + dy * dy) / (2.0 * uSigmaSpace * uSigmaSpace)
                          - difference * difference / (2.0 * uSigmaRange * uSigmaRange));
            sum += w * value;
            weights += w;
        }
    }
    outValue = sum / weights;
}
