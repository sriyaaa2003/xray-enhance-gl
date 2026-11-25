#version 430 core
// One direction of a Gaussian blur. Running it across and then down equals a full 2D blur, far cheaper.
layout(binding = 0) uniform sampler2D uImage;
uniform ivec2 uDirection;
uniform int uRadius;
uniform float uSigma;
layout(location = 0) out float outValue;

void main() {
    ivec2 last = textureSize(uImage, 0) - 1;
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float sum = 0.0;
    float weights = 0.0;
    for (int k = -uRadius; k <= uRadius; ++k) {
        float w = exp(-float(k * k) / (2.0 * uSigma * uSigma));
        sum += w * texelFetch(uImage, clamp(pixel + k * uDirection, ivec2(0), last), 0).r;
        weights += w;
    }
    outValue = sum / weights;
}
