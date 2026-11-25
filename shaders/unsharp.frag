#version 430 core
// Unsharp mask: add back part of the detail that a blur removed. It also amplifies noise.
layout(binding = 0) uniform sampler2D uImage;
layout(binding = 1) uniform sampler2D uBlurred;
uniform float uAmount;
layout(location = 0) out float outValue;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float value = texelFetch(uImage, pixel, 0).r;
    float blurred = texelFetch(uBlurred, pixel, 0).r;
    outValue = clamp(value + uAmount * (value - blurred), 0.0, 1.0);
}
