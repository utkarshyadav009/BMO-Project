#version 330

in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;

uniform sampler2D texture0;
uniform vec2 renderSize;
uniform float pixelSize;

void main() {
    // 1. Bypass if pixelation is disabled (1.0 or less)
    if (pixelSize <= 1.0) {
        finalColor = texture(texture0, fragTexCoord) * fragColor;
        return;
    }

    // 2. Calculate Grid
    vec2 pixels = renderSize / pixelSize;
    
    // 3. Snap UV to Grid
    vec2 coord = floor(fragTexCoord * pixels) / pixels;
    
    // 4. Center Sample (Fixes jitter/bleeding)
    coord += (1.0 / pixels) * 0.5; 

    finalColor = texture(texture0, coord) * fragColor;
}