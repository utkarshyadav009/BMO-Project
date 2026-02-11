#version 300 es
precision highp float;

in vec2 fragTexCoord;
in vec4 fragColor;
out vec4 finalColor;

uniform sampler2D texture0;

void main() {
    // Calculate the distance from the center of the glyph edge (0.5)
    float distanceFromEdge = texture(texture0, fragTexCoord).a;
    
    // Adjust smoothing based on the derivative (keeps it sharp at any zoom)
    float smoothing = fwidth(distanceFromEdge);
    float alpha = smoothstep(0.5 - smoothing, 0.5 + smoothing, distanceFromEdge);

    // Apply the alpha to the font color
    if (alpha < 0.01) discard;
    finalColor = vec4(fragColor.rgb, fragColor.a * alpha);
}