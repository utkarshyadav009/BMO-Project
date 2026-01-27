#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform vec4 uColor; // Typically Black

// --- BROW PARAMS ---
uniform float uEyebrowType;  // 0=None, 1=Angry, 2=Sad, 3=Raised
uniform float uBend;         // Curve intensity
uniform float uThickness;    
uniform float uEyeBrowLength;

// UTILS
vec2 rotate2D(vec2 p, float a) { 
    float s = sin(a), c = cos(a); 
    return vec2(c*p.x - s*p.y, s*p.x + c*p.y); 
}
float sdBox(vec2 p, vec2 b) { 
    vec2 d = abs(p) - b; 
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); 
}

void main() {
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    float r = 0.5 * min(uResolution.x, uResolution.y); // Use same scale ref as eye
    float th = max(uThickness, 1.0);

    // 1. ROTATION (Expression)
    float mode = uEyebrowType;
    float angle = (mode < 1.5) ? 0.45 : (mode < 2.5) ? -0.45 : 0.0;
    vec2 br = rotate2D(p, angle);

    // 2. BEND (Curve)
    // Parabolic bending
    float nx = br.x / r;
    br.y += (nx * nx) * (uBend * 0.5) * r; 

    // 3. SHAPE
    // Rounded Box
    vec2 size = vec2(r * uEyeBrowLength, th * 0.85);
    vec2 d = abs(br) - size;
    float cornerRadius = th * 0.5;
    float dist = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - cornerRadius;

    // 4. RENDER
    float alpha = 1.0 - smoothstep(-0.75, 0.75, dist);
    
    if (alpha < 0.01) discard;
    finalColor = vec4(uColor.rgb, alpha * uColor.a);
}