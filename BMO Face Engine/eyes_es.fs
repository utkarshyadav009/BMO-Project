#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform vec4 uColor;

// --- EYE PARAMETERS ---
uniform float uShapeID;   // 0=Circle, 1=Star, 2=Heart, 3=Spiral
uniform float uMorph;     // 0.0 to 1.0 (Blend between Circle and ShapeID)
uniform float uPupilSize; // 0.0 to 1.0
uniform float uEyelidTop; // 0.0 (Open) to 1.0 (Closed/Angry)
uniform float uEyelidBot; // 0.0 (Open) to 1.0 (Closed/Sad)

// --- SDF FUNCTIONS ---
float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

// 5-Pointed Star SDF
float sdStar5(in vec2 p, in float r, in float rf) {
    const vec2 k1 = vec2(0.809016994375, -0.587785252292);
    const vec2 k2 = vec2(-k1.x, k1.y);
    p.x = abs(p.x);
    p -= 2.0 * max(dot(k1, p), 0.0) * k1;
    p -= 2.0 * max(dot(k2, p), 0.0) * k2;
    p.x = abs(p.x);
    p.y -= r;
    vec2 ba = rf * vec2(-k1.y, k1.x) - vec2(0, 1);
    float h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);
    return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

// Heart SDF
float sdHeart(in vec2 p) {
    p.x = abs(p.x);
    if (p.y + p.x > 1.0) return sqrt(dot(p - vec2(0.25, 0.75), p - vec2(0.25, 0.75))) - sqrt(2.0) / 4.0;
    return sqrt(min(dot(p - vec2(0.00, 1.00), p - vec2(0.00, 1.00)),
                    dot(p - 0.5 * max(p.x + p.y, 0.0), p - 0.5 * max(p.x + p.y, 0.0))));
}

// Spiral (Pseudo-SDF based on polar stripes)
float sdSpiral(vec2 p, float r) {
    float len = length(p);
    float ang = atan(p.y, p.x);
    float spiral = sin(len * 20.0 + ang * 5.0); // Wavy pattern
    float dCircle = len - r;
    // Mix the spiral pattern inside the circle
    return max(dCircle, -spiral * 0.1); 
}

float getAlpha(float d) { return 1.0 - smoothstep(-2.0, 2.0, d); }

void main() {
    // Center coordinates (0,0 in middle)
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    float baseRadius = min(uResolution.x, uResolution.y) * 0.4;
    
    // 1. CALCULATE BASE SHAPES
    float dCircle = sdCircle(p, baseRadius);
    float dTarget = dCircle; // Default
    
    // 2. MORPH TARGET
    // Using hard ifs since uShapeID is discrete, but mixing results
    if (uShapeID > 0.5 && uShapeID < 1.5) {
        // Star
        dTarget = sdStar5(p, baseRadius, baseRadius * 0.5);
    } 
    else if (uShapeID > 1.5 && uShapeID < 2.5) {
        // Heart (Heart SDF needs scaling)
        vec2 hp = p / (baseRadius * 1.5);
        hp.y += 0.5; // Center it
        dTarget = sdHeart(hp) * baseRadius;
    }
    else if (uShapeID > 2.5) {
        // Spiral
        dTarget = sdSpiral(p, baseRadius);
    }
    
    // 3. BLEND
    float dFinal = mix(dCircle, dTarget, uMorph);
    
    // 4. EYELIDS (Subtraction)
    // Top Eyelid (Angry)
    float dTopLid = (p.y - (baseRadius * (1.0 - uEyelidTop * 2.0))) * -1.0;
    dFinal = max(dFinal, dTopLid);
    
    // Bottom Eyelid (Sad/Squint)
    float dBotLid = (p.y + (baseRadius * (1.0 - uEyelidBot * 2.0)));
    dFinal = max(dFinal, dBotLid);

    // 5. PUPIL (Subtraction)
    // Only valid for standard shapes, creates a hole in the center
    if (uPupilSize > 0.01 && uShapeID < 2.5) {
        float dPupil = sdCircle(p, baseRadius * uPupilSize * 0.4);
        dFinal = max(dFinal, -dPupil); // Subtract pupil
    }

    vec4 color = uColor;
    color.a *= getAlpha(dFinal);
    
    finalColor = color;
}