#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;

// --- TEAR PARAMS ---
uniform float uTearsLevel;   // 0.0-0.4: Triangle, 0.4-0.75: Drip, 0.75+: Wail
uniform int uTearMode;       // (Optional override)
uniform float uTime; 
uniform float uSide;       // 1.0 for left eye, -1.0 for right eye
// --- BLUSH PARAMS ---
uniform int uShowBlush;
uniform vec4 uBlushColor;    // Pink
uniform vec4 uTearColor;     // Blue (Default)

// UTILS
float hash(float n) { return fract(sin(n) * 43758.5453123); }

// Signed Distance to an Equilateral Triangle
// r = approx "radius" size of the triangle
float sdTriangle(vec2 p, float r) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x + k*p.y > 0.0 ) p = vec2(p.x - k*p.y, -k*p.x - p.y) / 2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p) * sign(p.y);
}

float sdTears(vec2 p, float level, float t) {
    float d = 1e5;
    
    // --- MODE 1: TRIANGLE TEARS (Green Anime Style) ---
    // Trigger: Low levels (0.0 - 0.4)
    if (level < 0.4) {
        float size = 35.0; 
        float speed = 0.2 + (level * 0.5);
        float fallDist = uResolution.y; 
        
        for(float i = 0.0; i < 5.0; i++) {
            // FIX 1: SPREAD THEM OUT
            // Instead of 0.1, we use roughly 1/3 (0.33)
            // This ensures one is at the top, one middle, one bottom.
            float offset = i / 5.0; 
            
            float cycle = fract((t * speed) + offset);
            
            // Position
            float yPos = 20.0 + (cycle * fallDist); 

            vec2 dropPos = p - vec2(0.0, yPos);
            dropPos.y = -dropPos.y; // Flip Y correction
            
            float dTri = sdTriangle(dropPos, size);
            d = min(d, dTri);
        }
    }
    
    // --- MODE 2: CLASSIC DRIP ---
    // Trigger: Mid levels (0.4 - 0.75)
    else if (level < 0.75) {
        float width = 10.0 + (level * 15.0);
        // Normalize level for this range so density works correctly
        float localLevel = (level - 0.4) * 2.5; 
        
        float density = mix(1.0, 4.0, localLevel); 
        
        for(float i = 0.0; i < 4.0; i++) {
            if (i > density) break;
            float speed = 2.0 + (localLevel * 2.0);
            float offset = i * 1.23; 
            float localTime = t * speed + offset;
            float cycleT = fract(localTime); 
            float rnd = hash(floor(localTime) + i * 13.0);
            float xOff = (rnd - 0.5) * 20.0 * localLevel;
            float yPos = -50.0 - (cycleT * cycleT * 300.0); 
            
            vec2 head = vec2(xOff, yPos);
            float dHead = length(p - head) - (width * 0.8);
            
            vec2 pa = p - vec2(xOff, -20.0);
            vec2 ba = head - vec2(xOff, -20.0);
            float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
            float dTrail = length(pa - ba * h) - (width * mix(0.2, 0.8, h));
            d = min(d, min(dHead, dTrail));
        }
    } 
    
    // --- MODE 3: WAIL STREAM ---
    // Trigger: High levels (0.75 - 1.0)
    else { 
        float width = 15.0 + (level * 10.0);
        float wave = sin(p.y * 0.05 - t * 8.0) * 5.0; 
        float xDist = abs(p.x + wave) - width * 0.6;
        float dropOff = smoothstep(0.0, -300.0, p.y); 
        d = xDist + (1.0 - dropOff) * 100.0; 
    }

    return d;
}

void main() {
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    float r = 0.4 * min(uResolution.x, uResolution.y);

    vec3 colorAccum = vec3(0.0);
    float alphaAccum = 0.0;

    // 1. BLUSH PASS
    if (uShowBlush == 1) {
        float dBlush = length(p - vec2(0.0, -r*0.5)) - r*0.6;
        float a = 1.0 - smoothstep(0.0, r, dBlush + r*0.2);
        a *= 0.5; // Opacity
        
        colorAccum = uBlushColor.rgb;
        alphaAccum = a;
    }

    // 2. TEAR PASS
    if (uTearsLevel > 0.01) {
        vec2 pTear = p - vec2(-160*uSide, -r * 1.5); // Start tears lower
        
        float dTear = sdTears(pTear, uTearsLevel, uTime);
        
        // --- COLOR SELECTION LOGIC ---
        // Default Blue
        vec3 activeTearColor = uTearColor.rgb; 
        
        // If in Triangle Mode (Level < 0.4), force GREEN
        if (uTearsLevel < 0.4) {
            activeTearColor = vec3(0.18039, 0.71373, 0.38431);
        }

        // Render shape
        float a = 1.0 - smoothstep(-0.75, 0.75, dTear);
        
        // Composite
        colorAccum = mix(colorAccum, activeTearColor, a);
        alphaAccum = max(alphaAccum, a);
    }

    if (alphaAccum < 0.01) discard;
    finalColor = vec4(colorAccum, alphaAccum);
}