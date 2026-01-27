#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;

// --- TEAR PARAMS ---
uniform float uTearsLevel;   // 0-1
uniform int uTearMode;       // 0=Drip, 1=Wail
uniform float uTime; 

// --- BLUSH PARAMS ---
uniform int uShowBlush;
uniform vec4 uBlushColor;    // Pink
uniform vec4 uTearColor;     // Blue

// UTILS
float hash(float n) { return fract(sin(n) * 43758.5453123); }

float sdTears(vec2 p, float level, int mode, float t) {
    float width = 10.0 + (level * 15.0);
    float d = 1e5;
    
    if (mode == 1) { // WAIL (Thick Stream)
        float wave = sin(p.y * 0.05 - t * 8.0) * 5.0; 
        float xDist = abs(p.x + wave) - width * 0.6;
        float dropOff = smoothstep(0.0, -300.0, p.y); 
        d = xDist + (1.0 - dropOff) * 100.0; 
    } 
    else { // DRIP
        float density = mix(1.0, 4.0, level); 
        for(float i = 0.0; i < 4.0; i++) {
            if (i > density) break;
            float speed = 2.0 + (level * 2.0);
            float offset = i * 1.23; 
            float localTime = t * speed + offset;
            float cycleT = fract(localTime); 
            float rnd = hash(floor(localTime) + i * 13.0);
            float xOff = (rnd - 0.5) * 20.0 * level;
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

    // 2. TEAR PASS (Overlays Blush)
    if (uTearsLevel > 0.01) {
        vec2 pTear = p - vec2(0.0, -r * 0.8); // Start tears lower
        float dTear = sdTears(pTear, uTearsLevel, uTearMode, uTime);
        float a = 1.0 - smoothstep(-0.75, 0.75, dTear);
        
        colorAccum = mix(colorAccum, uTearColor.rgb, a);
        alphaAccum = max(alphaAccum, a);
    }

    if (alphaAccum < 0.01) discard;
    finalColor = vec4(colorAccum, alphaAccum);
}