#version 330 core

// INPUTS
in vec2 fragTexCoord;
out vec4 finalColor;

// --- CONFIGURATION ---
#define MAX_PTS 64
#define EPSILON 1e-6

// --- UNIFORMS ---
uniform vec2 uMouthPts[MAX_PTS];
uniform int uMouthCount;

uniform vec2 uTopTeethPts[MAX_PTS];
uniform int uTopTeethCount;

uniform vec2 uBotTeethPts[MAX_PTS];
uniform int uBotTeethCount;

uniform vec2 uTonguePts[MAX_PTS];
uniform int uTongueCount;

uniform vec2 uResolution; 
uniform float uPadding; 
uniform float uScale;    // [NEW] To scale strokes/rounding properly
uniform float uOutlineThickness;
uniform vec4 uColBg;
uniform vec4 uColLine;
uniform vec4 uColTeeth;
uniform vec4 uColTongue;

// --- ROBUST SDF FUNCTION ---
float sdPoly(vec2 p, vec2 pts[MAX_PTS], int count) {
    if (count < 3) return 1000.0;
    
    float d = 1e10; 
    bool inside = false;
    
    for (int i = 0; i < count; i++) {
        int j = (i + 1) % count;
        vec2 e = pts[j] - pts[i];
        vec2 w = p - pts[i];
        
        float ee = dot(e,e);
        float h = (ee > EPSILON) ? clamp(dot(w, e) / ee, 0.0, 1.0) : 0.0;
        vec2 b = w - e * h;
        d = min(d, dot(b, b));
        
        bool cond1 = (pts[i].y > p.y);
        bool cond2 = (pts[j].y > p.y);
        
        if (cond1 != cond2) {
            float dy = pts[j].y - pts[i].y;
            if (abs(dy) > EPSILON) {
                float crossX = (pts[j].x - pts[i].x) * (p.y - pts[i].y) / dy + pts[i].x;
                if (p.x < crossX) inside = !inside;
            }
        }
    }
    float sign = inside ? -1.0 : 1.0;
    return sign * sqrt(d);
}

float getAlpha(float d) { return 1.0 - smoothstep(-0.5, 0.5, d); }
float getStroke(float d, float thick) { return 1.0 - smoothstep(thick - 0.5, thick + 0.5, abs(d)); }

void main() {
    vec2 pixelPos = fragTexCoord * uResolution;
    
    // 1. MOUTH SHAPE
    float dMouth = sdPoly(pixelPos, uMouthPts, uMouthCount);
    
    if (dMouth > uPadding) discard;

    float mouthAlpha = getAlpha(dMouth);
    vec4 color = mix(vec4(0.0), uColBg, mouthAlpha);

    // 2. INTERNALS
    if (mouthAlpha > 0.01) {
        if (uTongueCount >= 3) {
            float dTongue = sdPoly(pixelPos, uTonguePts, uTongueCount);
            color = mix(color, uColTongue, getAlpha(dTongue) * mouthAlpha);
        }

        float teethThick = 1.5 * uScale; // Scale the separation line too
        
        // [FIX] TEETH ROUNDING
        // Subtracting a radius (e.g. 3.0 * scale) expands the shape 
        // and creates rounded corners automatically.
        float roundRadius = 3.0 * uScale; 

        if (uTopTeethCount >= 3) {
            float dTop = sdPoly(pixelPos, uTopTeethPts, uTopTeethCount);
            // Use (dTop - roundRadius) to render an expanded, rounded shape
            color = mix(color, uColTeeth, getAlpha(dTop - roundRadius) * mouthAlpha);
            color = mix(color, uColLine, getStroke(dTop - roundRadius, teethThick) * mouthAlpha);
        }

        if (uBotTeethCount >= 3) {
            float dBot = sdPoly(pixelPos, uBotTeethPts, uBotTeethCount);
            color = mix(color, uColTeeth, getAlpha(dBot - roundRadius) * mouthAlpha);
            color = mix(color, uColLine, getStroke(dBot - roundRadius, teethThick) * mouthAlpha);
        }
    }

    // 3. OUTLINE
    // [FIX] Scale the outline thickness so it stays proportional
    // Base thickness 4.0 * Scale
    color = mix(color, uColLine, getStroke(dMouth, uOutlineThickness * uScale));

    if (color.a < 0.01) discard;
    finalColor = color;
}