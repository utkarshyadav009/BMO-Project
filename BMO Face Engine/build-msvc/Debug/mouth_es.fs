#version 330 core

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
uniform float uScale;    
uniform float uOutlineThickness;
uniform vec4 uColBg;
uniform vec4 uColLine;
uniform vec4 uColTeeth;
uniform vec4 uColTongue;

// --- SDF FUNCTIONS ---

// 1. Polygon (For Mouth & Tongue)
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

// 2. Rounded Box (For Perfect Pill Teeth)
float sdRoundedBox(vec2 p, vec2 b, vec4 r) {
    r.xy = (p.x > 0.0) ? r.xy : r.zw;
    r.x  = (p.y > 0.0) ? r.x  : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}

float getAlpha(float d) { return 1.0 - smoothstep(-0.5, 0.5, d); }
float getStroke(float d, float thick) { return 1.0 - smoothstep(thick - 0.5, thick + 0.5, abs(d)); }

void main() {
    vec2 pixelPos = fragTexCoord * uResolution;
    
    // 1. MOUTH SHAPE
    float dMouth = sdPoly(pixelPos, uMouthPts, uMouthCount);
    float actualMouthOutline = uOutlineThickness * uScale;
    
    // Performance Discard
    if (dMouth > actualMouthOutline + 2.0) discard;

    float mouthAlpha = getAlpha(dMouth);
    vec4 color = mix(vec4(0.0), uColBg, mouthAlpha);

    if (mouthAlpha > 0.01) {
        
        // --- INTERSECTION CLIPPER ---
        // This is the "Inner Wall" of the mouth.
        // The black outline extends inwards by half its thickness.
        // We set the clip wall 0.5px deeper to prevent any white bleeding.
        float dMouthInner = dMouth - (-(actualMouthOutline * 0.5) + 0.5);

        // --- TONGUE ---
        if (uTongueCount >= 3) {
            float dTongue = sdPoly(pixelPos, uTonguePts, uTongueCount);
            // Clip tongue to inner wall
            float dTongueClipped = max(dTongue, dMouthInner);
            color = mix(color, uColTongue, getAlpha(dTongueClipped));
        }

        // --- TEETH CONFIG ---
        float teethThick = 1.5 * uScale; 
        
        // The "Pill" Roundness. 
        // 10.0 * Scale makes them very round.
        float pillRadius = 10.0 * uScale; 

        // Gum Mask: Only hides the *Stroke* near the mouth wall
        // This prevents the "Double Line" artifact.
        float distInside = -dMouth;
        float gumMask = 1;

        // --- TOP TEETH ---
        if (uTopTeethCount >= 3) {
            // Extract Box from Points (TL and BR)
            vec2 minP = uTopTeethPts[0];
            vec2 maxP = uTopTeethPts[2];
            vec2 center = (minP + maxP) * 0.5;
            vec2 halfSize = (maxP - minP) * 0.5;

            // 1. Generate Pill Shape
            float dTop = sdRoundedBox(pixelPos - center, halfSize, vec4(pillRadius));

            // 2. FILL: Use Intersection (max)
            // This clips the fill EXACTLY at the inner wall. 
            // Result: Solid connection, no floating, no bleeding.
            float dTopClipped = max(dTop, dMouthInner);
            color = mix(color, uColTeeth, getAlpha(dTopClipped));

            // 3. STROKE: Use Gum Mask
            // We hide the outline near the wall so it doesn't double-up with the mouth outline.
            float strokeAlpha = getStroke(dTop, teethThick) * getAlpha(dMouthInner) * gumMask;
            color = mix(color, uColLine, strokeAlpha);
        }

        // --- BOTTOM TEETH ---
        if (uBotTeethCount >= 3) {
            vec2 minP = uBotTeethPts[0];
            vec2 maxP = uBotTeethPts[2];
            vec2 center = (minP + maxP) * 0.5;
            vec2 halfSize = (maxP - minP) * 0.5;

            float dBot = sdRoundedBox(pixelPos - center, halfSize, vec4(pillRadius));

            // Fill Clip
            float dBotClipped = max(dBot, dMouthInner);
            color = mix(color, uColTeeth, getAlpha(dBotClipped));

            // Stroke Mask
            float strokeAlpha = getStroke(dBot, teethThick) * getAlpha(dMouthInner) * gumMask;
            color = mix(color, uColLine, strokeAlpha);
        }
    }

    // 3. MAIN OUTLINE (Draws on top)
    color = mix(color, uColLine, getStroke(dMouth, actualMouthOutline));

    if (color.a < 0.01) discard;
    finalColor = color;
}