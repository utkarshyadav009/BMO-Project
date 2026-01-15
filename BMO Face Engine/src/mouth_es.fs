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
    
    if (dMouth > actualMouthOutline + 2.0) discard;

    float mouthAlpha = getAlpha(dMouth);
    vec4 color = mix(vec4(0.0), uColBg, mouthAlpha);

    if (mouthAlpha > 0.01) {
        
        // INTERSECTION CLIPPER:
        // Pushes white fill deep into the center of the black line (+0.5 offset)
        // effectively burying the edge under opaque black pixels.
        float dMouthInner = dMouth + 0.5;

        // TONGUE
        if (uTongueCount >= 3) {
            float dTongue = sdPoly(pixelPos, uTonguePts, uTongueCount);
            float dTongueClipped = max(dTongue, dMouthInner);
            color = mix(color, uColTongue, getAlpha(dTongueClipped));
        }

        float teethThick = 1.5 * uScale; 
        float pillRadius = 10.0 * uScale; 

        // [FIX] GUM MASK TUNING
        // We calculate distance from the center of the mouth outline.
        // Range (0.0 -> 1.5):
        //  - At 0.0 (center of mouth line): Mask is 0 (Stroke invisible).
        //  - At 1.5 (just inside): Mask is 1 (Stroke fully visible).
        // This tight transition allows the stroke to touch the wall without contracting into a V.
        float distInside = -dMouth;
        float gumMask = smoothstep(0.0, 1.5 * uScale, distInside);

        // TOP TEETH
        if (uTopTeethCount >= 3) {
            vec2 minP = uTopTeethPts[0];
            vec2 maxP = uTopTeethPts[2];
            vec2 center = (minP + maxP) * 0.5;
            vec2 halfSize = (maxP - minP) * 0.5;

            float dTop = sdRoundedBox(pixelPos - center, halfSize, vec4(pillRadius));

            // Fill
            float dTopClipped = max(dTop, dMouthInner);
            color = mix(color, uColTeeth, getAlpha(dTopClipped));

            // Stroke (Masked tightly)
            float strokeAlpha = getStroke(dTop, teethThick) * getAlpha(dMouthInner) * gumMask;
            color = mix(color, uColLine, strokeAlpha);
        }

        // BOTTOM TEETH
        if (uBotTeethCount >= 3) {
            vec2 minP = uBotTeethPts[0];
            vec2 maxP = uBotTeethPts[2];
            vec2 center = (minP + maxP) * 0.5;
            vec2 halfSize = (maxP - minP) * 0.5;

            float dBot = sdRoundedBox(pixelPos - center, halfSize, vec4(pillRadius));

            // Fill
            float dBotClipped = max(dBot, dMouthInner);
            color = mix(color, uColTeeth, getAlpha(dBotClipped));

            // Stroke (Masked tightly)
            float strokeAlpha = getStroke(dBot, teethThick) * getAlpha(dMouthInner) * gumMask;
            color = mix(color, uColLine, strokeAlpha);
        }
    }

    // MAIN OUTLINE (Draws on top)
    color = mix(color, uColLine, getStroke(dMouth, actualMouthOutline));

    if (color.a < 0.01) discard;
    finalColor = color;
}