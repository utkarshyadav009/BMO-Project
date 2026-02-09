#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

// --- CONFIGURATION ---
#define MAX_PTS 128
#define EPSILON 1e-6

// --- UNIFORMS ---
// Geometry Arrays
uniform vec2 uMouthPts[MAX_PTS];
uniform int uMouthCount;
uniform vec2 uTopTeethPts[MAX_PTS];
uniform int uTopTeethCount;
uniform vec2 uBotTeethPts[MAX_PTS];
uniform int uBotTeethCount;
uniform vec2 uTonguePts[MAX_PTS];
uniform int uTongueCount;

// Aesthetic Parameters
uniform float uStressLines; // Slider (0.0 to 1.0)

// System & Style
uniform vec2 uResolution; 
uniform float uPadding; 
uniform float uScale;    
uniform float uOutlineThickness;
uniform vec4 uColBg;
uniform vec4 uColLine;
uniform vec4 uColTeeth;
uniform vec4 uColTongue;

// --- MATH HELPERS ---
vec2 rotate2D(vec2 v, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c) * v;
}

// --- SDF FUNCTIONS ---

// Signed Distance to a Quadratic Bezier Curve
// A = Start, B = Control Point, C = End
float sdBezier(vec2 pos, vec2 A, vec2 B, vec2 C) {    
    vec2 a = B - A;
    vec2 b = A - 2.0*B + C;
    vec2 c = a * 2.0;
    vec2 d = A - pos;

    float kk = 1.0 / dot(b,b);
    float kx = kk * dot(a,b);
    float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
    float kz = kk * dot(d,a);      

    float res = 0.0;
    float p = ky - kx*kx;
    float p3 = p*p*p;
    float q = kx*(2.0*kx*kx - 3.0*ky) + kz;
    float h = q*q + 4.0*p3;

    if(h >= 0.0) { 
        h = sqrt(h);
        vec2 x = (vec2(h, -h) - q) / 2.0;
        vec2 uv = sign(x)*pow(abs(x), vec3(1.0/3.0).xy);
        float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);
        res = length(d + (c + b*t)*t);
    } else {
        float z = sqrt(-p);
        float v = acos( q/(p*z*2.0) ) / 3.0;
        float m = cos(v);
        float n = sin(v)*1.732050808;
        vec3 t = clamp(vec3(m+m, -n-m, n-m)*z-kx, 0.0, 1.0);
        res = min( dot(d+(c+b*t.x)*t.x, d+(c+b*t.x)*t.x),
                   dot(d+(c+b*t.y)*t.y, d+(c+b*t.y)*t.y) );
        res = min( res, dot(d+(c+b*t.z)*t.z, d+(c+b*t.z)*t.z) );
        res = sqrt( res );
    }
    return res;
}

// Arbitrary Polygon SDF
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

// Anti-aliasing helpers
// Anti-aliasing helpers
float getAlpha(float d) { 
    // Since 'd' is already in screen pixels, we use a fixed 1-pixel feather.
    // 0.0 to 1.0 is a 1px transition. -0.5 to 0.5 is also 1px but centered.
    // Using 0.75 gives a slightly "crisper" look than 1.0 without aliasing.
    return 1.0 - smoothstep(-0.75, 0.75, d);
}

float getStroke(float d, float thick) { 
    // No fwidth needed.
    // Standard 1px AA on the stroke edges.
    return 1.0 - smoothstep(thick - 0.75, thick + 0.75, abs(d));
}

// --- EFFECTS: STRESS LINES (BEZIER) ---
// Helper: Draws a single bezier curve
// p: current pixel coordinate
// origin: offset for this specific line
// scale: size multiplier
// bendFactor: Controls shape. 
//    +0.2 = Bulge Right (Parenthesis)
//    -0.2 = Bulge Left (Reverse Parenthesis)
//    +0.5 = Sharp Curve (Hook)
// length: Controls vertical height (0.3 is standard, 0.5 is long, 0.15 is short)
float stressCurve(vec2 p, vec2 origin, float scale, float length, float bendFactor, float rotateAngle,float r) {
    vec2 localP = p - origin;
    if (abs(rotateAngle) > 0.001) {
        localP = rotate2D(localP, -rotateAngle);
    }
    float localR = r * scale;
    
    // Thickness is based on Scale only (not length)
    float th = localR * 0.035; 

    // A: Start (Top)
    // We use 'length' here instead of a hardcoded 0.3
    vec2 A = vec2(-localR * 0.05, localR * length); 

    // C: End (Bottom)
    vec2 C = vec2(-localR * 0.05, -localR * length);

    // B: Control Point (Bulge)
    vec2 B = vec2(localR * bendFactor, 0.0); 

    return sdBezier(localP, A, B, C) - th;
}
float getStressLines(vec2 p, float r) {
    if (uStressLines < 0.01) return 1e5;

    // Coordinate Systems
    vec2 pSym = vec2(abs(p.x), p.y); // Mirrored (Left+Right)
    vec2 pRaw = p;                   // Asymmetric (Right only)

    float dFinal = 1e5;

    // ---------------------------------------------------------
    // --- SCENARIO CONFIGURATION ---
    // ---------------------------------------------------------

    // MODE 1: DIMPLES (The "Parentheses" look)
    // Position: Next to mouth corners
    // Bend: 0.25 (Outward curve)
    vec2  m1_pos  = vec2(r * 0.47, -r * 0.08); 
    float m1_size = 0.55;
    float m1_bend = -0.28;
    float m1_len  = 0.30; // Standard
    float m1_ang = 0.0;

    // MODE 2: SWEAT DROP (Classic Anime Sweat)
    // Position: High up, Asymmetric (Right side only)
    // Bend: 0.6 (Very round, almost a circle/hook)
    vec2  m2_pos  = vec2(r * 0.58, r * 0.05); 
    float m2_size = 0.7; 
    float m2_bend = -0.52;
    float m2_len  = 0.32; // Short/Stubby
    float m2_ang = 0.0;

    // MODE 3: STRESS TICK (Frustration Mark)
    // Position: Higher up on forehead
    // Bend: -0.3 (Inward "Tick" shape)
    vec2  m3_pos  = vec2(r * 0.45, r * 0.25); 
    float m3_size = 0.35;
    float m3_bend = -0.60; 
    float m3_len  = 0.50; // Long
    float m3_ang = -9.0;

    // ---------------------------------------------------------
    // --- SELECTION LOGIC ---
    // ---------------------------------------------------------
    
    if (uStressLines < 0.33) {
        // Mode 1: Symmetric Dimples
        dFinal = stressCurve(pSym, m1_pos, m1_size, m1_len, m1_bend,m1_ang, r);
    } 
    else if (uStressLines < 0.66) {
        // Mode 2: Asymmetric Sweat Drop
        dFinal = stressCurve(pRaw, m2_pos, m2_size, m2_len, m2_bend,m2_ang, r);
    } 
    else {
        // Mode 3: Symmetric Stress Ticks
        dFinal = stressCurve(pSym, m3_pos, m3_size, m3_len, m3_bend,m3_ang, r);
    }

    return dFinal;
}

// --- MAIN ---
void main() {
    // 1. Setup Coordinates
    vec2 pixelPos = fragTexCoord * uResolution;
    
    // Centered coordinate system for stress lines
    vec2 centerPos = pixelPos - (uResolution * 0.5);

    // 2. Compute Distances (SDFs)
    
    // A. Mouth Shape
    float dMouth = sdPoly(pixelPos, uMouthPts, uMouthCount);
    float actualMouthOutline = uOutlineThickness * uScale;
    float dMouthInner = dMouth + 0.5; // Slightly shrunk for fill calculations

    // B. Stress Lines
    float dStress = getStressLines(centerPos, 100.0 * uScale);
    float stressAlpha = getAlpha(dStress); // Solid fill
    
    // 3. Composite Logic
    bool insideMouth = (dMouth <= actualMouthOutline + 1.0);
    bool insideStress = (stressAlpha > 0.01);

    if (!insideMouth && !insideStress) discard;

    // 4. Drawing
    vec4 color = vec4(0.0); // Start Transparent

    // Layer 1: Stress Lines (Background)
    if (insideStress) {
        color = mix(color, uColLine, stressAlpha);
    }

    // Layer 2: Mouth (Foreground)
    if (insideMouth) {
        float mouthAlpha = getAlpha(dMouth);
        
        // Blend background (mouth cavity) over the stress lines
        color = mix(color, uColBg, mouthAlpha);

        // Tongue
        if (uTongueCount >= 3) {
            float dTongue = sdPoly(pixelPos, uTonguePts, uTongueCount);
            float dTongueClipped = max(dTongue, dMouthInner);
            color = mix(color, uColTongue, getAlpha(dTongueClipped));
        }

        // Teeth Common Settings
        float teethThick = 1.5 * uScale; 
        float pillRadius = 10.0 * uScale; 
        
        // Gum Mask
        float distInside = -dMouth;
        float gumMask = smoothstep(0.0, 1.5 * uScale, distInside);

        // Top Teeth
        if (uTopTeethCount >= 3) {
            vec2 minP = uTopTeethPts[0]; vec2 maxP = uTopTeethPts[2];
            vec2 center = (minP + maxP) * 0.5; vec2 halfSize = (maxP - minP) * 0.5;
            float dTop = sdRoundedBox(pixelPos - center, halfSize, vec4(pillRadius));

            color = mix(color, uColTeeth, getAlpha(max(dTop, dMouthInner)));
            float strokeAlpha = getStroke(dTop, teethThick) * getAlpha(dMouthInner) * gumMask;
            color = mix(color, uColLine, strokeAlpha);
        }

        // Bottom Teeth
        if (uBotTeethCount >= 3) {
            vec2 minP = uBotTeethPts[0]; vec2 maxP = uBotTeethPts[2];
            vec2 center = (minP + maxP) * 0.5; vec2 halfSize = (maxP - minP) * 0.5;
            float dBot = sdRoundedBox(pixelPos - center, halfSize, vec4(pillRadius));

            color = mix(color, uColTeeth, getAlpha(max(dBot, dMouthInner)));
            float strokeAlpha = getStroke(dBot, teethThick) * getAlpha(dMouthInner) * gumMask;
            color = mix(color, uColLine, strokeAlpha);
        }

        // Main Mouth Outline
        color = mix(color, uColLine, getStroke(dMouth, actualMouthOutline));
    }

    // Final Cleanup
    if (color.a < 0.01) discard;
    finalColor = color;
}