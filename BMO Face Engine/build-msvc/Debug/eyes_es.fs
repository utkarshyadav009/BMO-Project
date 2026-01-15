#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform vec4 uColor;

// --- PARAMS ---
uniform float uShapeID;   // 0=Dot, 1=Line(Blink), 2=Arc(Happy), 3=Cross(X), 4=Star, 5=Heart
uniform float uBend;      // Curvature for Line/Arc
uniform float uThickness; // For strokes (Line/Arc/Cross)
uniform float uPupilSize; // For Dot mode

// --- SDF PRIMITIVES ---

// 1. Circle (Standard Eye)
float sdCircle(vec2 p, float r) {
    return length(p) - r;
}

// 2. Segment (The Blink Line)
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// 3. Arc (Happy/Sad Eyes)
// sc is sin/cos of the aperture. ra is radius. rb is thickness.
float sdArc(vec2 p, vec2 sc, float ra, float rb) {
    p.x = abs(p.x);
    return ((sc.y * p.x > sc.x * p.y) ? length(p - sc * ra) : 
           abs(length(p) - ra)) - rb;
}

// 4. Cross (Dead Eyes >.< or X)
float sdCross(vec2 p, vec2 b, float r) {
    p = abs(p); p = (p.y > p.x) ? p.yx : p.xy;
    vec2  q = p - b;
    float k = max(q.y, q.x);
    vec2  w = (k > 0.0) ? q : vec2(b.y - p.x, -k);
    return sign(k) * length(max(w, 0.0)) + r;
}

// 5. Star (Excited)
float sdStar5(vec2 p, float r, float rf) {
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

// 6. Heart (Love)
float sdHeart(vec2 p) {
    p.x = abs(p.x);
    if (p.y + p.x > 1.0) return sqrt(dot(p - vec2(0.25, 0.75), p - vec2(0.25, 0.75))) - sqrt(2.0) / 4.0;
    return sqrt(min(dot(p - vec2(0.00, 1.00), p - vec2(0.00, 1.00)),
                    dot(p - 0.5 * max(p.x + p.y, 0.0), p - 0.5 * max(p.x + p.y, 0.0))));
}

float getAlpha(float d) { return 1.0 - smoothstep(-1.5, 1.5, d); }

void main() {
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    float r = min(uResolution.x, uResolution.y) * 0.4;
    float d = 1e10;
    
    // --- SHAPE SELECTOR ---
    if (uShapeID < 0.5) { 
        // 0: DOT
        d = sdCircle(p, r);
        if (uPupilSize > 0.01) d = max(d, -sdCircle(p, r * uPupilSize * 0.4));
    } 
    else if (uShapeID < 1.5) { 
        // 1: LINE (Blink) - Slight curve downwards
        float bend = uBend * 20.0;
        d = sdSegment(p - vec2(0, bend), vec2(-r, 0), vec2(r, 0)) - uThickness;
    }
    else if (uShapeID < 2.5) { 
        // 2: ARC (Happy)
        // Rotate 180 for happy, 0 for sad
        vec2 pRot = vec2(p.x, -p.y); // Flip Y for "Happy" arch
        float ang = 0.8 + (uBend * 0.5); // Aperture
        d = sdArc(pRot, vec2(sin(ang), cos(ang)), r * 0.8, uThickness);
    }
    else if (uShapeID < 3.5) {
        // 3: CROSS (X)
        // Rotate p by 45 degrees
        float s = 0.7071; float c = 0.7071;
        vec2 pX = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
        d = sdCross(pX, vec2(r * 0.8, r * 0.2), 2.0); // 2.0 = rounded corners
    }
    else if (uShapeID < 4.5) {
        // 4: STAR
        d = sdStar5(p, r, r * 0.5);
    }
    else {
        // 5: HEART
        vec2 hp = p / (r * 1.5); hp.y += 0.5;
        d = sdHeart(hp) * r;
    }

    vec4 color = uColor;
    color.a *= getAlpha(d);
    if (color.a < 0.01) discard;
    finalColor = color;
}