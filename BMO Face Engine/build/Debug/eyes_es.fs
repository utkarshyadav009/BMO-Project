#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform vec4 uColor;

// --- PARAMS ---
//All of these parameter are used to control and shape the eye rendering.
uniform float uShapeID;      // 0-8=Standard, 9=Kawaii, 10=Shocked
uniform float uBend;         
uniform float uThickness;    
uniform float uPupilSize;    
uniform float uTime; 
uniform float uSpiralSpeed;

// --- SURFACE EFFECTS ---
uniform float uStressLevel;  // 0-1: Angry lines under eye
uniform float uGloomLevel;   // 0-1: Shocked vertical lines
uniform int uDistortMode;    // 1: Squash/Stretch distortion


// ---------------------------
// SDF HELPERS
// ---------------------------
float sdCircle(vec2 p, float r) { return length(p) - r; }
float sdBox(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h);
}
vec2 rotate2D(vec2 p, float a) { float s = sin(a), c = cos(a); return vec2(c*p.x - s*p.y, s*p.x + c*p.y); }
float sdArc(vec2 p, float ra, float rb, float aperture) {
    vec2 sc = vec2(sin(aperture), cos(aperture)); p.x = abs(p.x);
    float k = (sc.y*p.x > sc.x*p.y) ? length(p - sc*ra) : abs(length(p) - ra); return k - rb;
}

// ---------------------------
// SHAPE FUNCTIONS
// ---------------------------
float sdStar5(vec2 p, float r, float rf) {
    p /= r; float inner = rf / r; const vec2 k1 = vec2(0.809016994375, -0.587785252292); const vec2 k2 = vec2(-k1.x, k1.y);
    p.x = abs(p.x); p -= 2.0 * max(dot(k1, p), 0.0) * k1; p -= 2.0 * max(dot(k2, p), 0.0) * k2;
    p.x = abs(p.x); p.y -= 1.0; vec2 ba = inner * vec2(-k1.y, k1.x) - vec2(0, 1);
    float h = clamp(dot(p, ba) / dot(ba, ba), 0.0, 1.0);
    return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y) * r;
}
float sdStar4(vec2 p, float r, float rf) {
    p /= r; float k = rf / r; p = abs(p); if (p.x < p.y) p = p.yx;
    const float s = 0.70710678; vec2 p1 = vec2(1.0, 0.0); vec2 p2 = vec2(k*s, k*s);
    vec2 e = p2 - p1; vec2 w = p - p1; vec2 b = w - e*clamp(dot(w,e)/dot(e,e), 0.0, 1.0);
    float d = length(b); float det = e.x*w.y - e.y*w.x; float sgn = (det > 0.0) ? -1.0 : 1.0; 
    return d * sgn * r;
}
float sdHeart(vec2 p) {
    p.x = abs(p.x);
    if (p.y + p.x > 1.0) { vec2 d = p - vec2(0.25, 0.75); return sqrt(dot(d,d)) - sqrt(2.0)/4.0; }
    vec2 d1 = p - vec2(0.00, 1.00); vec2 d2 = p - 0.5 * max(p.x + p.y, 0.0);
    return sqrt(min(dot(d1,d1), dot(d2,d2))) * sign(p.x - p.y);
}
float sdSpiral(vec2 p, float R, float turns, float halfTh, float thetaOffset, float dir, float osc) {
    const float TWO_PI = 6.2831853;
    p.x *= dir; float r = length(p); float a = atan(p.y, p.x); a -= thetaOffset; 
    float spacing = (R / max(turns, 1e-4)) * (1.0 + 0.05 * osc);
    float n_est = (r / spacing) - (a / TWO_PI); float n_center = round(n_est);
    float minD = 1e5;
    for(float i = -1.0; i <= 1.0; i += 1.0) {
        float k = n_center + i; float phi = TWO_PI * k + a;
        float phi_clamped = clamp(phi, 0.0, turns * TWO_PI);
        float r_curve = spacing * (phi_clamped / TWO_PI);
        vec2 p_curve = vec2(cos(phi_clamped), sin(phi_clamped)) * r_curve;
        minD = min(minD, distance(p, p_curve));
    }
    return minD - halfTh;
}

// KAWAII EYE (ID 9)
float sdKawaii(vec2 p, float r) {
    float d = sdCircle(p, r);
    vec2 hl1 = p - vec2(-r*0.35, -r*0.35); float dHl1 = sdCircle(hl1, r * 0.25);
    vec2 hl2 = p - vec2(r*0.2, r*0.3); float dHl2 = sdCircle(hl2, r * 0.1);
    d = max(d, -dHl1); d = max(d, -dHl2);
    return d;
}

// SHOCKED EYE (ID 10)
float sdShocked(vec2 p, float r, float th) {
    float dOuter = sdCircle(p, r);
    float dInner = sdCircle(p, r - th);
    return max(dOuter, -dInner);
}

// STRESS LINES (Angry Feature)
float sdStressLines(vec2 p, float r) {
    p.y -= r * 1.2; p.x = abs(p.x);
    vec2 p1 = rotate2D(p - vec2(r*0.3, 0.0), 0.5);
    float d = sdBox(p1, vec2(r*0.02, r*0.15));
    float d2 = sdBox(p - vec2(0.0, r*0.05), vec2(r*0.02, r*0.15));
    return min(d, d2);
}

// GLOOM LINES (Shocked Feature)
float sdGloomLines(vec2 p, float r) {
    float xPattern = abs(sin(p.x * 20.0)); 
    float d = smoothstep(0.9, 1.0, xPattern); 
    float mask = smoothstep(0.0, -r, p.y);
    return (1.0 - d) * mask; 
}

// ---------------------------
// ANTI-ALIASING
// ---------------------------
float aaFill_HighQuality(float d) { float w = fwidth(d); w = max(w, 1.0); return 1.0 - smoothstep(0.0, w, d); }
float aaFill_Safe(float d) { return 1.0 - smoothstep(-0.75, 0.75, d); }

// ---------------------------
// MAIN
// ---------------------------
void main() {
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    float r = 0.4 * min(uResolution.x, uResolution.y);
    float th = max(uThickness, 1.0);
    bool useSafeAA = false;

    // 1. DOMAIN DISTORTION
    if (uDistortMode == 1) { 
        float nx = p.x / r; 
        p.y += (nx * nx) * (uBend * 0.5) * r;
    }

   // 2. MAIN SHAPE
    float d = 1e5;
    if (uShapeID < 0.5) { // DOT
        d = sdCircle(p, r);
        if (uPupilSize > 0.001) {
            vec2 hlPos = p - vec2(r * 0.28, -r * 0.28);
            d = max(d, -sdCircle(hlPos, r * 0.28) * clamp(uPupilSize, 0.0, 1.0));
        }
    }
    else if (uShapeID < 1.5) { d = sdBox(p, vec2(r, th)); }
    else if (uShapeID < 2.5) { // ARC
        float bend = clamp(uBend, -1.0, 1.0);
        vec2 pp = p; pp.y *= (bend >= 0.0) ? -1.0 : 1.0;
        float a = mix(1.35, 0.85, abs(bend));
        vec2 pp2 = pp; pp2.x /= max(1.2, 0.001); 
        d = sdArc(pp2, r * 0.85, th, a);
    }
    else if (uShapeID < 3.5) { // CROSS
        vec2 p1 = rotate2D(p, 0.785398); vec2 p2 = rotate2D(p, -0.785398);
        d = min(sdBox(p1, vec2(r, th)), sdBox(p2, vec2(r, th)));
    }
    else if (uShapeID < 4.5) { // STAR 5
        float sharp = 0.5 + (uBend * 0.3); 
        d = sdStar5(rotate2D(p, -0.610865), r, r * sharp);
    }
    else if (uShapeID < 5.5) { // HEART
        vec2 hp = p / (r * 1.0); hp.y *= -1.0; hp.y += 0.5;
        d = sdHeart(hp) * (r * 1.0);
    }
    else if (uShapeID < 6.5) { // SPIRAL
        useSafeAA = true;
        float turns = 3.0; float R = r * 0.92; float dir = (uBend >= 0.0) ? 1.0 : -1.0;
        float spin = uTime * uSpiralSpeed;
        d = sdSpiral(rotate2D(p, spin), R, turns, th, 0, dir, 0.0);
    }
    else if (uShapeID < 7.5){ // CHEVRON
        float dir = (uBend >= 0.0) ? 1.0 : -1.0; vec2 pc = vec2(p.x * dir, p.y);
        vec2 a = vec2(-r * 0.45, -r * 0.35); vec2 b = vec2( r * 0.35,  0.00); vec2 c = vec2(-r * 0.45,  r * 0.35);
        d = min(sdSegment(pc, a, b) - th, sdSegment(pc, c, b) - th);
    }
    else if (uShapeID < 8.5) { // SHURIKEN
        float sharp = 0.4 + (uBend * 0.3); d = sdStar4(p, r, r * sharp);
    }
    else if (uShapeID < 9.5) { // KAWAII
        d = sdKawaii(p, r);
    }
    else if (uShapeID < 10.5) { // SHOCKED
        d = sdShocked(p, r, th);
    }

    // 3. ANGRY STRESS LINES
    if (uStressLevel > 0.01) {
        float dStress = sdStressLines(p, r);
        d = min(d, dStress);
    }

    // 4. COLOR & COMPOSITE
    vec4 col = uColor;
    float alpha = (useSafeAA ? aaFill_Safe(d) : aaFill_HighQuality(d));

    // 5. SHOCKED GLOOM (Vertical lines overlay)
    if (uGloomLevel > 0.01) {
        float gloom = sdGloomLines(p, r);
        col.rgb = mix(col.rgb, vec3(0.1, 0.1, 0.4), gloom * uGloomLevel * 0.6);
    }

    if (alpha < 0.01) discard;
    finalColor = vec4(col.rgb, alpha * col.a);
}