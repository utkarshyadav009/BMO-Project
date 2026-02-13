#version 300 es
precision highp float;

in vec2 fragTexCoord;
out vec4 finalColor;

// ---------------------------
// UNIFORMS
// ---------------------------
uniform vec2 uResolution;
uniform vec4 uColor;
uniform float uTime;

// Eye Parameters
uniform float uShapeID;      // 0-8:Standard, 9:Kawaii, 10:Shocked, 11:Teary
uniform float uBend;         
uniform float uThickness;    
uniform float uSpiralSpeed;
uniform float uEyeSide;
uniform float uAngle;
uniform float uSquareness;

// Surface Effects
uniform float uStressLevel;  // 0-1: Angry lines
uniform float uGloomLevel;   // 0-1: Shocked vertical lines
uniform int uDistortMode;    // 1: Squash/Stretch

// ---------------------------
// CONSTANTS
// ---------------------------
const float PI = 3.14159265359;

// ---------------------------
// SDF PRIMITIVES
// ---------------------------
float sdCircle(vec2 p, float r) { return length(p) - r; }
float sdBox(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }
vec2 rotate2D(vec2 p, float a) { float s = sin(a), c = cos(a); return vec2(c*p.x - s*p.y, s*p.x + c*p.y); }

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h);
}

float sdCapsule(vec2 p, vec2 a, vec2 b, float r) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdBezier(vec2 pos, vec2 A, vec2 B, vec2 C) {    
    vec2 a = B - A; vec2 b = A - 2.0*B + C; vec2 c = a * 2.0; vec2 d = A - pos;
    float kk = 1.0 / dot(b,b); float kx = kk * dot(a,b); float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0; float kz = kk * dot(d,a);      
    float res = 0.0; float p = ky - kx*kx; float p3 = p*p*p; float q = kx*(2.0*kx*kx - 3.0*ky) + kz; float h = q*q + 4.0*p3;
    if(h >= 0.0) { 
        h = sqrt(h); vec2 x = (vec2(h, -h) - q) / 2.0; vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        float t = clamp(uv.x+uv.y-kx, 0.0, 1.0); res = length(d + (c + b*t)*t);
    } else {
        float z = sqrt(-p); float v = acos( q/(p*z*2.0) ) / 3.0; float m = cos(v); float n = sin(v)*1.732050808;
        vec3 t = clamp(vec3(m+m, -n-m, n-m)*z-kx, 0.0, 1.0);
        res = min( dot(d+(c+b*t.x)*t.x, d+(c+b*t.x)*t.x), dot(d+(c+b*t.y)*t.y, d+(c+b*t.y)*t.y) );
        res = min( res, dot(d+(c+b*t.z)*t.z, d+(c+b*t.z)*t.z) ); res = sqrt( res );
    }
    return res;
}

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
    float r = length(p); float a = atan(p.y, p.x); a -= thetaOffset; 
    float spacing = (R / max(turns, 1e-4)) * (1.0 + 0.05 * osc);
    float n_est = (r / spacing) - (a / (2.0*PI)); float n_center = round(n_est);
    float minD = 1e5;
    for(float i = -1.0; i <= 1.0; i += 1.0) {
        float k = n_center + i; float phi = 2.0*PI * k + a;
        float phi_clamped = clamp(phi, 0.0, turns * 2.0*PI);
        float r_curve = spacing * (phi_clamped / (2.0*PI));
        vec2 p_curve = vec2(cos(phi_clamped), sin(phi_clamped)) * r_curve;
        minD = min(minD, distance(p, p_curve));
    }
    return minD - halfTh;
}

float sdMorphShape(vec2 p, float r) {
    float dCircle = length(p) - r;
    float dBox = sdBox(p, vec2(r, r)); 
    return mix(dCircle, dBox, uSquareness);
}

// ---------------------------
// COMPLEX EYE STYLES
// ---------------------------

// Style 9: Kawaii (Round with highlights)
float eyeKawaii(vec2 p, float r, inout vec3 col) {
    float d = length(p) - r;
    
    // Highlights
    vec2 pos1 = rotate2D(p - vec2(-r * 0.25, -r * 0.30), radians(37.0));
    float dH1 = length(vec2(pos1.x * 0.7, pos1.y * 0.95)) - (r * 0.52);

    vec2 pos2 = p - vec2(r * 0.40, r * 0.35);
    float dH2 = length(pos2) - (r * 0.27);

    vec2 pos3 = p - vec2(-r * 0.15, r * 0.50);
    float dH3 = length(pos3) - (r * 0.1);

    float highlights = min(dH1, min(dH2, dH3));
    float highlightMask = smoothstep(0.35, -0.35, highlights); // AA 0.7 split

    col = mix(col, vec3(1.0), highlightMask);
    return d;
}

// Style 10: Shocked (Ring with white center)
float eyeShocked(vec2 p, float r, float th, inout vec3 col) {
    float dist = length(p);
    float d = dist - r; // Outer geometry
    
    float boundary = r - th;
    float blend = smoothstep(boundary - 0.375, boundary + 0.375, dist);
    
    // Mix White Center -> Color Edge
    col = mix(vec3(1.0), col, blend);
    return d;
}

// Style 11: Teary/Glossy (Oval with highlights)
float eyeTeary(vec2 p, float r, inout vec3 col) {
    // Main Shape: Vertical Oval
    float d = length(vec2(p.x * 1.35, p.y)) - (r * 1.1);

    // Highlights
    vec2 pos1 = rotate2D(p - vec2(-r * 0.30, -r * 0.65), radians(40.0));
    float dH1 = length(vec2(pos1.x * 0.8, pos1.y)) - (r * 0.24);

    vec2 pos2 = rotate2D(p - vec2(r * 0.07, -r * 0.35), radians(40.0));
    float dH2 = length(vec2(pos2.x * 0.8, pos2.y)) - (r * 0.10);

    float highlights = min(dH1, dH2);
    float softAA = r * 0.04; 
    float highlightMask = smoothstep(softAA, -softAA, highlights);

    col = mix(col, vec3(1.0), highlightMask);
    return d;
}

// ---------------------------
// MAIN SHAPE DISPATCH
// ---------------------------
float getEyeShape(vec2 p, float r, float th, float id, inout vec3 col, inout bool safeAA) {
    float d = 1e5;
    
    if (id < 0.5) { // 0: DOT
        d = sdCircle(p, r);
    }
    else if (id < 1.5) { // 1: BOX
        d = sdBox(p, vec2(r, th)); 
    }
    else if (id < 2.5) { // 2: ARC
        float width = r * 0.9;
        float bendHeight = uBend * r * 1.5;
        vec2 B = vec2(0.0, -bendHeight);
        d = sdBezier(p, vec2(-width, 0.0), B, vec2(width, 0.0)) - th;
    }
    else if (id < 3.5) { // 3: CROSS
        vec2 p1 = rotate2D(p, 0.785398); 
        vec2 p2 = rotate2D(p, -0.785398);
        d = min(sdBox(p1, vec2(r, th)), sdBox(p2, vec2(r, th)));
    }
    else if (id < 4.5) { // 4: STAR 5
        float sharp = 0.5 + (uBend * 0.3);
        d = sdStar5(rotate2D(p, -0.610865), r, r * sharp);
    }
    else if (id < 5.5) { // 5: HEART
        vec2 hp = p / (r * 1.0); hp.y *= -1.0; hp.y += 0.5;
        float dHeart = sdHeart(hp) * r;
        d = dHeart - th;
        float aa = 1.0;
        float outlineMix = smoothstep(-aa*0.5, aa*0.5, dHeart);
        col = mix(col, vec3(0.0), outlineMix);
    }
    else if (id < 6.5) { // 6: SPIRAL
        safeAA = true;
        float R = r * 0.92; 
        float dir = (uBend >= 0.0) ? 1.0 : -1.0;
        float spin = uTime * uSpiralSpeed;
        d = sdSpiral(rotate2D(p, spin), R, 3.0, th, 0.0, dir, 0.0);
    }
    else if (id < 7.5) { // 7: CHEVRON
        float dir = uEyeSide; vec2 pc = vec2(p.x * dir, p.y);
        d = min(sdSegment(pc, vec2(-r*0.45, -r*0.55), vec2(r*0.35, 0.0)) - th, 
                sdSegment(pc, vec2(-r*0.45,  r*0.55), vec2(r*0.35, 0.0)) - th);
    }
    else if (id < 8.5) { // 8: SHURIKEN
        float sharp = 0.4 + (uBend * 0.3); 
        d = sdStar4(p, r, r * sharp);
    }
    else if (id < 9.5) { // 9: KAWAII
        d = eyeKawaii(p, r, col);
    }
    else if (id < 10.5) { // 10: SHOCKED
        d = eyeShocked(p, r, th, col);
    }
    else if (id < 11.5) { // 11: TEARY
        d = eyeTeary(p, r, col);
    }
    else if (id < 12.5) { //12 : : eyes, with two colons 
        //Creating the :: eye shape by rendering two extra circles, under the main eye circle
        float spacing = r * 2.5;

        vec2 pTop = p- vec2(0.0, -spacing);
        float dTop = sdMorphShape(pTop, r);

        vec2 pBottom = p - vec2(0.0, spacing-12.0);
        float dBottom = sdMorphShape(pBottom, r);

        d = min(dTop, dBottom);
    }
    
    return d;
}

// ---------------------------
// EFFECTS (Stress, Gloom)
// ---------------------------
float getStressLines(vec2 p, float r) {
    float d = 1e5;
    float th = r * 0.04;
    float outerDir = -uEyeSide;
    
    if (uStressLevel > 0.9) {
        if(uEyeSide == -1.0) { // Specific case for right eye level 3
            vec2 anchor = vec2(outerDir * r * 0.8, r * 1.1);
            vec2 q = rotate2D(p - anchor - vec2(0.0, -r*0.2), radians(-61.0) * outerDir);
            float lx = q.x - (-r * 5.4 * outerDir);
            q.y += 0.5 * (lx * lx) / r;
            d = sdCapsule(q, vec2(-r * 5.85*outerDir, r * 1.25), vec2(-r * 4.9*outerDir, r * 1.15), th);
        }
    }
    else if (uStressLevel > 0.6) {
        vec2 anchor = vec2(outerDir * r * 0.8, r * 1.1);
        vec2 q = rotate2D(p - anchor, radians(-33.0) * outerDir);
        float lx = q.x - (-r * 0.95 * outerDir);
        q.y += 0.2 * (lx * lx) / r;
        d = sdCapsule(q, vec2(-r * 2.1*outerDir, r * 1.4), vec2(r * 0.01*outerDir, r * 1.1), th);
    }
    else if (uStressLevel > 0.3) {
        outerDir = uEyeSide;
        vec2 anchor = vec2(outerDir * r * 0.8, r * 1.1);
        vec2 baseQ = p - anchor;
        
        // Mark 1
        vec2 q1 = rotate2D(baseQ, radians(33.0) * outerDir);
        q1.y += 0.3 * (q1.x * q1.x) / r;
        float d1 = sdCapsule(q1, vec2(-r*0.55, r*0.50), vec2(r*0.55, r*0.50), th);
        
        // Mark 2
        vec2 q2 = rotate2D(baseQ - vec2(0.0, -r*0.2), radians(-30.0) * outerDir);
        q2.x -= 0.9 * ((q2.y - (-r*3.25)) * (q2.y - (-r*3.25)) / r) * -outerDir;
        float d2 = sdCapsule(q2, vec2(r*0.7*outerDir, -r*3.3), vec2(r*0.7*outerDir, -r*2.9), th);
        
        d = min(d1, d2);
    }
    return d;
}

float getGloomLines(vec2 p, float r) {
    if (uEyeSide != 1.0) return 1e5; // Only left eye
    
    vec2 q = p - vec2(-r * 1.0, -r * 1.3);
    float th = r * 0.04;
    
    float d1 = sdCapsule(q, vec2(-r*1.07, -r*0.1), vec2(-r*1.07, r*1.5), th);
    float d2 = sdCapsule(q, vec2(-r*0.60, -r*0.1), vec2(-r*0.60, r*1.5), th);
    float d3 = sdCapsule(q, vec2(-r*0.13, -r*0.1), vec2(-r*0.13, r*1.5), th);
    
    return min(d1, min(d2, d3));
}
// ---------------------------
// MAIN
// ---------------------------
void main() {
    // 1. Setup Coordinates
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    float r = 0.13 * min(uResolution.x, uResolution.y);
    float th = max(uThickness, 1.0);
    vec3 col = uColor.rgb;
    bool useSafeAA = false;

    // STABLE PIXEL WIDTH: Calculate this BEFORE any distortion or loops
    // This represents the size of one pixel in your 'p' coordinate space.
    float pixelWidth = fwidth(p.x);
    
    // 2. Domain Distortion (Squash/Stretch)
    if (uDistortMode == 1) { 
        float nx = p.x / r; 
        p.y += (nx * nx) * (uBend * 0.5) * r;
    }

    // 3. Calculate Main Shape
    // Pass useSafeAA by reference so getEyeShape can toggle it for the spiral
    float d = getEyeShape(p, r, th, uShapeID, col, useSafeAA);

    // 4. Calculate Effects
    if (uStressLevel > 0.3) {
        d = min(d, getStressLines(p, r));
    }
    
    if (uGloomLevel > 0.01) {
        float gloomDist = getGloomLines(p, r);
        d = min(d, gloomDist);
        float gloomMask = 1.0 - smoothstep(0.0, 1.5, gloomDist);
        vec3 gloomColor = vec3(0.37647, 0.13333, 0.61961);
        col = mix(col, gloomColor, gloomMask * uGloomLevel);
    }

    if (uSquareness > 0.0 && uShapeID != 12.0) {
        float squareDistX = sdMorphShape(p, r);
        d = mix(d, squareDistX, uSquareness);
    }

    // 5. Render Anti-Aliasing
    float alpha;
    if (useSafeAA) {
        // APPROACH: Use the stable coordinate width instead of fwidth(d)
        // This removes the 'ghost' outlines caused by spiral branch switching.
        alpha = 1.0 - smoothstep(-pixelWidth, pixelWidth, d);
    } else {
        // Standard AA for non-looping shapes
        float w = fwidth(d);
        alpha = 1.0 - smoothstep(-w, w, d);
    }

    if (alpha < 0.01) discard;
    finalColor = vec4(col, alpha * uColor.a);
}
