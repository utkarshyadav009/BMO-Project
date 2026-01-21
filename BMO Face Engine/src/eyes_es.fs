#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform vec4 uColor;

// --- PARAMS ---
uniform float uShapeID;      // 0-8=Eyes, 10=Blush, 20=Eyebrow, 21=Tears
uniform float uBend;         
uniform float uThickness;    
uniform float uPupilSize;    

// ELEMENT TOGGLES (0 = Off, 1 = On)
uniform int uShowBrow;
uniform int uShowTears;
uniform int uShowBlush;

// ELEMENT PARAMS
uniform float uEyebrowType;  // 0=None, 1=Angry, 2=Sad, 3=Raised
uniform float uEyebrowY;     
uniform float uTearsLevel;   // 0.0 to 1.0
uniform float uTime; 
uniform float uSpiralSpeed;

// NEW: DISTORTION (For eyebrows or wavy eyes)
uniform int uDistortMode;


// ---------------------------
// SDF HELPERS (Keep existing)
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
// SHAPE FUNCTIONS (Keep existing)
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

    // -------------------------
    // A) EYE SHAPE (uShapeID 0..8)
    // -------------------------
    float dEye = 1e10;

    // Domain Distortion (Only affects Eye, not Brow)
    vec2 pEye = p; 
    if (uDistortMode == 1) { 
        float nx = pEye.x / r; 
        pEye.y += (nx * nx) * (uBend * 0.5) * r;
    }

    if (uShapeID < 0.5) { // DOT
        dEye = sdCircle(pEye, r);
        if (uPupilSize > 0.001) {
            vec2 hlPos = pEye - vec2(r * 0.28, -r * 0.28);
            dEye = max(dEye, -sdCircle(hlPos, r * 0.28) * clamp(uPupilSize, 0.0, 1.0));
        }
    }
    else if (uShapeID < 1.5) { dEye = sdBox(pEye, vec2(r, th)); }
    else if (uShapeID < 2.5) { // ARC
        float bend = clamp(uBend, -1.0, 1.0);
        vec2 pp = pEye; pp.y *= (bend >= 0.0) ? -1.0 : 1.0;
        float a = mix(1.35, 0.85, abs(bend));
        vec2 pp2 = pp; pp2.x /= max(1.2, 0.001); 
        dEye = sdArc(pp2, r * 0.85, th, a);
    }
    else if (uShapeID < 3.5) { // CROSS
        vec2 p1 = rotate2D(pEye, 0.785398); vec2 p2 = rotate2D(pEye, -0.785398);
        dEye = min(sdBox(p1, vec2(r, th)), sdBox(p2, vec2(r, th)));
    }
    else if (uShapeID < 4.5) { // STAR 5
        float sharp = 0.5 + (uBend * 0.3); 
        dEye = sdStar5(rotate2D(pEye, -0.610865), r, r * sharp);
    }
    else if (uShapeID < 5.5) { // HEART
        vec2 hp = pEye / (r * 1.0); hp.y *= -1.0; hp.y += 0.5;
        dEye = sdHeart(hp) * (r * 1.0);
    }
    else if (uShapeID < 6.5) { // SPIRAL
        useSafeAA = true;
        float turns = 3.0; float R = r * 0.92; float dir = (uBend >= 0.0) ? 1.0 : -1.0;
        float spin = uTime * uSpiralSpeed;
        dEye = sdSpiral(rotate2D(pEye, spin), R, turns, th, 0, dir, 0.0);
    }
    else if (uShapeID < 7.5){ // CHEVRON
        float dir = (uBend >= 0.0) ? 1.0 : -1.0; vec2 pc = vec2(pEye.x * dir, pEye.y);
        vec2 a = vec2(-r * 0.45, -r * 0.35); vec2 b = vec2( r * 0.35,  0.00); vec2 c = vec2(-r * 0.45,  r * 0.35);
        dEye = min(sdSegment(pc, a, b) - th, sdSegment(pc, c, b) - th);
    }
    else { // SHURIKEN
        float sharp = 0.4 + (uBend * 0.3); dEye = sdStar4(pEye, r, r * sharp);
    }
    
    // -------------------------
    // B) EYEBROWS
    // -------------------------
    float dBrow = 1e10;
    if (uShowBrow == 1) {
        float browBaseY = -r * 1.5 + uEyebrowY; // Position above eye
        vec2 bp = p - vec2(0.0, browBaseY);
        float mode = uEyebrowType;
        float browAngle = (mode < 1.5) ? 0.45 : (mode < 2.5) ? -0.45 : 0.0;
        
        vec2 br = rotate2D(bp, browAngle);
        
        // Use uBend for brow curve (Parabola)
        float nx = br.x / r; 
        br.y += (nx * nx) * (uBend * 0.5) * r;

        dBrow = sdBox(br, vec2(r * 0.8, th * 0.85));
    }

    // -------------------------
    // C) TEARS
    // -------------------------
    float dTear = 1e10;
    if (uShowTears == 1 && uTearsLevel > 0.01) {
        float t = clamp(uTearsLevel, 0.0, 1.0);
        vec2 start = vec2(0.0, -r); 
        vec2 end   = vec2(0.0, r * (0.5 + 1.5 * t));
        vec2 tp = p; tp.x -= sin(tp.y * 0.03) * (r * 0.05) * t;
        float stream = sdSegment(tp, start, end) - (th * 0.65) * mix(0.8, 1.3, t);
        float drop = sdCircle(tp - end, r * 0.10 * mix(0.8, 1.25, t));
        dTear = min(stream, drop);
    }

    // -------------------------
    // D) BLUSH (Alpha Only)
    // -------------------------
    float blushAlpha = 0.0;
    if (uShowBlush == 1) {
        float dBlush = sdCircle(p - vec2(0.0, r*2), r * 0.6); // Below eye
        blushAlpha = 1.0 - smoothstep(0.0, r, dBlush + r * 0.2);
        blushAlpha *= 0.5; // Opacity
    }

    // -------------------------
    // E) COMBINE & COLOR
    // -------------------------
    
    // 1. Determine Main Shape Distance
    float dSolid = min(dEye, min(dBrow, dTear));
    
    // 2. Determine Color (Multi-Material Support!)
    vec3 outColor = uColor.rgb; // Default to Eye Color
    
    // If Brow is the closest shape, make it Black
    if (dBrow < dEye && dBrow < dTear) {
        outColor = vec3(0.0, 0.0, 0.0);
    }
    // If Tear is the closest shape, make it Blue
    else if (dTear < dEye && dTear < dBrow) {
        outColor = vec3(0.4, 0.7, 1.0);
    }

    // 3. Alpha Calculation
    float alpha = (useSafeAA ? aaFill_Safe(dSolid) : aaFill_HighQuality(dSolid));
    
    // 4. Mix Blush (Additive-ish)
    vec3 blushColor = vec3(1.0, 0.7, 0.8); // Pink
    outColor = mix(outColor, blushColor, blushAlpha);
    alpha = max(alpha, blushAlpha);

    if (alpha < 0.01) discard;
    finalColor = vec4(outColor, alpha);
}