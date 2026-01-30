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
uniform float uTime; 
uniform float uSpiralSpeed;
uniform float uEyeSide;

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
float sdCapsule(vec2 p, vec2 a, vec2 b, float r)
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
vec2 rotate2D(vec2 p, float a) { float s = sin(a), c = cos(a); return vec2(c*p.x - s*p.y, s*p.x + c*p.y); }
float sdArc(vec2 p, float ra, float rb, float aperture) {
    vec2 sc = vec2(sin(aperture), cos(aperture)); p.x = abs(p.x);
    float k = (sc.y*p.x > sc.x*p.y) ? length(p - sc*ra) : abs(length(p) - ra); return k - rb;
}

//Bezier SDF from Inigo Quilez
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
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0)); // Fixed vec3->vec2 cast
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
    return length(p) - r;
}

// SHOCKED EYE (ID 10)
float sdShocked(vec2 p, float r, float th) {
    float dOuter = sdCircle(p, r);
    float dInner = sdCircle(p, r - th);
    return max(dOuter, -dInner);
}

// STRESS MARKS (Target: Bottom-Outer Corner)
// side: 1.0 for Left Eye, -1.0 for Right Eye
float sdStressLines(vec2 p, float r, float side) {
    // --- 1. SETUP ---
    float downDir = 1.0; 
    float outerDir = -side; 

    // Global Anchor (Center of the stress mark cluster)
    vec2 anchor = vec2(outerDir * r * 0.8, r * 1.1 * downDir);
    vec2 baseQ = p - anchor; // Store the clean, un-rotated coordinates

    float th = r * 0.04;

    // --- MARK 1 (Upper) ---
    vec2 q1 = baseQ; 
    
    // Rotate q1
    float angle1 = radians(33.0) * outerDir * downDir;
    q1 = rotate2D(q1, angle1);
    
    // Bend q1
    q1.y += 0.3 * (q1.x * q1.x) / r;
    
    // Draw Mark 1
    float d1 = sdCapsule(q1, 
        vec2(-r * 0.55, r * 0.50), 
        vec2( r * 0.55, r * 0.50), 
        th);


    // --- MARK 2 (Lower) ---
    // Start fresh from baseQ (ignores Mark 1's rotation/bend)
    vec2 q2 = baseQ;

    // Move it UP/DOWN relative to the anchor (in clean screen space)
    // Change '0.2' to move vertically. Negative = Up.
    q2.y -= r * 0.2; 
    
    // Rotate q2 independently
    // You can now set this angle to whatever you want (e.g. 35.0 to match, or 45.0)
    float angle2 = radians(-30.0)* outerDir * downDir;
    q2 = rotate2D(q2, angle2);

    float lineCenterY = -r * 3.25;
    float localY = q2.y - lineCenterY;

    q2.x -= 0.9 * (localY * localY / r) * -outerDir;

    // Bend q2 independently
    //q2.y -= 0.15 * (q2.x * q2.x) / r;

    // Draw Mark 2
    // Note: Use simple coordinates now, since we handled the offset in 'q2.y -= ...'
    float d2 = sdCapsule(q2, 
        vec2(r * 0.7*outerDir, -r * 3.3), 
        vec2( r * 0.70 *outerDir,  -r * 2.9), 
        th);

    return min(d1, d2);
}


float sdStressLines1(vec2 p, float r, float side) {
    // --- 1. SETUP ---
    float downDir = 1.0; 
    float outerDir = -side; 

    // Global Anchor (Center of the stress mark cluster)
    vec2 anchor = vec2(outerDir * r * 0.8, r * 1.1 * downDir);
    vec2 baseQ = p - anchor; // Store the clean, un-rotated coordinates

    float th = r * 0.04;

    // --- MARK 1 (Upper) ---
    vec2 q1 = baseQ; 
    
    // Rotate q1
    float angle1 = radians(-33.0) * outerDir * downDir;
    q1 = rotate2D(q1, angle1);

    // 1. Find the center of your capsule coordinates
    // Start: -2.0, End: 0.1. Midpoint = (-2.0 + 0.1) / 2 = -0.9

    float lineCenterX = -r * 0.95 * outerDir;
    float localX = q1.x - lineCenterX;
    q1.y += 0.2 * (localX * localX) / r;
    
    // Draw Mark 1
    float d1 = sdCapsule(q1, 
        vec2(-r * 2.1*outerDir, r * 1.4), 
        vec2( r * 0.01*outerDir, r * 1.1), 
        th);

    return d1;
}

float sdStressLines2(vec2 p, float r, float side) {
    // --- 1. SETUP ---

    if(side == -1.0)
    {

        float downDir = 1.0; 
        float outerDir = -side; 

        // Global Anchor (Center of the stress mark cluster)
        vec2 anchor = vec2(outerDir * r * 0.8, r * 1.1 * downDir);
        vec2 baseQ = p - anchor; // Store the clean, un-rotated coordinates

        float th = r * 0.04;

        // --- MARK 1 (Upper) ---
        vec2 q1 = baseQ;
        q1.y -= r * 0.2; 

        // Rotate q1
        float angle1 = radians(-61.0) * outerDir * downDir;
        q1 = rotate2D(q1, angle1);

        // 1. Find the center of your capsule coordinates
        // Start: -2.0, End: 0.1. Midpoint = (-2.0 + 0.1) / 2 = -0.9

        float lineCenterX = -r * 5.4 * outerDir;
        float localX = q1.x - lineCenterX;
        q1.y += 0.5 * (localX * localX) / r;

        // Draw Mark 1
        float d1 = sdCapsule(q1, 
            vec2(-r * 5.85*outerDir, r * 1.25), 
            vec2( -r * 4.9*outerDir, r * 1.15), 
            th);

        return d1;

    }
    else
    {
        return 1e5;
    }
    
}


// GLOOM LINES: Vertical Capsules (Returns Distance)
float sdGloomLines(vec2 p, float r, float side) {
    
    if(side == 1.0)
    {

        // --- 1. POSITIONING ---
        // Move Outside the Eye (Top-Left)
        // X: -1.0 * r (Left of the eye edge)
        // Y: -1.3 * r (Above the eye edge)
        vec2 anchor = vec2(-r * 1.0, -r * 1.3);
        vec2 q = p - anchor;

        // Optional: Rotate the whole group slightly so they don't look stiff
        q = rotate2D(q, radians(0.0));

        float th = r * 0.04; // Thickness

        // --- 2. DEFINE 3 CAPSULES ---
        
        // Line 1 (Left, Short)
        float d1 = sdCapsule(q, 
            vec2(-r * 1.07, -r*0.1), 
            vec2(-r * 1.07, r * 2.0), 
            th);

        // Line 2 (Middle, Long)
        float d2 = sdCapsule(q, 
            vec2(-21, -r * 0.1), 
            vec2(-21, r * 1.5), 
            th);

        // Line 3 (Right, Short)
        float d3 = sdCapsule(q, 
            vec2(-r * 0.15, -r * 0.1), 
            vec2(-r * 0.15, r * 1.5), 
            th);

        // Return the combined shape
        return min(d1, min(d2, d3));
    }
    else
    {
        return 1e5;
    }
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
    float r = 0.13 * min(uResolution.x, uResolution.y);
    float th = max(uThickness, 1.0);
    bool useSafeAA = false;
    vec4 col = uColor;

    // 1. DOMAIN DISTORTION
    if (uDistortMode == 1) { 
        float nx = p.x / r; 
        p.y += (nx * nx) * (uBend * 0.5) * r;
    }

   // 2. MAIN SHAPE
    float d = 1e5;
    if (uShapeID < 0.5) { // DOT
        d = sdCircle(p, r);
    }
    else if (uShapeID < 1.5) { d = sdBox(p, vec2(r, th)); }
    else if (uShapeID < 2.5) { // ARC
        // 1. DEFINE WIDTH
        // How wide the arch is. 0.9 * r matches your previous 'r * 0.85' vaguely.
        float width = r * 0.9; 
    
        // 2. DEFINE BEND HEIGHT
        // uBend comes in as -1.0 to 1.0. 
        // We multiply by 'r' to convert that percentage into pixels.
        // '1.5' is a multiplier to match the "sharpness" of your old arc.
        // If presets feel too flat, increase 1.5. If too loops, decrease it.
        float bendHeight = uBend * r * 1.5;
    
        // 3. SET CONTROL POINTS
        // A = Left, C = Right
        vec2 A = vec2(-width, 0.0);
        vec2 C = vec2( width, 0.0);
        
        // B = The Peak
        // If uBend is 0, B is at (0,0) -> Straight line.
        // If uBend is 1, B is high up -> Curve.
        vec2 B = vec2(0.0, -bendHeight); 
    
        // 4. CALCULATE DISTANCE
        // sdBezier returns the distance to the thin curve.
        // We subtract thickness (th) to give it body.
        d = sdBezier(p, A, B, C) - th;
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
        float dir = uEyeSide; vec2 pc = vec2(p.x * dir, p.y);
        vec2 a = vec2(-r * 0.45, -r * 0.35); vec2 b = vec2( r * 0.35,  0.00); vec2 c = vec2(-r * 0.45,  r * 0.35);
        d = min(sdSegment(pc, a, b) - th, sdSegment(pc, c, b) - th);
    }
    else if (uShapeID < 8.5) { // SHURIKEN
        float sharp = 0.4 + (uBend * 0.3); d = sdStar4(p, r, r * sharp);
    }
    else if (uShapeID < 9.5) { // KAWAII
        d = sdKawaii(p, r);
        // 1. BIG HIGHLIGHT (Top-Left)
        vec2 pos1 = p - vec2(-r * 0.25, -r * 0.30);
        pos1 = rotate2D(pos1, radians(37.0)); // Slight rotation for realism
        float dH1 = length(vec2(pos1.x * 0.7, pos1.y*0.95)) - (r * 0.52);

        // 2. SMALL HIGHLIGHT (Bottom-Right)
        // Moved to 0.25, 0.35 (Bottom-Right)
        // Radius increased to 0.15 * r
        vec2 pos2 = p - vec2(r * 0.40, r * 0.35);
        float dH2 = length(pos2) - (r * 0.27);

        // 3. TINY DOT (Optional, from your reference image)
        // There is a tiny third dot near the bottom in the image.
        vec2 pos3 = p - vec2(-r * 0.15, r * 0.50);
        float dH3 = length(pos3) - (r * 0.1);

        // Combine all highlights (min = union)
        float highlights = min(dH1, min(dH2, dH3));

        // 4. APPLY WHITE COLOR
        // If we are inside the highlight shapes (dist < 0), paint White.
        // smoothstep provides nice Anti-Aliasing (AA).
        float aa = 0.7; // AA width in pixels
        float highlightMask = smoothstep(aa*0.5, aa*-0.5, highlights); // Invert distance for mask

        col.rgb = mix(col.rgb, vec3(1.0), highlightMask);
    }
    else if (uShapeID < 10.5) { // SHOCKED - Ring with white center
        float dist = length(p);
        float ringWidth = uThickness; 
        
        // 1. Geometry: Define the outer shape normally
        d = dist - r;
        
        // 2. Color Mixing with AA
        // We want the boundary at (r - ringWidth).
        // smoothstep blends from 0.0 to 1.0 over 1.5 pixels width.
        float boundary = r - ringWidth;
        float aa = 0.5;
        float blend = smoothstep(boundary - 0.75*aa, boundary + 0.75*aa, dist);
        
        // Mix: 0.0 = White (Center), 1.0 = Ring Color (Edge)
        col.rgb = mix(vec3(1.0), uColor.rgb, blend);
    }

    // 3. ANGRY STRESS LINES
    if (uStressLevel > 0.9) {
        float stress = sdStressLines2(p, r, uEyeSide);
        d = min(d, stress);
    }
    else if (uStressLevel > 0.6) {
        float stress = sdStressLines1(p, r, uEyeSide);
        d = min(d, stress);
    }
    else if (uStressLevel > 0.3) {
        float stress = sdStressLines(p, r, -uEyeSide);
        d = min(d, stress);
    }

     if (uGloomLevel > 0.01) {
        float gloomDist = sdGloomLines(p, r, uEyeSide);
        d = min(d, gloomDist);
        float gloomMask = 1.0 - smoothstep(0.0, 1.5, gloomDist);
        
        vec3 gloomColor = vec3(0.37647, 0.13333, 0.61961); // Dark Purple/Blue
        
        // Apply color based on mask strength and uniform level
        col.rgb = mix(col.rgb, gloomColor, gloomMask * uGloomLevel);
    }

    // 4. COLOR & COMPOSITE
    float alpha = (useSafeAA ? aaFill_Safe(d) : aaFill_HighQuality(d));

    // 5. SHOCKED GLOOM (Vertical lines overlay)
   

    if (alpha < 0.01) discard;
    finalColor = vec4(col.rgb, alpha * col.a);
}