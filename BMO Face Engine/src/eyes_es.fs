#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform vec4 uColor;

// --- PARAMS ---
uniform float uShapeID;      // 0=Dot, 1=Line, 2=Arc, 3=Cross, 4=Star, 5=Heart, 6=Spiral, 7=Chevron
uniform float uBend;         // shape-dependent params
uniform float uThickness;    // line half-thickness
uniform float uPupilSize;    

// NEW PARAMS
uniform float uEyebrow;      
uniform float uEyebrowY;     
uniform float uTears;   
uniform float uSpiralSpeed;     
uniform float uTime; 

// ---------------------------
// SDF helpers
// ---------------------------
float sdCircle(vec2 p, float r) { return length(p) - r; }

float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float denom = dot(ba, ba);
    float h = (denom > 1e-8) ? clamp(dot(pa, ba) / denom, 0.0, 1.0) : 0.0;
    return length(pa - ba * h);
}

vec2 rotate2D(vec2 p, float a) {
    float s = sin(a), c = cos(a);
    return vec2(c*p.x - s*p.y, s*p.x + c*p.y);
}

float sdArc(vec2 p, float ra, float rb, float aperture) {
    vec2 sc = vec2(sin(aperture), cos(aperture));
    p.x = abs(p.x);
    float k = (sc.y*p.x > sc.x*p.y) ? length(p - sc*ra) : abs(length(p) - ra);
    return k - rb;
}

// ---------------------------
// STAR FUNCTIONS
// ---------------------------

// 5-POINT STAR (Fixed for Scale)
float sdStar5(vec2 p, float r, float rf) {
    // 1. Normalize to Unit Space (Fixes the black fill issue)
    p /= r;
    float inner = rf / r; // Normalize inner radius

    const vec2 k1 = vec2(0.809016994375, -0.587785252292);
    const vec2 k2 = vec2(-k1.x, k1.y);
    
    p.x = abs(p.x);
    p -= 2.0 * max(dot(k1, p), 0.0) * k1;
    p -= 2.0 * max(dot(k2, p), 0.0) * k2;
    p.x = abs(p.x);
    p.y -= 1.0; // Top point is now at 1.0
    
    vec2 ba = inner * vec2(-k1.y, k1.x) - vec2(0, 1);
    float h = clamp(dot(p, ba) / dot(ba, ba), 0.0, 1.0);
    
    // 2. Denormalize return value back to Pixels
    return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y) * r;
}

// 4-POINT SHURIKEN (Fixed Sign)
float sdStar4(vec2 p, float r, float rf) {
    // 1. Normalize
    p /= r;
    float k = rf / r; // inner ratio
    
    // 2. Symmetry (1/8th sector)
    p = abs(p);
    if (p.x < p.y) p = p.yx;
    
    // 3. Edge definition
    const float s = 0.70710678; // sin(45)
    vec2 p1 = vec2(1.0, 0.0);
    vec2 p2 = vec2(k*s, k*s); // Inner valley at 45 deg
    
    vec2 e = p2 - p1;
    vec2 w = p - p1;
    
    // 4. Distance to edge
    vec2 b = w - e*clamp(dot(w,e)/dot(e,e), 0.0, 1.0);
    float d = length(b);
    
    // 5. Sign Correction
    // Cross product: e.x*w.y - e.y*w.x
    // det > 0 means we are on the "inside" of the edge vector.
    // SDF Convention: Inside must be NEGATIVE.
    float det = e.x*w.y - e.y*w.x;
    float sgn = (det > 0.0) ? -1.0 : 1.0; // [FIXED] Flipped sign here
    
    return d * sgn * r;
}

float sdHeart(vec2 p) {
    p.x = abs(p.x);
    p.y += 0.25;
    float a = atan(p.x, p.y) / 3.14159265;
    float r = length(p);
    float h = abs(a);
    float d = r - (0.8 - 0.2*h);
    d = max(d, -(p.y - 0.15));
    return d;
}

// ---------------------------
// ARTIFACT-FREE SPIRAL
// ---------------------------
float sdSpiral(vec2 p, float R, float turns, float halfTh, float thetaOffset, float dir, float osc)
{
    const float TWO_PI = 6.2831853;

    // 1. Transform Space
    p.x *= dir; // Flip X
    float r = length(p);
    float a = atan(p.y, p.x);
    a -= thetaOffset; // Rotate

    // 2. Dynamic Spacing
    float spacing = (R / max(turns, 1e-4)) * (1.0 + 0.05 * osc);

    // 3. Estimate closest winding
    // r ~= spacing * (n + a/2pi)  ->  n ~= r/spacing - a/2pi
    float n_est = (r / spacing) - (a / TWO_PI);
    float n_center = round(n_est);

    // 4. Check Neighbors (The "Iterative" Fix)
    // We check the estimated ring, plus the inner and outer neighbors.
    // This removes the "seam" artifact where the winding number jumps.
    
    float minD = 1e5;

    for(float i = -1.0; i <= 1.0; i += 1.0) {
        float k = n_center + i;

        // Calculate the angle "phi" for this specific winding
        // phi goes from 0 to infinity along the spiral line
        float phi = TWO_PI * k + a;

        // 5. CLAMP the spiral (The Fade Fix)
        // By clamping phi, we restrict the geometry to exactly the requested turns.
        // If the pixel is past the end, "phi_clamped" becomes the tip angle.
        // The distance calculation then naturally becomes the distance to the tip cap.
        float phi_clamped = clamp(phi, 0.0, turns * TWO_PI);

        // Calculate the point on the spiral curve at phi_clamped
        float r_curve = spacing * (phi_clamped / TWO_PI);
        
        // Note: We must use phi_clamped for the position!
        vec2 p_curve = vec2(cos(phi_clamped), sin(phi_clamped)) * r_curve;

        // Distance from pixel p to this point on the curve
        float dist = distance(p, p_curve);
        minD = min(minD, dist);
    }

    // Subtract thickness
    return minD - halfTh;
}

// Replace your old aaFill with this
float aaFill(float d) {
    // Don't use fwidth(d). It explodes on discontinuities.
    // Use a fixed pixel width (1.5 is a safe blur amount).
    return 1.0 - smoothstep(-0.75, 0.75, d);
}

// // Proper AA from derivatives
// float aaFill(float d) {
//     float w = fwidth(d);
//     w = max(w, 1.0);
//     return 1.0 - smoothstep(0.0, w, d);
// }

void main() {
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    float r = 0.4 * min(uResolution.x, uResolution.y);
    float th = max(uThickness, 1.0); 

    float d = 1e10;

    // --- 1) BASE EYE SHAPE ---
    if (uShapeID < 0.5) { // 0: DOT
        d = sdCircle(p, r);
        if (uPupilSize > 0.001) {
            vec2 hlPos = p - vec2(r * 0.28, -r * 0.28);
            float hl = sdCircle(hlPos, r * 0.28);
            d = max(d, -hl * clamp(uPupilSize, 0.0, 1.0));
        }
    }
    else if (uShapeID < 1.5) { // 1: LINE
        d = sdBox(p, vec2(r, th));
    }
    else if (uShapeID < 2.5) { // 2: ARC
        float bend = clamp(uBend, -1.0, 1.0);
        vec2 pp = p;
        pp.y *= (bend >= 0.0) ? -1.0 : 1.0;
        float a = mix(1.35, 0.85, abs(bend));
        float w = max(1.2, 0.001); // avoid divide by 0
        pp.x /= w; 
        d = sdArc(pp, r * 0.85, th, a);
    }
    else if (uShapeID < 3.5) { // 3: CROSS
        vec2 p1 = rotate2D(p, 0.785398);
        vec2 p2 = rotate2D(p, -0.785398);
        d = min(sdBox(p1, vec2(r, th)), sdBox(p2, vec2(r, th)));
    }
    else if (uShapeID < 4.5) { // 4: STAR (5-POINT)
        // Map uBend (-1 to 1) to Inner Radius (0.2 to 0.7)
        // Default (uBend=0) gives 0.5 ratio
        float sharp = 0.5 + (uBend * 0.3); 
        d = sdStar5(p, r, r * sharp);
    }
    else if (uShapeID < 5.5) { // 5: HEART
        vec2 hp = p / (r * 0.95);
        d = sdHeart(hp) * (r * 0.95);
    }
    else if (uShapeID < 6.5) { // 6: SPIRAL
        float turns = 3.0;
        float R = r * 0.92; // Max radius
        float dir = (uBend >= 0.0) ? 1.0 : -1.0;

        float spin      = uTime * uSpiralSpeed;           // rigid rotation
        float osc       = sin(uTime * 1.7);      // breathing

        vec2 pSpin = rotate2D(p, spin);

        d = sdSpiral(pSpin, R, turns, th, 0, dir, osc);
        
    }
    else if (uShapeID < 7.5){ // 7: CHEVRON
        float dir = (uBend >= 0.0) ? 1.0 : -1.0;
        vec2 pc = vec2(p.x * dir, p.y);
        vec2 a = vec2(-r * 0.45, -r * 0.35);
        vec2 b = vec2( r * 0.35,  0.00);
        vec2 c = vec2(-r * 0.45,  r * 0.35);
        d = min(sdSegment(pc, a, b) - th, sdSegment(pc, c, b) - th);
    }
    else { // 8: SHURIKEN (4-POINT STAR)
        // Map uBend to sharpness
        float sharp = 0.4 + (uBend * 0.3);
        d = sdStar4(p, r, r * sharp);
    }

    // --- 2) EYEBROWS ---
    if (uEyebrow > 0.5) {
        float browBaseY = -r * 1.25 + uEyebrowY;
        vec2 bp = p - vec2(0.0, browBaseY);
        float mode = uEyebrow;
        float browAngle = 0.0;
        float raise = 0.0;
        if (mode < 1.5) { browAngle = 0.45; }
        else if (mode < 2.5) { browAngle = -0.45; }
        else { browAngle = 0.0; raise = -r * 0.12; }
        bp.y += raise;
        vec2 br = rotate2D(bp, browAngle);
        d = min(d, sdBox(br, vec2(r * 0.75, th * 0.85)));
    }

    // --- 3) TEARS ---
    if (uTears > 0.01) {
        float t = clamp(uTears, 0.0, 1.0);
        vec2 start = vec2(0.0, r * 0.65);
        vec2 end   = vec2(0.0, r * (1.75 + 0.5 * t));
        vec2 tp = p;
        tp.x -= sin(tp.y * 0.03) * (r * 0.05) * t;
        float stream = sdSegment(tp, start, end) - (th * 0.65) * mix(0.8, 1.3, t);
        float drop = sdCircle(tp - end, r * 0.10 * mix(0.8, 1.25, t));
        float mask = (p.y > start.y) ? -1e5 : 1e5;
        d = min(d, max(min(stream, drop), mask));
    }

    // --- Final shading ---
    vec4 color = uColor;
    color.a *= aaFill(d);

    if (color.a < 0.01) discard;
    finalColor = color;
}


