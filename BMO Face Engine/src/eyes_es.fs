#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;   // render target size in pixels
uniform vec4 uColor;

// --- PARAMS ---
uniform float uShapeID;      // 0=Dot, 1=Line, 2=Arc, 3=Cross, 4=Star, 5=Heart, 6=Spiral, 7=Chevron
uniform float uBend;         // shape-dependent: arc curvature / chevron direction / spiral twist
uniform float uThickness;    // line half-thickness in pixels
uniform float uPupilSize;    // 0..1 (dot highlight cutout strength)

// NEW PARAMS
uniform float uEyebrow;      // 0=None, 1=Angry, 2=Sad, 3=Raised
uniform float uEyebrowY;     // vertical offset (pixels, positive moves eyebrow down)
uniform float uTears;        // 0..1

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
    // Avoid NaNs if a==b
    float h = (denom > 1e-8) ? clamp(dot(pa, ba) / denom, 0.0, 1.0) : 0.0;
    return length(pa - ba * h);
}

vec2 rotate2D(vec2 p, float a) {
    float s = sin(a), c = cos(a);
    return vec2(c*p.x - s*p.y, s*p.x + c*p.y);
}

// Rounded arc SDF (thin ring section)
// ra = radius, rb = half thickness, aperture in radians (roughly)
float sdArc(vec2 p, float ra, float rb, float aperture) {
    vec2 sc = vec2(sin(aperture), cos(aperture));
    p.x = abs(p.x);
    float k = (sc.y*p.x > sc.x*p.y) ? length(p - sc*ra) : abs(length(p) - ra);
    return k - rb;
}

// 5-point star SDF (unchanged, pretty good)
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
    return length(p - ba*h) * sign(p.y*ba.x - p.x*ba.y);
}

// Simple heart SDF (scaled to fit nicely)
// Source style: common analytic heart approximation
float sdHeart(vec2 p) {
    // Normalize-ish heart; caller should scale p
    p.x = abs(p.x);
    p.y += 0.25;
    float a = atan(p.x, p.y) / 3.14159265;
    float r = length(p);
    float h = abs(a);
    float d = r - (0.8 - 0.2*h);
    // Push in the top cleft a bit
    d = max(d, -(p.y - 0.15));
    return d;
}

// Cheap spiral distance approximation, masked to a circle
float sdSpiral(vec2 p, float outerR, float turns, float lineHalfThick) {
    float r = length(p);
    if (r > outerR) return r - outerR;

    float ang = atan(p.y, p.x);                // -pi..pi
    float t = (ang + 3.14159265) / (6.2831853); // 0..1
    // Spiral target radius decreases with angle (Archimedean-ish)
    float target = outerR * (1.0 - t * turns);
    // Repeat turns by wrapping target (keeps multiple windings inside)
    float k = outerR / max(turns, 0.001);
    float dr = abs(mod(target - r + 0.5*k, k) - 0.5*k);
    return dr - lineHalfThick;
}

// Proper AA from derivatives
float aaFill(float d) {
    float w = fwidth(d);
    // If derivatives are zero (rare), fall back to 1px-ish
    w = max(w, 1.0);
    return 1.0 - smoothstep(0.0, w, d);
}

void main() {
    // Centered pixel space
    vec2 p = (fragTexCoord - 0.5) * uResolution;

    float r = 0.4 * min(uResolution.x, uResolution.y);
    float th = max(uThickness, 1.0); // half thickness in pixels

    float d = 1e10;

    // ---------------------------
    // 1) BASE EYE SHAPE
    // ---------------------------
    if (uShapeID < 0.5) { // 0: DOT / OVAL
        d = sdCircle(p, r);

        // highlight cutout (controlled by uPupilSize 0..1)
        if (uPupilSize > 0.001) {
            vec2 hlPos = p - vec2(r * 0.28, -r * 0.28);
            float hl = sdCircle(hlPos, r * 0.28);
            // cut-out inside the circle (keep it stable with max)
            d = max(d, -hl * clamp(uPupilSize, 0.0, 1.0));
        }
    }
    else if (uShapeID < 1.5) { // 1: LINE
        d = sdBox(p, vec2(r, th));
    }
    else if (uShapeID < 2.5) { // 2: ARC (smile/frown)
        // Convention: uBend > 0 = smile (arc up), uBend < 0 = frown (arc down)
        float bend = clamp(uBend, -1.0, 1.0);
        vec2 pp = p;
        pp.y *= (bend >= 0.0) ? -1.0 : 1.0;

        // aperture: wider when less bend, tighter when more bend
        float a = mix(1.35, 0.85, abs(bend)); // radians-ish
        d = sdArc(pp, r * 0.85, th, a);
    }
    else if (uShapeID < 3.5) { // 3: CROSS (X)
        // Two thin rotated rectangles
        vec2 p1 = rotate2D(p, 0.785398163);  // +45
        vec2 p2 = rotate2D(p, -0.785398163); // -45
        float d1 = sdBox(p1, vec2(r, th));
        float d2 = sdBox(p2, vec2(r, th));
        d = min(d1, d2);
    }
    else if (uShapeID < 4.5) { // 4: STAR
        d = sdStar5(p, r, r * 0.5);
    }
    else if (uShapeID < 5.5) { // 5: HEART
        // Scale p into heart space so it fills similarly to other shapes
        vec2 hp = p / (r * 0.95);
        d = sdHeart(hp) * (r * 0.95);
    }
    else if (uShapeID < 6.5) { // 6: SPIRAL
        float turns = mix(1.5, 3.5, clamp(abs(uBend), 0.0, 1.0));
        d = sdSpiral(p, r * 0.85, turns, th);
    }
    else { // 7: CHEVRON (< or >)
        // uBend sign selects direction
        float dir = (uBend >= 0.0) ? 1.0 : -1.0;
        vec2 pc = vec2(p.x * dir, p.y);

        // Build a "V" from two segments, then thicken
        vec2 a = vec2(-r * 0.45, -r * 0.35);
        vec2 b = vec2( r * 0.35,  0.00);
        vec2 c = vec2(-r * 0.45,  r * 0.35);

        float s1 = sdSegment(pc, a, b) - th;
        float s2 = sdSegment(pc, c, b) - th;

        d = min(s1, s2);
    }

    // ---------------------------
    // 2) EYEBROWS
    // ---------------------------
    if (uEyebrow > 0.5) {
        // baseline: above the eye
        float browBaseY = -r * 1.25 + uEyebrowY;
        vec2 bp = p - vec2(0.0, browBaseY);

        float mode = uEyebrow;
        float browAngle = 0.0;
        float raise = 0.0;

        // 1: Angry (tilt down toward center)
        if (mode < 1.5) { browAngle = 0.45; }
        // 2: Sad (tilt up toward center)
        else if (mode < 2.5) { browAngle = -0.45; }
        // 3: Raised (higher + slight curve impression via offset)
        else { browAngle = 0.0; raise = -r * 0.12; }

        bp.y += raise;
        vec2 br = rotate2D(bp, browAngle);

        // Eyebrow as a thick segment/box
        float dBrow = sdBox(br, vec2(r * 0.75, th * 0.85));
        d = min(d, dBrow);
    }

    // ---------------------------
    // 3) TEARS
    // ---------------------------
    if (uTears > 0.01) {
        float t = clamp(uTears, 0.0, 1.0);

        // Start point at bottom of eye
        vec2 start = vec2(0.0, r * 0.65);
        // End point downwards
        vec2 end   = vec2(0.0, r * (1.75 + 0.5 * t));

        // slight waviness (still stable)
        vec2 tp = p;
        tp.x -= sin(tp.y * 0.03) * (r * 0.05) * t;

        float stream = sdSegment(tp, start, end) - (th * 0.65) * mix(0.8, 1.3, t);

        // droplet at end
        float drop = sdCircle(tp - end, r * 0.10 * mix(0.8, 1.25, t));

        // Only below the eye: mask tears above the start point
        float mask = (p.y > start.y) ? -1e5 : 1e5; // keep below start (remember y grows downward in this p-space)
        float dTear = min(stream, drop);
        dTear = max(dTear, mask);

        d = min(d, dTear);
    }

    // ---------------------------
    // Final shading
    // ---------------------------
    vec4 color = uColor;
    color.a *= aaFill(d);

    if (color.a < 0.01) discard;
    finalColor = color;
}
