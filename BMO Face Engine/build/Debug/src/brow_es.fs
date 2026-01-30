#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform vec4 uColor; // Typically Black

// --- BROW PARAMS ---
uniform float uEyebrowType;  // 0=None, 1=Angry, 2=Sad, 3=Raised, 4=TickMark
uniform float uBend;         // Curve intensity
uniform float uThickness;    
uniform float uEyeBrowLength;
uniform float uEyeBrowY;      // New parameter to move eyebrow up/down
uniform float uBrowSide;      // 1.0 for left, -1.0 for right (flips the brow)
uniform float uAngle;
uniform float uBendOffset;

// Exact distance to a Quadratic Bezier Curve
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

// UTILS
vec2 rotate2D(vec2 p, float a) { 
    float s = sin(a), c = cos(a); 
    return vec2(c*p.x - s*p.y, s*p.x + c*p.y); 
}
float sdBox(vec2 p, vec2 b) { 
    vec2 d = abs(p) - b; 
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); 
}

// Signed distance to a thick circular arc (stroke)
float sdArcStroke(vec2 p, float R, float halfAngle, float strokeR)
{
    // Arc is centered at origin, opening upward by default
    // Use symmetry like your original sdArc
    p.x = abs(p.x);

    vec2 sc = vec2(sin(halfAngle), cos(halfAngle)); // direction of arc end
    float k = (sc.y*p.x > sc.x*p.y) ? length(p - sc*R) : abs(length(p) - R);
    return k - strokeR; // stroke thickness
}


void main() {
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    
    // Safety check
    float minRes = min(uResolution.x, uResolution.y);
    if (minRes < 1.0) discard;
    float r = 0.5 * minRes;
    
    float th = max(uThickness, 1.0);
    float dist = 0.0;
    
    // --- 1. ROTATION ---
    // We apply rotation first. 
    // uBrowSide ensures they rotate symmetrically (mirrored)
    float angle = radians(uAngle) * uBrowSide;
    vec2 br = rotate2D(p, angle);
    
    // Vertical Offset
    //br.y -= uEyeBrowY * r; 
    
    // --- 2. DEFINE SHAPE POINTS ---
    float halfLen = r * uEyeBrowLength; 
    
    // Determine the "Outer" direction based on which eye this is.
    // Left Eye (1.0) -> Outer is Left (-1.0)
    // Right Eye (-1.0) -> Outer is Right (1.0)
    float outerDir = -uBrowSide; 
    
    // Point A (Inner Edge): Close to the nose
    vec2 A = vec2(-outerDir * halfLen, 0.0);
    
    // Point C (Outer Edge): Far side of face
    vec2 C = vec2(outerDir * halfLen, 0.0);
    
    // Point B (Control Point): The "Peak" of the tick/curve
    // We place it towards the Outer Edge (0.8 distance) and UP
    float pivotX = outerDir * (halfLen * uBendOffset);
    float bendHeight = uBend * r * 2.0;
    
    vec2 B = vec2(pivotX, bendHeight);

    // --- 3. CALCULATE DISTANCE ---
    // Use the exact Bezier function with our precisely placed points
    float dCurve = sdBezier(br, A, B, C);
    
    // Subtract thickness
    dist = dCurve - (th * 0.85);

    // --- 4. RENDER ---
    float alpha = 1.0 - smoothstep(-0.75, 0.75, dist);
    if (alpha < 0.01) discard;
    finalColor = vec4(uColor.rgb, alpha * uColor.a);
}