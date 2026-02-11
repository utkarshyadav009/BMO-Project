#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

// --------------------------------------------------------
// UNIFORMS
// --------------------------------------------------------
uniform vec2 uResolution;
uniform float uScale;        // 0.66 on small screens, 2.0 on 4K
uniform float uTime;
uniform float uSide;         // 1.0 = Left, -1.0 = Right

// Tear Params
uniform float uTearsLevel;   // 0.0-0.4: Pool, 0.4-0.75: Drip, 0.75+: Wail
uniform vec4 uTearColor;     // Base color from C++

// Blush Params
uniform int uShowBlush;
uniform int uBlushMode;      // 0 = Pink, 1 = Green
uniform vec4 uBlushColor;

// --------------------------------------------------------
// MATH HELPERS
// --------------------------------------------------------
vec2 rotate2D(vec2 p, float a) { 
    float s = sin(a), c = cos(a); 
    return vec2(c*p.x - s*p.y, s*p.x + c*p.y); 
}

// --------------------------------------------------------
// SDF FUNCTIONS
// --------------------------------------------------------
float sdTriangle(vec2 p, float r) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x + k*p.y > 0.0 ) p = vec2(p.x - k*p.y, -k*p.x - p.y) / 2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p) * sign(p.y);
}

float sdCapsule(vec2 p, vec2 a, vec2 b, float r) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// --------------------------------------------------------
// SHAPE GENERATION
// --------------------------------------------------------
float getTearShapeDist(vec2 p, float level, float t) {
    float d = 1e5;

    // MODE 1: POOL (Static Oval)
    if (level < 0.4) {
        float yOffset = 110.0 * uScale;
        vec2 poolPos = p - vec2(0.0, yOffset);
        float radius = 37.0 * uScale;
        // Squash Y (1.6) to make it a flat oval
        d = length(vec2(poolPos.x, poolPos.y * 1.6)) - radius;
    }
    
    // MODE 2: DRIPS
    else if (level < 0.75) {
        float size = 35.0 * uScale;
        float speed = 0.2 + (level * 0.5);
        float fallDist = uResolution.y + uScale;

        for(float i = 0.0; i < 5.0; i++) {
            float offset = i / 5.0;
            float cycle = fract((t * speed) + offset);
            
            float yPos = (170.0 * uScale) + (cycle * fallDist);
            float xPos = -(150.0 * uScale) * uSide; // Jitter X

            vec2 dropPos = p - vec2(xPos, yPos);
            dropPos.y = -dropPos.y; // Flip for triangle
            
            d = min(d, sdTriangle(dropPos, size));
        }
    } 
    
    // MODE 3: WAIL STREAM
    else { 
        float width = (60.0 + (level * 20.0)) * uScale;
        float wave = sin(p.y * 0.04 - t * 6.0) * (12.0 * uScale);
        float distToStream = abs(p.x + wave) - width;
        
        // Frown Curve Cut
        float curvature = 0.004 / uScale;
        float curveY = (p.x * p.x) * curvature;
        float topEdge = -(p.y - curveY) - (10.0 * uScale);
        
        d = max(distToStream, topEdge);
    }

    return d;
}

// --------------------------------------------------------
// STYLE HELPERS
// --------------------------------------------------------

// Calculates the Gradient, Caustics, and Highlights for the Pool Mode
vec4 getPoolStyle(vec2 p, float d) {
    // 1. Setup Geometry (Matching hardcoded values)
    float yOffset = 140.0 * uScale;
    float xOffset = -5.0 * uScale;
    vec2 center = p - vec2(xOffset, yOffset);

    // 2. Base Alpha (70% Opacity)
    float w = fwidth(d);
    float alpha = (1.0 - smoothstep(-w, w, d)) * 0.7;
    if (alpha < 0.01) return vec4(0.0);

    // 3. Gradient Logic
    // Vertical radius derived from 55.0 radius / 1.6 squash
    float verticalRadius = (30.0 * uScale) / 1.6;
    float normY = center.y / verticalRadius;

    vec3 colTop = vec3(0.1, 0.6, 0.4); 
    vec3 colBot = vec3(0.5, 0.95, 0.7);
    float grad = smoothstep(-0.8, 0.8, normY);
    vec3 finalRGB = mix(colTop, colBot, grad);

    // 4. Caustic Rim (Bottom Glow)
    float rim = smoothstep(verticalRadius - (8.0 * uScale), verticalRadius, center.y);
    finalRGB += vec3(0.3, 0.5, 0.4) * rim * 0.8;

    // 5. Highlight Logic
    vec2 hPos = center - vec2(-10.0 * uScale, -35.0 * uScale);
    hPos = rotate2D(hPos, radians(25.0));

    // Highlight Shape (1.5 squash, 16.0 size)
    float hDist = length(vec2(hPos.x, hPos.y * 1.5)) - (16.0 * uScale);
    float hAlpha = 1.0 - smoothstep(0.0, 2.0 * uScale, hDist);

    // Composite Highlight
    finalRGB = mix(finalRGB, vec3(1.0), hAlpha);
    alpha = max(alpha, hAlpha); // Boost opacity for highlight

    return vec4(finalRGB, alpha);
}

// --------------------------------------------------------
// COMPONENT RENDERERS
// --------------------------------------------------------
vec4 getBlush(vec2 p) {
    if (uShowBlush == 0) return vec4(0.0);

    // --- MODE 3: ZIGZAG SCRIBBLE (Black) ---
    if (uBlushMode == 2) {
        float r = 0.4 * min(uResolution.x, uResolution.y);
        
        // 1. POSITION & ROTATION
        vec2 pos = p - vec2(0.0, 1.2 * r); 

        // 2. CONFIGURATION
        float width = r * 0.9;         // Total width
        float height = r * 0.25;       // Height of the peaks
        float thickness = r * 0.05;    // Thickness of the line
        int segments = 5;              // How many lines in the zigzag

        // 3. CONSTRUCT WAVE USING CAPSULES
        float d = 1e5; // Start with a large distance
        
        // Calculate step size based on total width
        float stepX = width / float(segments);
        float startX = -width * 0.5;

        for (int i = 0; i < segments; i++) {
            // Determine X coordinates for this segment
            float xA = startX + (float(i) * stepX);
            float xB = startX + (float(i + 1) * stepX);

            // Determine Y coordinates (Oscillate Up/Down)
            // Even numbers go Down (-height), Odd numbers go Up (+height)
            float yA = (mod(float(i), 2.0) == 0.0) ? -height : height;
            float yB = (mod(float(i + 1), 2.0) == 0.0) ? -height : height;

            // Create a capsule between Point A and Point B
            float dSeg = sdCapsule(pos, vec2(xA, yA), vec2(xB, yB), thickness);
            
            // Combine with the rest of the shape (Union)
            d = min(d, dSeg);
        }

        // 4. RENDER
        // Use smooth edges for a nice marker pen look
        float alpha = 1.0 - smoothstep(-0.5, 1.5, d);
        
        // Return Black (0,0,0)
        return vec4(0.0, 0.0, 0.0, alpha);
    }

    // --- MODE 0/1: STANDARD OVALS ---
    else {
        bool isPink = (uBlushMode < 1);
        vec4 color = isPink ? vec4(1.0, 0.784, 0.847, 1.0) : vec4(0.470, 0.654, 0.454, 1.0);

        float r = 0.4 * min(uResolution.x, uResolution.y);
        vec2 pos = p - vec2(0.0, 1.2 * r);
        
        // Ellipse Shape
        float d = length(pos / vec2(1.0, 0.6)) - (r * 0.85);

        float alpha = 0.0;
        if (isPink) {
            float delta = fwidth(d);
            alpha = 1.0 - smoothstep(-delta, delta, d);
        } else {
            alpha = 1.0 - smoothstep(-0.5 * r, 0.5 * r, d);
            alpha *= 0.9;
        }

        return vec4(color.rgb, alpha);
    }
}

vec4 getTears(vec2 p) {
    if (uTearsLevel <= 0.01) return vec4(0.0);

    // 1. Get Distance Field
    float d = getTearShapeDist(p, uTearsLevel, uTime);

    // 2. Select Rendering Mode
    
    // --- MODE A: POOL (Gradient + Gloss) ---
    if (uTearsLevel <= 0.4) {
        return getPoolStyle(p, d);
    }
    
    // --- MODE B: DRIP/WAIL (Flat + Outline) ---
    else {
        vec3 baseColor;
        if (uTearsLevel < 0.75) baseColor = vec3(0.180, 0.713, 0.384); // Drip
        else baseColor = vec3(0.403, 0.839, 0.639); // Wail

        vec3 outlineColor = vec3(0.584, 0.952, 0.772);
        float outlineWidth = 4.0 * uScale;

        float w = fwidth(d);
        float fillAlpha = 1.0 - smoothstep(-w, w, d);
        float borderAlpha = 1.0 - smoothstep(outlineWidth - w, outlineWidth + w, d);

        vec3 finalRGB = outlineColor;
        finalRGB = mix(finalRGB, baseColor, fillAlpha);

        return vec4(finalRGB, borderAlpha);
    }
}

// --------------------------------------------------------
// MAIN
// --------------------------------------------------------
void main() {
    // 1. Normalize Coordinates (Top-Center Origin)
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    p.y += uResolution.y * 0.5;

    // 2. Render Layers
    vec4 blush = getBlush(p);
    vec4 tear  = getTears(p);

    // 3. Composite (Painter's Algorithm)
    vec3 colorAccum = vec3(0.0);
    float alphaAccum = 0.0;

    // Blush
    if (blush.a > 0.0) {
        colorAccum = blush.rgb;
        alphaAccum = blush.a;
    }

    // Tears (Top Layer)
    if (tear.a > 0.0) {
        colorAccum = mix(colorAccum, tear.rgb, tear.a);
        alphaAccum = max(alphaAccum, tear.a);
    }

    // 4. Output
    if (alphaAccum < 0.01) discard;
    finalColor = vec4(colorAccum, alphaAccum);
}