#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform float uScale;        // [CRITICAL] 0.66 on small screens, 2.0 on 4K
uniform float uTime;
uniform float uSide;         // 1.0 = Left, -1.0 = Right

// --- TEAR PARAMS ---
uniform float uTearsLevel;   // 0.0-0.4: Pool, 0.4-0.75: Drip, 0.75+: Wail
uniform vec4 uTearColor;     // Base color from C++

// --- BLUSH PARAMS ---
uniform int uShowBlush;
uniform int uBlushMode;      // 0 = Pink (Solid), 1 = Green (Soft)
uniform vec4 uBlushColor;


//Helper functoin to rotate a 2D point around the origin
vec2 rotate2D(vec2 p, float a) { float s = sin(a), c = cos(a); return vec2(c*p.x - s*p.y, s*p.x + c*p.y); }

// --------------------------------------------------------
// SDF SHAPE FUNCTIONS
// --------------------------------------------------------

// Standard Triangle (used for Drips)
float sdTriangle(vec2 p, float r) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x + k*p.y > 0.0 ) p = vec2(p.x - k*p.y, -k*p.x - p.y) / 2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p) * sign(p.y);
}

// --------------------------------------------------------
// SHAPE LOGIC
// --------------------------------------------------------

// Calculates the Distance Field for the Tear Shape
float getTearShapeDist(vec2 p, float level, float t) {
    float d = 1e5;

    // --- MODE 1: POOL OF TEARS (Static Oval) ---
    if (level < 0.4) {
        float yOffset = 130.0 * uScale;
        vec2 poolPos = p - vec2(0.0, yOffset);

        // Flattened Oval
        float squashFactor = 1.6;
        float radius = 55.0 * uScale;
        d = length(vec2(poolPos.x, poolPos.y * squashFactor)) - radius;
    }
    
    // --- MODE 2: CLASSIC DRIP ---
    else if (level < 0.75) {
        float size = 35.0 * uScale;
        float speed = 0.2 + (level * 0.5);
        float fallDist = uResolution.y + uScale;

        for(float i = 0.0; i < 5.0; i++) {
            float offset = i / 5.0;
            float cycle = fract((t * speed) + offset);
            
            float yPos = (170.0 * uScale) + (cycle * fallDist);
            float xPos = -(120.0 * uScale) * uSide; // Jitter X

            vec2 dropPos = p - vec2(xPos, yPos);
            dropPos.y = -dropPos.y; // Flip for Triangle orientation
            
            float dTri = sdTriangle(dropPos, size);
            d = min(d, dTri);
        }
    } 
    
    // --- MODE 3: WAIL STREAM ---
    else { 
        float width = (60.0 + (level * 20.0)) * uScale;
        
        // Sine Wave + Box
        float wave = sin(p.y * 0.04 - t * 6.0) * (12.0 * uScale);
        float distToStream = abs(p.x + wave) - width;
        
        // Curved Top Cut (Frown Shape)
        float curvature = 0.004 / uScale;
        float curveY = (p.x * p.x) * curvature;
        float topEdge = -(p.y - curveY) - (10.0 * uScale);
        
        d = max(distToStream, topEdge);
    }

    return d;
}

// --------------------------------------------------------
// COMPONENT RENDERERS
// --------------------------------------------------------

// Returns: vec4(r, g, b, alpha)
vec4 getBlush(vec2 p) {
    if (uShowBlush == 0) return vec4(0.0);

    // 1. Setup Colors & Mode
    bool isPink = (uBlushMode < 1); // 0 = Pink
    vec4 color = isPink 
        ? vec4(1.0, 0.784, 0.847, 1.0)  // Pink
        : vec4(0.470, 0.654, 0.454, 1.0); // Green

    // 2. Shape (Ellipse)
    float r = 0.4 * min(uResolution.x, uResolution.y);
    float blushY = 1.2 * r;
    vec2 pos = p - vec2(0.0, blushY);
    
    // Squash Y to make ellipse
    float d = length(pos / vec2(1.0, 0.6)) - (r * 0.85);

    // 3. Render
    float alpha = 0.0;
    if (isPink) {
        // Solid, crisp edges
        float delta = fwidth(d);
        alpha = 1.0 - smoothstep(-delta, delta, d);
    } else {
        // Soft, bleeding edges
        alpha = 1.0 - smoothstep(-0.5 * r, 0.5 * r, d);
        alpha *= 0.9; // Slightly transparent
    }

    return vec4(color.rgb, alpha);
}

vec4 getTears(vec2 p) {
    if (uTearsLevel <= 0.01) return vec4(0.0);

    // 1. Get Base Shape Distance
    float d = getTearShapeDist(p, uTearsLevel, uTime);
    
    // 2. Determine Base Color
    vec3 baseColor;
    if (uTearsLevel <= 0.4) {
        baseColor = vec3(0.325, 0.721, 0.411); // Pool Green
    } else if (uTearsLevel < 0.75) {
        baseColor = vec3(0.180, 0.713, 0.384); // Drip Green
    } else {
        baseColor = vec3(0.403, 0.839, 0.639); // Wail Green
    }

    // --- BRANCH: POOL MODE (Highlight, No Outline) ---
    if (uTearsLevel <= 0.4) {
        // 1. SETUP GEOMETRY (Your hardcoded values)
        float yOffset = 140.0 * uScale;
        float xOffset = -5.0 * uScale;
        vec2 center = p - vec2(xOffset, yOffset);

        // 2. BASE ALPHA (Your 70% opacity logic)
        float alpha = (1.0 - smoothstep(-0.5, 0.5, d)) * 0.7;
        
        // --- NEW: GRADIENT & CAUSTIC LOGIC ---
        // We use 'center.y' to create the vertical gradient.
        // Since radius was 55.0 and squash was 1.6, the height is approx 34.0.
        // We normalize Y from -1 (top) to 1 (bottom).
        float verticalRadius = (55.0 * uScale) / 1.6;
        float normY = center.y / verticalRadius;

        // A. Colors: Deep Teal (Top) -> Bright Mint (Bottom)
        vec3 colTop = vec3(0.1, 0.6, 0.4); 
        vec3 colBot = vec3(0.5, 0.95, 0.7);
        
        // B. Gradient Mix
        float grad = smoothstep(-0.8, 0.8, normY);
        vec3 finalRGB = mix(colTop, colBot, grad);

        // C. Caustic Rim (Bottom Glow)
        // Adds a bright rim at the very bottom edge
        float rim = smoothstep(verticalRadius - (8.0 * uScale), verticalRadius, center.y);
        finalRGB += vec3(0.3, 0.5, 0.4) * rim * 0.8;

        // --- HIGHLIGHT LOGIC (Your exact transforms) ---
        vec2 hPos = center - vec2(-20.0 * uScale, -25.0 * uScale);
        
        // Rotation (Yours)
        hPos = rotate2D(hPos, radians(25.0));

        // Highlight Shape (Yours: 1.5 squash, 16.0 size)
        float hDist = length(vec2(hPos.x, hPos.y * 1.5)) - (16.0 * uScale);
        
        // Render Highlight (Soft edge)
        float hAlpha = 1.0 - smoothstep(0.0, 2.0 * uScale, hDist);
        
        // --- COMPOSITE ---
        // Mix White Highlight on top of Gradient
        finalRGB = mix(finalRGB, vec3(1.0), hAlpha);
        
        // Boost alpha for the highlight so it pops
        alpha = max(alpha, hAlpha); 

        return vec4(finalRGB, alpha);
    }
    
    // --- BRANCH: DRIP/WAIL MODE (Outline, No Highlight) ---
    else {
        vec3 outlineColor = vec3(0.584, 0.952, 0.772);
        float outlineWidth = 4.0 * uScale;

        // Fill Alpha
        float fillAlpha = 1.0 - smoothstep(-0.5, 0.5, d);
        
        // Outline Alpha
        float borderAlpha = 1.0 - smoothstep(outlineWidth - 0.5, outlineWidth + 0.5, d);

        // Composite: Start with Outline, mix in Base Fill
        // This ensures the fill covers the inner part of the outline
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

    // 2. Accumulators
    vec3 colorAccum = vec3(0.0);
    float alphaAccum = 0.0;

    // 3. Render Layers
    vec4 blush = getBlush(p);
    vec4 tear  = getTears(p);

    // 4. Mix Layers (Painter's Algorithm: Blush -> Tears)
    
    // Layer 1: Blush
    if (blush.a > 0.0) {
        colorAccum = blush.rgb;
        alphaAccum = blush.a;
    }

    // Layer 2: Tears (On Top)
    if (tear.a > 0.0) {
        // Standard Alpha Blending: Result = Foreground * SrcAlpha + Background * (1 - SrcAlpha)
        colorAccum = mix(colorAccum, tear.rgb, tear.a);
        
        // Accumulate Alpha (Simple max usually works for UI/Decals)
        alphaAccum = max(alphaAccum, tear.a);
    }

    // 5. Output
    if (alphaAccum < 0.01) discard;
    finalColor = vec4(colorAccum, alphaAccum);
}