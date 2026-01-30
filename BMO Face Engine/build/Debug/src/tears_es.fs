#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution;
uniform float uScale;        // [CRITICAL] 0.66 on small screens, 2.0 on 4K

// --- TEAR PARAMS ---
uniform float uTearsLevel;   // 0.0-0.4: Triangle, 0.4-0.75: Drip, 0.75+: Wail
uniform int uBlushmode;       
uniform float uTime; 
uniform float uSide;         // 1.0 = Left, -1.0 = Right

// --- BLUSH PARAMS ---
uniform int uShowBlush;
uniform vec4 uBlushColor;    
uniform vec4 uTearColor;     

// UTILS
float hash(float n) { return fract(sin(n) * 43758.5453123); }

// Triangle SDF
float sdTriangle(vec2 p, float r) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x + k*p.y > 0.0 ) p = vec2(p.x - k*p.y, -k*p.x - p.y) / 2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p) * sign(p.y);
}

float sdTears(vec2 p, float level, float t) {
    float d = 1e5;
    
    // --- MODE 1: TRIANGLE TEARS ---
    if (level < 0.4) {

        //Need to add the teary eye effect here 
    }
    
    // --- MODE 2: CLASSIC DRIP ---
    else if (level < 0.75) {

         // [FIX] SCALE SIZE (35.0 -> Scaled)
        float size = 35.0 * uScale; 
        
        float speed = 0.2 + (level * 0.5);
        
        // [FIX] Buffer fall distance so they don't pop out
        float fallDist = uResolution.y + uScale; 
        
        for(float i = 0.0; i < 5.0; i++) {
            // [FIX] Spread them out (0.2 offset per drop)
            float offset = i / 5.0; 
            
            float cycle = fract((t * speed) + offset);
            
            // Position
            float yPos = (170.0 * uScale) + (cycle * fallDist);

            // [FIX] ADD X-JITTER (Scaled)
            // This prevents them from falling in a perfect single-file line
            float xPos = -(120.0 * uScale)*uSide ;

            vec2 dropPos = p - vec2(xPos, yPos);
            dropPos.y = -dropPos.y; 
            
            float dTri = sdTriangle(dropPos, size);
            d = min(d, dTri);
        
        }
    } 
    
    // --- MODE 3: WAIL STREAM ---
    else { 
        float width = (60.0 + (level * 20.0)) * uScale;
        
        // 1. THE WAVE (Sine Distortion)
        float wave = sin(p.y * 0.04 - t * 6.0) * (12.0 * uScale); 
        
        // 2. THE SHAPE (Vertical Box + Wave)
        float distToStream = abs(p.x + wave) - width;
        
        // 3. [FIX] CURVED TOP CUT
        // The eye is a frown shape (^).
        // That means the center (x=0) is higher than the sides.
        // In our Y+ Down system, that means Y is lower (more negative) at X=0.
        // Formula: y = Curvature * x^2
        float curvature = 0.004 / uScale; // Adjust curvature tightness
        float curveY = (p.x * p.x) * curvature;
        
        // The "Cut Plane" follows this curve.
        // We subtract the curve from p.y.
        // We also add an offset (-10.0) to tuck it slightly up behind the eye.
        float topEdge = -(p.y - curveY) - (10.0 * uScale);
        
        // Clip: Only draw where we are BELOW the curve
        d = max(distToStream, topEdge);
    }

    return d;
}

void main() {
    // 1. NORMALIZE
    vec2 p = (fragTexCoord - 0.5) * uResolution;
    
    // [CRITICAL FIX] SHIFT ORIGIN TO TOP
    // Since our C++ rect starts at the Eye Center, we move (0,0) to the Top Edge.
    // Now p.y = 0.0 IS the Eye Center. Positive Y is falling down.
    p.y += uResolution.y * 0.5;

    // (We removed the manual 'pTear' offset because the C++ rect handles position now)

    vec3 colorAccum = vec3(0.0);
    float alphaAccum = 0.0;

    // 2. BLUSH PASS
    if (uShowBlush == 1) {
        // Blush scales automatically because it uses 'r' (based on Rect size)
        float r = 0.4 * min(uResolution.x, uResolution.y);

        vec4 activeBlushColor = uBlushColor; 
        if(uBlushmode == 1)
        {
            activeBlushColor = vec4(0.47058, 0.65490, 0.45490,0.58823);
        }
        if(uBlushmode == 0)
        {
            activeBlushColor = vec4(1, 0.78431, 0.84705,1);
        }
        
        // Position blush slightly below eye (using the new coordinate system)
        float blushY = 70.0 * uScale; 
        float dBlush = length(p - vec2(0.0, blushY)) - r*0.6;
        
        float a = 1.0 - smoothstep(0.0, r, dBlush + r*0.2);
        a *= 0.5; 
        
        colorAccum = activeBlushColor.rgb;
        alphaAccum = a;
    }

    // 3. TEAR PASS
    if (uTearsLevel > 0.01) {
        float dTear = sdTears(p, uTearsLevel, uTime);
        
        // Color Selection
        vec3 activeTearColor = uTearColor.rgb; 
        if (uTearsLevel < 0.75 && uTearsLevel > 0.4 ) {
            activeTearColor = vec3(0.18039, 0.71373, 0.38431); // Green
        }
        if(uTearsLevel>0.75)
        {
            activeTearColor = vec3(0.40392, 0.83921, 0.63921);
        }

        // [FIX] OUTLINE COLOR
        // Make it 50% darker than the base color (matches reference style)
        vec3 outlineColor = vec3(0.58431, 0.95294, 0.77254);
        
        // --- 2. RENDER SHAPES ---
        float outlineWidth = 4.0 * uScale; // Scalable outline thickness

        // Alpha for the FILL (Inner)
        float fillAlpha = 1.0 - smoothstep(-0.5, 0.5, dTear);
        
        // Alpha for the OUTLINE (Outer Edge)
        // We check 'outlineWidth' pixels away from the edge
        float borderAlpha = 1.0 - smoothstep(outlineWidth - 0.5, outlineWidth + 0.5, dTear);
        
        // --- 3. COMPOSITE ---
        // Start with Outline Color
        vec3 finalRGB = outlineColor;
        
        // Blend Fill Color on top of Outline Color
        // (This creates the effect of a dark stroke around a light core)
        finalRGB = mix(finalRGB, activeTearColor, fillAlpha);
        
        // Add to accumulator
        colorAccum = mix(colorAccum, finalRGB, borderAlpha);
        alphaAccum = max(alphaAccum, borderAlpha);
    }

    if (alphaAccum < 0.01) discard;
    finalColor = vec4(colorAccum, alphaAccum);
}