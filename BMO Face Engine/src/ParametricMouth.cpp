// ParametricMouth_Platinum.cpp
// STATUS: FINAL PRODUCTION VERSION
// - ARCHITECTURE: Local Render Texture (Optimized)
// - TOPOLOGY: Diamond Indexing (Stable Teeth)
// - GEOMETRY: Radial Clamping (Perfect Tongue Fit)
// - SAFETY: Crash fixes for divide-by-zero and degenerates

#include "raylib.h"
#include "rlgl.h" 
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream> 

// ------------------------------
// Math Helpers
// ------------------------------
#include "raymath.h"

static inline Vector2 V2Add(Vector2 a, Vector2 b) { return {a.x + b.x, a.y + b.y}; }
static inline Vector2 V2Sub(Vector2 a, Vector2 b) { return {a.x - b.x, a.y - b.y}; }
static inline Vector2 V2Scale(Vector2 a, float s) { return {a.x * s, a.y * s}; }
static inline float   V2Cross(Vector2 a, Vector2 b) { return a.x*b.y - a.y*b.x; }

static inline int ClampInt(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static float SignedArea(const std::vector<Vector2>& p) {
    if (p.size() < 3) return 0.0f;
    double a = 0.0;
    for (size_t i=0; i<p.size(); i++){
        size_t j = (i+1) % p.size();
        a += (double)p[i].x*(double)p[j].y - (double)p[j].x*(double)p[i].y;
    }
    return (float)(0.5 * a);
}

static Vector2 CatmullRom(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float t2 = t * t; float t3 = t2 * t;
    float v0 = ((-t3) + (2 * t2) - t) * 0.5f;
    float v1 = ((3 * t3) - (5 * t2) + 2) * 0.5f;
    float v2 = ((-3 * t3) + (4 * t2) + t) * 0.5f;
    float v3 = (t3 - t2) * 0.5f;
    return { (p0.x * v0) + (p1.x * v1) + (p2.x * v2) + (p3.x * v3),
             (p0.y * v0) + (p1.y * v1) + (p2.y * v2) + (p3.y * v3) };
}

static void RemoveNearDuplicates(std::vector<Vector2>& poly, float eps=0.5f) {
    if (poly.size() < 2) return;
    std::vector<Vector2> out;
    out.reserve(poly.size());
    out.push_back(poly[0]);
    for (size_t i=1; i<poly.size(); i++){
        Vector2 a = poly[i];
        Vector2 b = out.back();
        float dx = a.x-b.x, dy=a.y-b.y;
        if (dx*dx + dy*dy > eps*eps) out.push_back(a);
    }
    if (out.size() > 2) {
        Vector2 first = out.front(); Vector2 last = out.back();
        float dx = first.x-last.x, dy = first.y-last.y;
        if (dx*dx + dy*dy <= eps*eps) out.pop_back();
    }
    poly.swap(out);
}
// [NEW] Removes points that form a straight line to stop Triangulator Panic
static void RemoveCollinearPoints(std::vector<Vector2>& poly, float threshold = 0.995f) {
    if (poly.size() < 3) return;
    std::vector<Vector2> out;
    out.push_back(poly[0]); // Keep first
    
    for (size_t i = 1; i < poly.size() - 1; i++) {
        Vector2 prev = out.back();
        Vector2 curr = poly[i];
        Vector2 next = poly[i+1];
        
        Vector2 v1 = V2Sub(curr, prev);
        Vector2 v2 = V2Sub(next, curr);
        
        // Normalize
        float l1 = sqrtf(v1.x*v1.x + v1.y*v1.y);
        float l2 = sqrtf(v2.x*v2.x + v2.y*v2.y);
        
        if (l1 < 0.1f || l2 < 0.1f) continue; // Skip tiny duplicates
        
        v1.x /= l1; v1.y /= l1;
        v2.x /= l2; v2.y /= l2;
        
        // Dot product: 1.0 means perfectly straight
        float dot = v1.x*v2.x + v1.y*v2.y;
        
        // If the angle is sharp enough, keep the point. If it's straight, skip it.
        if (dot < threshold) {
            out.push_back(curr);
        }
    }
    out.push_back(poly.back()); // Keep last
    poly = out;
}

// Helper: Is point P inside the polygon?
static bool IsPointInsidePoly(Vector2 p, const std::vector<Vector2>& poly) {
    if (poly.size() < 3) return false;
    // [FIX] Const-cast required for Raylib API compatibility
    return CheckCollisionPointPoly(p, const_cast<Vector2*>(poly.data()), (int)poly.size());
}

// Helper: Find the closest point on the polygon boundary to P
static Vector2 GetClosestPointOnPoly(Vector2 p, const std::vector<Vector2>& poly) {
    float minDistSq = 10000000.0f;
    Vector2 bestPoint = p;

    for (size_t i = 0; i < poly.size(); i++) {
        Vector2 a = poly[i];
        Vector2 b = poly[(i + 1) % poly.size()];

        Vector2 ap = V2Sub(p, a);
        Vector2 ab = V2Sub(b, a);
        
        // [FIX] Prevent divide-by-zero on degenerate edges
        float denom = (ab.x * ab.x + ab.y * ab.y);
        if (denom < 0.000001f) continue;

        float t = (ap.x * ab.x + ap.y * ab.y) / denom;
        t = (t < 0.0f) ? 0.0f : ((t > 1.0f) ? 1.0f : t);
        
        Vector2 closest = V2Add(a, V2Scale(ab, t));
        
        float dSq = (p.x - closest.x)*(p.x - closest.x) + (p.y - closest.y)*(p.y - closest.y);
        if (dSq < minDistSq) {
            minDistSq = dSq;
            bestPoint = closest;
        }
    }
    return bestPoint;
}

// Topology Normalization
static void RotateContourToLeftmost(std::vector<Vector2>& c) {
    if (c.empty()) return;
    int best = 0;
    for (int i = 1; i < (int)c.size(); i++) {
        if (c[i].x < c[best].x || (c[i].x == c[best].x && c[i].y < c[best].y))
            best = i;
    }
    std::rotate(c.begin(), c.begin() + best, c.end());
}

static void NormalizeContourForTeeth(std::vector<Vector2>& c) {
    if (c.size() < 4) return;
    RotateContourToLeftmost(c);
    int n = (int)c.size();
    if (c[n/4].y > c[(3*n)/4].y) {
        std::reverse(c.begin(), c.end());
        RotateContourToLeftmost(c); 
    }
}

struct SimpleTriangulator {
    static bool IsPointInTriangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c) {
        float cp1 = V2Cross(V2Sub(b, a), V2Sub(p, a));
        float cp2 = V2Cross(V2Sub(c, b), V2Sub(p, b));
        float cp3 = V2Cross(V2Sub(a, c), V2Sub(p, c));
        return (cp1 >= 0 && cp2 >= 0 && cp3 >= 0) || (cp1 <= 0 && cp2 <= 0 && cp3 <= 0);
    }

    static std::vector<Vector2> Triangulate(const std::vector<Vector2>& poly) {
        std::vector<Vector2> triangles;
        if (poly.size() < 3) return triangles;
        
        std::vector<int> indices(poly.size());
        for (int i = 0; i < (int)poly.size(); i++) indices[i] = i;
        
        float area = SignedArea(poly);
        float wantSign = (area >= 0.0f) ? 1.0f : -1.0f;

        int count = (int)poly.size(); 
        int safety = count * 4; 

        while (count > 2 && safety-- > 0) {
            bool earFound = false;
            for (int i = 0; i < count; i++) {
                int prev = (i - 1 + count) % count; 
                int curr = i; 
                int next = (i + 1) % count;
                
                Vector2 a = poly[indices[prev]]; 
                Vector2 b = poly[indices[curr]]; 
                Vector2 c = poly[indices[next]];
                
                float cross = V2Cross(V2Sub(b, a), V2Sub(c, b));
                if (cross * wantSign < 0) continue; 

                bool clean = true;
                for (int j = 0; j < count; j++) {
                    if (j == prev || j == curr || j == next) continue;
                    if (IsPointInTriangle(poly[indices[j]], a, b, c)) { clean = false; break; }
                }

                if (clean) {
                    triangles.push_back(a); triangles.push_back(b); triangles.push_back(c);
                    indices.erase(indices.begin() + curr); 
                    count--; 
                    earFound = true;
                    break;
                }
            }
            if (!earFound) break; 
        }
        return triangles;
    }
};

struct FacialParams {
    float open = 0.05f; 
    float width = 0.5f; 
    float curve = 0.0f; 
    float squeezeTop = 0.0f;    // 0.0 to 1.0
    float squeezeBottom = 0.0f; // 0.0 to 1.0 
    float teethY = 0.0f; 
    float tongueUp = 0.0f;      
    float tongueX = 0.0f;
    float tongueWidth = 0.65f;  

    // [NEW] Controls vertical asymmetry
    // 0.0 = Normal, 1.0 = Flat Top ("D"), -1.0 = Flat Bottom
    float asymmetry = 0.0f;
    // [NEW] Controls how "boxy" the shape is. 
    // 0.0 = Oval (Pointed tips), 1.0 = Rounded Rect (Blunt tips)
    float squareness = 0.0f;

    float teethWidth = 0.50f;   // 0.1 to 0.95 (Ratio of mouth width)
    float teethGap = 45.0f;     // 0.0 to 100.0 (Pixels in screen space)
    //Global Scale Multiplier
    float scale = 1.0f;
};

struct ParametricMouth {
    FacialParams current, target, velocity;
    std::vector<Vector2> controlPoints; 
    std::vector<Vector2> smoothContour; 
    std::vector<Vector2> topTeethPoly, botTeethPoly;
    std::vector<Vector2> tonguePoly;
    
    RenderTexture2D maskTexture;
    bool textureLoaded = false;
    
    Vector2 centerPos; 
    bool usePhysics = true;
    const float SS = 4.0f; 
    // CHANGE 1024 TO 2048
    const int RT_SIZE = 2048;

    Color colBg     = { 61, 93, 55, 255 };
    Color colLine   = { 20, 35, 20, 255 };
    Color colTeeth  = { 245, 245, 245, 255 };
    Color colTongue = { 152, 161, 101, 255 }; 

    void Init(Vector2 pos) {
        velocity = {}; 
        centerPos = pos;
        
        // [FIX] Reserve memory to prevent re-allocations
        controlPoints.resize(16);
        smoothContour.reserve(512);
        topTeethPoly.reserve(64);
        botTeethPoly.reserve(64);
        tonguePoly.reserve(64);
        
        maskTexture = LoadRenderTexture(RT_SIZE, RT_SIZE);
        SetTextureFilter(maskTexture.texture, TEXTURE_FILTER_BILINEAR);
        textureLoaded = true;
        
        target = { 0.05f, 0.5f, 0.2f,
           0.0f, 0.0f,
          -1.0f,
           0.0f, 0.0f, 0.65f,
           0.0f, 0.0f,
           0.5f, 45.0f,
           1.0f };
        current = target;
    }

    void Unload() { if(textureLoaded) UnloadRenderTexture(maskTexture); }

    void UpdatePhysics(float dt) {
        target.open        = Clamp(target.open, 0.0f, 1.2f);
        target.width       = Clamp(target.width, 0.1f, 1.5f);
        target.curve       = Clamp(target.curve, -1.0f, 1.0f);
        target.squeezeTop  = Clamp(target.squeezeTop, 0.0f, 1.0f);
        target.squeezeBottom = Clamp(target.squeezeBottom, 0.0f, 1.0f);
        target.teethY      = Clamp(target.teethY, -1.0f, 1.0f);
        target.tongueUp    = Clamp(target.tongueUp, 0.0f, 1.0f);
        target.tongueX     = Clamp(target.tongueX, -1.0f, 1.0f);
        target.tongueWidth = Clamp(target.tongueWidth, 0.3f, 1.0f);
        target.asymmetry   = Clamp(target.asymmetry, -1.0f, 1.0f);
        target.squareness  = Clamp(target.squareness, 0.0f, 1.0f);
        target.teethWidth  = Clamp(target.teethWidth, 0.1f, 0.95f);
        target.teethGap    = Clamp(target.teethGap, 0.0f, 100.0f);
        target.scale      = Clamp(target.scale, 0.5f, 5.0f); // Allows 0.5x to 5x scaling

        if (!usePhysics) { 
            current = target; 
            return; 
        }
        if (dt > 0.05f) dt = 0.05f; 

        const float STIFFNESS = 180.0f;
        const float DAMPING   = 14.0f;    
        
        auto Upd = [&](float& c, float& v, float t) {
            float f = STIFFNESS * (t - c);
            float d = DAMPING * v;
            v += (f - d) * dt; 
            c += v * dt;
        };

        Upd(current.open, velocity.open, target.open);
        Upd(current.width, velocity.width, target.width);
        Upd(current.curve, velocity.curve, target.curve);
        Upd(current.squeezeTop, velocity.squeezeTop, target.squeezeTop);
        Upd(current.squeezeBottom, velocity.squeezeBottom, target.squeezeBottom);
        Upd(current.teethY, velocity.teethY, target.teethY);
        Upd(current.tongueUp, velocity.tongueUp, target.tongueUp);
        Upd(current.tongueX, velocity.tongueX, target.tongueX);
        Upd(current.tongueWidth, velocity.tongueWidth, target.tongueWidth);
        Upd(current.asymmetry, velocity.asymmetry, target.asymmetry);
        Upd(current.squareness, velocity.squareness, target.squareness);
        Upd(current.teethWidth, velocity.teethWidth, target.teethWidth);
        Upd(current.teethGap,   velocity.teethGap,   target.teethGap);
        Upd(current.scale,     velocity.scale,     target.scale);

        current.open        = Clamp(current.open, 0.0f, 1.2f);
        current.width       = Clamp(current.width, 0.1f, 1.5f);
        current.curve       = Clamp(current.curve, -1.0f, 1.0f);
        current.squeezeTop = Clamp(current.squeezeTop, 0.0f, 1.0f);
        current.squeezeBottom = Clamp(current.squeezeBottom, 0.0f, 1.0f);
        current.teethY      = Clamp(current.teethY, -1.0f, 1.0f);
        current.tongueUp    = Clamp(current.tongueUp, 0.0f, 1.0f);
        current.tongueX     = Clamp(current.tongueX, -1.0f, 1.0f);
        current.tongueWidth = Clamp(current.tongueWidth, 0.3f, 1.0f);
        current.asymmetry   = Clamp(current.asymmetry, -1.0f, 1.0f);
        current.squareness  = Clamp(current.squareness, 0.0f, 1.0f);
        current.teethWidth = Clamp(current.teethWidth, 0.1f, 0.95f);
        current.teethGap   = Clamp(current.teethGap, 0.0f, 100.0f);
        current.scale      = Clamp(current.scale, 0.5f, 5.0f);
    }

    // Tongue with Radial Clamping
    void GenerateTongue(float minY, float maxY, float cx, float cy, float mouthW) {
        tonguePoly.clear();
        // [FIX] Removed check for tongueUp <= 0.01 so it shows at rest
        if (current.open < 0.10f) return;

        float shiftX = current.tongueX * (mouthW * 0.4f);
        float tongueCX = cx + shiftX;

        float margin = 4.0f * SS;
        float tongueBottom = maxY - margin;
        float tongueTip = Lerp(tongueBottom, minY + margin, current.tongueUp);
        float halfW = Clamp(mouthW * 0.55f * current.tongueWidth, 6.0f * SS, mouthW * 0.95f);

        std::vector<Vector2> rawPoly;
        rawPoly.reserve(32); // Optimization

        const int seg = 16;
        for (int i = 0; i <= seg; i++) {
            float t = (float)i / (float)seg;     
            float xRatio = -cosf(t * PI);        
            float yRatio = sinf(t * PI);         

            float tipTaper = Lerp(1.0f, 0.85f, yRatio); 
            float px = tongueCX + (xRatio * halfW * tipTaper);
            float py = Lerp(tongueBottom, tongueTip, yRatio);
            rawPoly.push_back({ px, py });
        }
        rawPoly.push_back({ tongueCX + halfW, tongueBottom + margin });
        rawPoly.push_back({ tongueCX - halfW, tongueBottom + margin });

        // THE CLAMP PASS
        for (Vector2 p : rawPoly) {
            if (IsPointInsidePoly(p, smoothContour)) {
                tonguePoly.push_back(p);
            } else {
                // Snap to contour if leaking
                Vector2 snapped = GetClosestPointOnPoly(p, smoothContour);
                Vector2 dirToCenter = V2Sub({tongueCX, cy}, snapped);
                float dist = sqrtf(dirToCenter.x*dirToCenter.x + dirToCenter.y*dirToCenter.y);
                if (dist > 0.001f) {
                    Vector2 pull = V2Scale(dirToCenter, (1.0f / dist)); 
                    tonguePoly.push_back(V2Add(snapped, pull));
                } else {
                    tonguePoly.push_back(snapped);
                }
            }
        }
        
        RemoveNearDuplicates(tonguePoly);
        
        // [FIX] Degeneracy check
        if (tonguePoly.size() < 3) tonguePoly.clear();
    }

    // [UPDATED] Now accepts limitLeft/limitRight to force perfect alignment
    void BuildTeethToLine(const std::vector<Vector2>& spline, int start, int endExclusive,
                          float targetY, float dirY, 
                          float limitLeft, float limitRight, 
                          std::vector<Vector2>& outPoly)
    {
        outPoly.clear();
        int n = (int)spline.size();
        
        start = ClampInt(start + 1, 0, n-1);
        endExclusive = ClampInt(endExclusive - 1, 0, n);
        
        if (start >= endExclusive) return;

        // [FIX] DETECT DIRECTION
        // Check if this segment runs Left->Right or Right->Left
        bool isReversed = (spline[start].x > spline[endExclusive - 1].x);

        // Assign limits based on direction
        // If normal (L->R): Start=Left, End=Right
        // If reversed (R->L): Start=Right, End=Left
        float startLimit = isReversed ? limitRight : limitLeft;
        float endLimit   = isReversed ? limitLeft  : limitRight;

        float rangeX = limitRight - limitLeft;
        if (fabsf(rangeX) < 1.0f) rangeX = (rangeX > 0) ? 1.0f : -1.0f;

        // --- PHASE 1: THE GUM LINE (From Spline) ---
        for (int i = start; i < endExclusive; i++) {
            Vector2 p = spline[i];
            
            // Apply direction-aware limits
            if (i == start) p.x = startLimit;
            if (i == endExclusive - 1) p.x = endLimit;
            
            p.x = Clamp(p.x, limitLeft, limitRight);
            outPoly.push_back(p);
        }

        // --- PHASE 2: THE BITING EDGE (Generated) ---
        for (int i = endExclusive - 1; i >= start; i--) {
            Vector2 p = spline[i];
            
            // Apply direction-aware limits
            if (i == start) p.x = startLimit;
            if (i == endExclusive - 1) p.x = endLimit;
            p.x = Clamp(p.x, limitLeft, limitRight);

            // Calculate T (Position Percentage)
            // T should always be 0.0 at limitLeft and 1.0 at limitRight
            float t = (p.x - limitLeft) / rangeX;
            t = Clamp(t, 0.0f, 1.0f); 

            float taperSharpness = Lerp(0.2f, 0.05f, current.squareness);
            float s = std::sin(t * PI);
            s = Clamp(s, 0.0f, 1.0f);
            float taper = std::pow(s, taperSharpness); 

            float distToLine = (targetY - p.y) * dirY; 
            float h = std::max(0.0f, distToLine);
            float finalY = p.y + (dirY * h * taper);

            outPoly.push_back({p.x, finalY});
        }
        
        RemoveNearDuplicates(outPoly, 2.0f); 
        RemoveCollinearPoints(outPoly);
    }
    // Helper: Find start/end indices that fit within horizontal bounds [minX, maxX]
    static void GetIndicesByX(const std::vector<Vector2>& poly, int rangeStart, int rangeEnd, 
                              float minX, float maxX, int& outStart, int& outEnd) 
    {
        outStart = -1;
        outEnd = -1;

        // Scan the specified range of the polygon
        for (int i = rangeStart; i < rangeEnd; i++) {
            float x = poly[i].x;

            // Check if this point is inside the dental zone
            if (x >= minX && x <= maxX) {
                if (outStart == -1) outStart = i; // First valid point
                outEnd = i;                       // Update last valid point
            }
        }

        // Safety: If nothing found, collapse to rangeStart
        if (outStart == -1) {
            outStart = rangeStart;
            outEnd = rangeStart;
        } else {
            // [IMPORTANT] Buffer: Add one extra point on each side to ensure 
            // the curve reaches all the way to the edge of the clip zone.
            outStart = std::max(rangeStart, outStart - 1);
            outEnd   = std::min(rangeEnd,   outEnd + 2); // +2 because standard loop is < end
        }
    }
    // [NEW] Safety Clamp: Forces any polygon to stay strictly inside the mouth
    // Used to prevent teeth corners from poking out of rounded mouths
    static void ClampPolyToContainer(std::vector<Vector2>& subject, const std::vector<Vector2>& container, Vector2 center) {
        if (subject.empty() || container.size() < 3) return;

        std::vector<Vector2> safePoly;
        safePoly.reserve(subject.size());

        for (Vector2 p : subject) {
            // 1. If point is already inside, keep it.
            if (IsPointInsidePoly(p, container)) {
                safePoly.push_back(p);
            } 
            else {
                // 2. If outside, snap it to the nearest edge of the container
                Vector2 snapped = GetClosestPointOnPoly(p, container);

                // 3. Optional: Pull it inward by a tiny fraction (0.1 pixel) 
                // to ensure it doesn't fail floating point checks later
                Vector2 dir = V2Sub(center, snapped);
                float dist = sqrtf(dir.x*dir.x + dir.y*dir.y);
                if (dist > 0.001f) {
                    // Pull in by 0.5 unit relative to scale to be safe
                    float pullFactor = 0.5f / dist; 
                    snapped = V2Add(snapped, V2Scale(dir, pullFactor));
                }
                safePoly.push_back(snapped);
            }
        }
        // Update the original polygon
        subject = safePoly;
    }

    void GenerateGeometry() {
        float baseRadius = 40.0f * SS; 
        float w = baseRadius * (0.5f + current.width);
        float h = (current.open < 0.08f) ? 0.0f : (baseRadius * (0.2f + current.open * 1.5f));
        
        float cx = RT_SIZE * 0.5f; 
        float cy = RT_SIZE * 0.5f;

        for (int i = 0; i < 16; i++) {
            float t = (float)i / 16.0f;
            float angle = t * PI * 2.0f + PI; 
            float x = cosf(angle) * w;
            // [NEW] Squareness Logic (Superellipse)
            // Instead of y = sin(angle)*h, we use power function to inflate corners
            float rawSin = sinf(angle);
            float sign = (rawSin >= 0.0f) ? 1.0f : -1.0f;
            
            // Map squareness 0..1 to exponent 1.0..0.2
            // Exponent < 1.0 makes the wave "fatter" (approaching a square wave)
            float power = 1.0f - (current.squareness * 0.8f); 
            float shapedSin = std::pow(std::abs(rawSin), power) * sign;
            
            float y = shapedSin * h;
            
            float bendFactor = 15.0f * SS; 
            float normalizedX = x / w;
            
            // Standard bend calculation
            float rawBend = (normalizedX * normalizedX) * bendFactor * current.curve;

            // [NEW] ASYMMETRY LOGIC
            // Indices 0-8 are the Top Arch. Indices 8-16 are the Bottom Arch.
            bool isTop = (i <= 8);
            
            float bendMult = 1.0f;
            if (current.asymmetry > 0.05f) {
                // Positive asymmetry -> Flatten Top (Reduce bend on top)
                if (isTop) bendMult = 1.0f - current.asymmetry;
            } 
            else if (current.asymmetry < -0.05f) {
                // Negative asymmetry -> Flatten Bottom (Reduce bend on bottom)
                if (!isTop) bendMult = 1.0f - fabsf(current.asymmetry);
            }

            y -= rawBend * bendMult; 

            // [NEW] INDEPENDENT SQUEEZE LOGIC
            float activeSqueeze = isTop ? current.squeezeTop : current.squeezeBottom;

            // Calculate normalized distance from center (0.0 = center, 1.0 = corner)
            float tX = std::abs(x) / w;
            
            // COSINE FALLOFF: Creates a smooth "Hill" shape
            // 1.0 at center, smoothly curving down to 0.0 at corners.
            // This replaces the hard "if (x < 0.6)" check.
            float influence = std::max(0.0f, cosf(tX * PI * 0.5f));

            // Apply deformation
            // We use 'influence' to blend the effect.
            // Power of 2 makes the pinch slightly sharper at the very center (optional)
            // y *= Lerp(1.0f, 1.0f - activeSqueeze, influence); 
            
            // Simple Multiplier version (Production Safe):
            y *= (1.0f - (activeSqueeze * influence * 0.8f));
            
            controlPoints[i] = { cx + x, cy + y };
        }

        if (current.open < 0.08f) {
            smoothContour.clear(); 
            topTeethPoly.clear();
            botTeethPoly.clear();
            tonguePoly.clear();
            return;
        }

        smoothContour.clear();
        for (int i = 0; i < 16; i++) {
            Vector2 p0 = controlPoints[(i-1+16)%16];
            Vector2 p1 = controlPoints[i];
            Vector2 p2 = controlPoints[(i+1)%16];
            Vector2 p3 = controlPoints[(i+2)%16];
            const int subDivs = 16;
            
            for (int k = 0; k < subDivs; k++) {
                float tt = (float)k / (float)subDivs;
                smoothContour.push_back(CatmullRom(p0, p1, p2, p3, tt));
        }
        RemoveNearDuplicates(smoothContour);
        NormalizeContourForTeeth(smoothContour);

        // --- INTERIOR LOGIC ---
        topTeethPoly.clear(); botTeethPoly.clear(); tonguePoly.clear();
        
        if (current.open > 0.10f && !smoothContour.empty()) {
            float minY = smoothContour[0].y;
            float maxY = smoothContour[0].y;
            for (const auto& p : smoothContour) {
                if (p.y < minY) minY = p.y;
                if (p.y > maxY) maxY = p.y;
            }
            
            GenerateTongue(minY, maxY, cx, cy, w);

            if ((maxY - minY) < 5.0f * SS) return; 

            // Teeth Generation
            // [NEW] 1. Calculate the PHYSICAL boundaries of the teeth
            float mouthMinX = 10000.0f;
            float mouthMaxX = -10000.0f;
            for(const auto& p : smoothContour) {
                if(p.x < mouthMinX) mouthMinX = p.x;
                if(p.x > mouthMaxX) mouthMaxX = p.x;
            }
            
            float mouthRealWidth = mouthMaxX - mouthMinX;
            float teethCX = (mouthMinX + mouthMaxX) * 0.5f;
            
            // Calculate the exact pixel X where teeth should start and end
            float teethHalfW = (mouthRealWidth * current.teethWidth) * 0.5f;
            float targetLeftX  = teethCX - teethHalfW;
            float targetRightX = teethCX + teethHalfW;

            // [NEW] 2. Find indices that match these physical coordinates
            int n = (int)smoothContour.size();
            int half = n / 2;
            
            int tStart, tEnd, bStart, bEnd;
            
            // Scan Top Arch (0 to Half)
            GetIndicesByX(smoothContour, 0, half, targetLeftX, targetRightX, tStart, tEnd);
            
            // Scan Bottom Arch (Half to End)
            GetIndicesByX(smoothContour, half, n, targetLeftX, targetRightX, bStart, bEnd);

            float shiftY = current.teethY * 20.0f * SS;
            float gap = current.teethGap * SS;
            
            float topTarget = cy + shiftY - (gap * 0.5f);
            float botTarget = cy + shiftY + (gap * 0.5f);

            float margin = 2.0f * SS;
            topTarget = Clamp(topTarget, minY + margin, maxY - margin);
            botTarget = Clamp(botTarget, minY + margin, maxY - margin);
            
            // [Alignment Fix] Recalculate center based on clamps
            float mid = (topTarget + botTarget) * 0.5f;
            float halfGap = gap * 0.5f;
            topTarget = mid - halfGap;
            botTarget = mid + halfGap;
            topTarget = Clamp(topTarget, minY + margin, maxY - margin);
            botTarget = Clamp(botTarget, minY + margin, maxY - margin);

            // Build the initial geometry (which might poke out)
            if (topTarget < botTarget - 1.0f) {
                // [FIX] Pass targetLeftX and targetRightX here to satisfy the 8-argument requirement
                BuildTeethToLine(smoothContour, tStart, tEnd, topTarget, 1.0f, 
                                 targetLeftX, targetRightX, 
                                 topTeethPoly);
                                 
                BuildTeethToLine(smoothContour, bStart, bEnd, botTarget, -1.0f, 
                                 targetLeftX, targetRightX, 
                                 botTeethPoly);
                
                // [FIX] THE SAFETY PASS
                // Force the newly created teeth to strictly respect the mouth boundaries.
                // This crunches the sharp corners of the teeth so they fit inside the rounded lips.
                ClampPolyToContainer(topTeethPoly, smoothContour, {teethCX, cy});
                ClampPolyToContainer(botTeethPoly, smoothContour, {teethCX, cy});
                
                // Clean up any micro-segments created by the clamping
                RemoveNearDuplicates(topTeethPoly, 1.0f);
                RemoveNearDuplicates(botTeethPoly, 1.0f);
            }
        }
    }

    void DrawPoly(const std::vector<Vector2>& poly, Color c) {
        if(poly.size() < 3) return;
        std::vector<Vector2> tris = SimpleTriangulator::Triangulate(poly);
        if (tris.size() < 3) return;

        rlBegin(RL_TRIANGLES);
        rlColor4ub(c.r, c.g, c.b, c.a);
        for(auto& p : tris) rlVertex2f(p.x, p.y);
        rlEnd();
    }
    // Draws a strip of lines with round joints (No gaps, no jagged corners)
    void DrawLineStrip(const std::vector<Vector2>& points, float thick, Color c) {
        if(points.size() < 2) return;
        for(size_t i=0; i<points.size() - 1; i++) {
            Vector2 p1 = points[i];
            Vector2 p2 = points[i+1];
            
            DrawLineEx(p1, p2, thick, c);
            DrawCircleV(p1, thick * 0.5f, c); // The Joint Fix
        }
        // Cap the very end
        DrawCircleV(points.back(), thick * 0.5f, c);
    }
    
    void DrawPolyOutline(const std::vector<Vector2>& poly, float thick, Color c) {
           if(poly.size() < 2) return;
           
           // Use the default circle resolution (smooth enough)
           // Draw segment AND a round cap at the start
           for(size_t i=0; i<poly.size(); i++) {
               Vector2 p1 = poly[i];
               Vector2 p2 = poly[(i+1)%poly.size()];
               
               // 1. Draw the segment
               DrawLineEx(p1, p2, thick, c);
               
               // 2. Draw the Round Join (The Fix)
               // This covers the jagged corners where segments meet
               DrawCircleV(p1, thick * 0.5f, c);
           }
    }

    void Draw() {
        rlDisableBackfaceCulling();
        BeginTextureMode(maskTexture);
        rlSetBlendMode(BLEND_ALPHA);
        ClearBackground(BLANK);

        if (current.open < 0.08f) {
            // 1. Generate a High-Res Spline for the "Smile Line" (Top Arch: Indices 0-8)
            // We use the same math as the open mouth so it matches quality perfectly.
            std::vector<Vector2> closedLine;
            closedLine.reserve(128); 
            
            for (int i = 0; i < 8; i++) { // Segments 0->1 ... 7->8
                Vector2 p0 = controlPoints[(i - 1 + 16) % 16];
                Vector2 p1 = controlPoints[i];
                Vector2 p2 = controlPoints[(i + 1) % 16];
                Vector2 p3 = controlPoints[(i + 2) % 16];
                
                const int subDivs = 16; // Match your open mouth quality
                float tt = 0.0f;
                for (int k = 0; k < subDivs; k++) {
                    tt = (float)k / (float)subDivs;
                    closedLine.push_back(CatmullRom(p0, p1, p2, p3, tt));
                }
            }
            // Add the final corner point
            closedLine.push_back(controlPoints[8]);
            
            // 2. Draw with Round Joins
            DrawLineStrip(closedLine, 6.0f * SS, colLine);
        }
        else if (smoothContour.size() >= 3) {
            DrawPoly(smoothContour, WHITE);
            
            rlDrawRenderBatchActive();
            rlSetBlendFactors(RL_DST_ALPHA, RL_ONE_MINUS_DST_ALPHA, RL_FUNC_ADD);

            DrawPoly(smoothContour, colBg);
            if (!tonguePoly.empty()) DrawPoly(tonguePoly, colTongue);
            if (!topTeethPoly.empty()) DrawPoly(topTeethPoly, colTeeth);
            if (!botTeethPoly.empty()) DrawPoly(botTeethPoly, colTeeth);
            
            float thick = 3.0f * SS; 
            if (!topTeethPoly.empty()) DrawPolyOutline(topTeethPoly, thick, colLine);
            if (!botTeethPoly.empty()) DrawPolyOutline(botTeethPoly, thick, colLine);

            rlDrawRenderBatchActive();
            rlSetBlendMode(BLEND_ALPHA);

            rlEnableSmoothLines();
            DrawPolyOutline(smoothContour, 6.0f * SS, colLine);
            rlDisableSmoothLines();
        }

        EndTextureMode();
        rlEnableBackfaceCulling();

        Rectangle src = {0, 0, (float)RT_SIZE, -(float)RT_SIZE};
        float finalScale =  (RT_SIZE / SS) * current.scale;
        Rectangle dst = { centerPos.x, centerPos.y, finalScale, finalScale };
        Vector2 origin = { dst.width * 0.5f, dst.height * 0.5f };
        
        DrawTexturePro(maskTexture.texture, src, dst, origin, 0.0f, WHITE);
    }
};

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 720, "BMO Rig - GOLDEN MASTER + TONGUE");
    SetTargetFPS(60);

    ParametricMouth mouth;
    mouth.Init({640, 360});
    
    float logTimer = 0.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        
        if (IsKeyDown(KEY_Q)) mouth.target.open += 2.0f * dt;
        if (IsKeyDown(KEY_A)) mouth.target.open -= 2.0f * dt;
        if (IsKeyDown(KEY_W)) mouth.target.width += 1.0f * dt;
        if (IsKeyDown(KEY_S)) mouth.target.width -= 1.0f * dt;
        if (IsKeyDown(KEY_E)) mouth.target.curve += 2.0f * dt; 
        if (IsKeyDown(KEY_D)) mouth.target.curve -= 2.0f * dt; 
        if (IsKeyDown(KEY_T)) mouth.target.teethY += 2.0f * dt;
        if (IsKeyDown(KEY_G)) mouth.target.teethY -= 2.0f * dt;
        if (IsKeyDown(KEY_R)) mouth.target.squeeze += 2.0f * dt;
        if (IsKeyDown(KEY_F)) mouth.target.squeeze -= 2.0f * dt;
        if (IsKeyPressed(KEY_SPACE)) mouth.usePhysics = !mouth.usePhysics;

        if (IsKeyDown(KEY_Y)) mouth.target.tongueUp += 2.0f * dt;
        if (IsKeyDown(KEY_H)) mouth.target.tongueUp -= 2.0f * dt;
        if (IsKeyDown(KEY_U)) mouth.target.tongueWidth += 1.0f * dt;
        if (IsKeyDown(KEY_J)) mouth.target.tongueWidth -= 1.0f * dt;

        if (IsKeyDown(KEY_Z)) mouth.debugTeethWidthRatio -= 0.5f * dt;
        if (IsKeyDown(KEY_X)) mouth.debugTeethWidthRatio += 0.5f * dt;
        if (IsKeyDown(KEY_C)) mouth.debugTeethGap -= 20.0f * dt;
        if (IsKeyDown(KEY_V)) mouth.debugTeethGap += 20.0f * dt;

        mouth.debugTeethWidthRatio = Clamp(mouth.debugTeethWidthRatio, 0.1f, 0.95f);
        mouth.debugTeethGap = Clamp(mouth.debugTeethGap, 0.0f, 100.0f);

        if (IsKeyPressed(KEY_ONE))   mouth.target = { 0.05f, 1.17f, 1.0f, 0.0f, -1.0f, 0.0f, 0.65f }; 
        if (IsKeyPressed(KEY_TWO))   mouth.target = { 1.0f, 0.6f, 0.8f, 0.0f, 0.2f, 0.3f, 0.7f }; 
        if (IsKeyPressed(KEY_THREE)) mouth.target = { 0.6f, 0.5f, 0.2f, 0.0f, -1.0f, 0.85f, 0.6f }; 
        if (IsKeyPressed(KEY_FOUR))  mouth.target = { 0.5f, 0.8f, -0.2f, 0.0f, 0.0f, 0.1f, 0.8f }; 

        mouth.UpdatePhysics(dt);
        mouth.GenerateGeometry();
        
        BeginDrawing();
        ClearBackground({201, 228, 195, 255}); 
        mouth.Draw();
        
        DrawText("Controls: Q/A(Open) W/S(Width) E/D(Curve) R/F(Squeeze) T/G(TeethY)", 20, 20, 20, DARKGRAY);
        DrawText("Tongue: Y/H(Up) U/J(Width) | Teeth: Z/X(Width) C/V(Gap)", 20, 50, 20, DARKGRAY);
        DrawText("Size: UP/DOWN | Presets: 1-4", 20, 80, 20, DARKGRAY);
        
        // logTimer += dt;
        // if (logTimer > 0.2f) {
        //      printf("Tgt: O:%.2f W:%.2f C:%.2f TUp:%.2f TW:%.2f\n", 
        //        mouth.target.open, mouth.target.width, mouth.target.curve, 
        //        mouth.target.tongueUp, mouth.target.tongueWidth);
        //      logTimer = 0.0f;
        // }

        EndDrawing();
    }
    mouth.Unload();
    CloseWindow();
    return 0;
}