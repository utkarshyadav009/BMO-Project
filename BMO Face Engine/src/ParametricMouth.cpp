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
    float tongueWidth = 0.65f;  

    // [NEW] Controls vertical asymmetry
    // 0.0 = Normal, 1.0 = Flat Top ("D"), -1.0 = Flat Bottom
    float asymmetry = 0.0f;
    // [NEW] Controls how "boxy" the shape is. 
    // 0.0 = Oval (Pointed tips), 1.0 = Rounded Rect (Blunt tips)
    float squareness = 0.0f;

    float teethWidth = 0.50f;   // 0.1 to 0.95 (Ratio of mouth width)
    float teethGap = 45.0f;     // 0.0 to 100.0 (Pixels in screen space)
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
    float outputScale = 1.0f; 
    bool usePhysics = true;
    const float SS = 4.0f; 
    // CHANGE 1024 TO 2048
    const int RT_SIZE = 4096;

    Color colBg     = { 61, 93, 55, 255 };
    Color colLine   = { 20, 35, 20, 255 };
    Color colTeeth  = { 245, 245, 245, 255 };
    Color colTongue = { 152, 161, 101, 255 }; 

    void Init(Vector2 pos) {
        velocity = {}; 
        centerPos = pos;
        
        // [FIX] Reserve memory to prevent re-allocations
        controlPoints.resize(16);
        smoothContour.reserve(128);
        topTeethPoly.reserve(64);
        botTeethPoly.reserve(64);
        tonguePoly.reserve(64);
        
        maskTexture = LoadRenderTexture(RT_SIZE, RT_SIZE);
        SetTextureFilter(maskTexture.texture, TEXTURE_FILTER_BILINEAR);
        textureLoaded = true;
        
        target = { 0.05f, 0.5f, 0.2f, 0.0f, -1.0f, 0.0f, 0.65f }; 
        current = target;
    }

    void Unload() { if(textureLoaded) UnloadRenderTexture(maskTexture); }

    void UpdatePhysics(float dt) {
        target.open        = Clamp(target.open, 0.0f, 1.2f);
        target.width       = Clamp(target.width, 0.1f, 1.5f);
        target.curve       = Clamp(target.curve, -1.0f, 1.0f);
        target.squeezeTop = Clamp(target.squeezeTop, 0.0f, 1.0f);
        target.squeezeBottom = Clamp(target.squeezeBottom, 0.0f, 1.0f);
        target.teethY      = Clamp(target.teethY, -1.0f, 1.0f);
        target.tongueUp    = Clamp(target.tongueUp, 0.0f, 1.0f);
        target.tongueWidth = Clamp(target.tongueWidth, 0.3f, 1.0f);
        target.asymmetry   = Clamp(target.asymmetry, -1.0f, 1.0f);
        target.squareness  = Clamp(target.squareness, 0.0f, 1.0f);
        target.teethWidth = Clamp(target.teethWidth, 0.1f, 0.95f);
        target.teethGap   = Clamp(target.teethGap, 0.0f, 100.0f);

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
        Upd(current.tongueWidth, velocity.tongueWidth, target.tongueWidth);
        Upd(current.asymmetry, velocity.asymmetry, target.asymmetry);
        Upd(current.squareness, velocity.squareness, target.squareness);
        Upd(current.teethWidth, velocity.teethWidth, target.teethWidth);
        Upd(current.teethGap,   velocity.teethGap,   target.teethGap);

        current.open        = Clamp(current.open, 0.0f, 1.2f);
        current.width       = Clamp(current.width, 0.1f, 1.5f);
        current.curve       = Clamp(current.curve, -1.0f, 1.0f);
        current.squeezeTop = Clamp(current.squeezeTop, 0.0f, 1.0f);
        current.squeezeBottom = Clamp(current.squeezeBottom, 0.0f, 1.0f);
        current.teethY      = Clamp(current.teethY, -1.0f, 1.0f);
        current.tongueUp    = Clamp(current.tongueUp, 0.0f, 1.0f);
        current.tongueWidth = Clamp(current.tongueWidth, 0.3f, 1.0f);
        current.asymmetry   = Clamp(current.asymmetry, -1.0f, 1.0f);
        current.squareness  = Clamp(current.squareness, 0.0f, 1.0f);
        current.teethWidth = Clamp(current.teethWidth, 0.1f, 0.95f);
        current.teethGap   = Clamp(current.teethGap, 0.0f, 100.0f);
    }

    // Tongue with Radial Clamping
    void GenerateTongue(float minY, float maxY, float cx, float cy, float mouthW) {
        tonguePoly.clear();
        // [FIX] Removed check for tongueUp <= 0.01 so it shows at rest
        if (current.open < 0.10f) return;

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
            float px = cx + (xRatio * halfW * tipTaper);
            float py = Lerp(tongueBottom, tongueTip, yRatio);
            rawPoly.push_back({ px, py });
        }
        rawPoly.push_back({ cx + halfW, tongueBottom + margin });
        rawPoly.push_back({ cx - halfW, tongueBottom + margin });

        // THE CLAMP PASS
        for (Vector2 p : rawPoly) {
            if (IsPointInsidePoly(p, smoothContour)) {
                tonguePoly.push_back(p);
            } else {
                // Snap to contour if leaking
                Vector2 snapped = GetClosestPointOnPoly(p, smoothContour);
                Vector2 dirToCenter = V2Sub({cx, cy}, snapped);
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

    void BuildTeethToLine(const std::vector<Vector2>& spline, int start, int endExclusive,
                          float targetY, float dirY, std::vector<Vector2>& outPoly)
    {
        outPoly.clear();
        int n = (int)spline.size();
        
        start = ClampInt(start + 1, 0, n-1);
        endExclusive = ClampInt(endExclusive - 1, 0, n);
        
        if (start >= endExclusive) return;
        int span = endExclusive - start;
        if (span < 2) return;

        for (int i = start; i < endExclusive; i++) outPoly.push_back(spline[i]);

        for (int i = endExclusive - 1; i >= start; i--) {
            Vector2 p = spline[i];
            float t = (float)(i - start) / (float)(span - 1);
            
            float s = std::sin(t * PI);
            s = Clamp(s, 0.0f, 1.0f);
            float taper = std::pow(s, 0.2f); 

            float distToLine = (targetY - p.y) * dirY; 
            float h = std::max(0.0f, distToLine);
            float finalY = p.y + (dirY * h * taper);

            outPoly.push_back({p.x, finalY});
        }
        
        RemoveNearDuplicates(outPoly);
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
            for (int k = 0; k < 6; k++) smoothContour.push_back(CatmullRom(p0, p1, p2, p3, k / 6.0f));
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
            int n = (int)smoothContour.size();
            int half = n / 2;
            float span = (1.0f - current.teethWidth) / 2.0f;
            int tStart = (int)(half * span);
            int tEnd   = (int)(half * (1.0f - span));
            int bStart = half + (int)((n - half) * span);
            int bEnd   = half + (int)((n - half) * (1.0f - span));

            float shiftY = current.teethY * 20.0f * SS;
            float gap = current.teethGap * SS;
            
            float topTarget = cy + shiftY - (gap * 0.5f);
            float botTarget = cy + shiftY + (gap * 0.5f);

            float margin = 2.0f * SS;
            topTarget = Clamp(topTarget, minY + margin, maxY - margin);
            botTarget = Clamp(botTarget, minY + margin, maxY - margin);
            
            float mid = (topTarget + botTarget) * 0.5f;
            float halfGap = gap * 0.5f;
            
            topTarget = mid - halfGap;
            botTarget = mid + halfGap;
            
            topTarget = Clamp(topTarget, minY + margin, maxY - margin);
            botTarget = Clamp(botTarget, minY + margin, maxY - margin);

            if (topTarget < botTarget - 1.0f) {
                BuildTeethToLine(smoothContour, tStart, tEnd, topTarget, 1.0f, topTeethPoly);
                BuildTeethToLine(smoothContour, bStart, bEnd, botTarget, -1.0f, botTeethPoly);
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
    
    void DrawPolyOutline(const std::vector<Vector2>& poly, float thick, Color c) {
        if(poly.size() < 2) return;
        for(size_t i=0; i<poly.size(); i++) 
            DrawLineEx(poly[i], poly[(i+1)%poly.size()], thick, c);
    }

    void Draw() {
        rlDisableBackfaceCulling();
        BeginTextureMode(maskTexture);
        rlSetBlendMode(BLEND_ALPHA);
        ClearBackground(BLANK);

        if (current.open < 0.08f) {
            rlEnableSmoothLines();
            for (int i = 0; i < 8; i++) 
                DrawLineEx(controlPoints[i], controlPoints[i+1], 6.0f * SS, colLine);
            rlDisableSmoothLines();
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
        Rectangle dst = { centerPos.x, centerPos.y, (float)(RT_SIZE / SS) * outputScale, (float)(RT_SIZE / SS) * outputScale };
        Vector2 origin = { dst.width * 0.5f, dst.height * 0.5f };
        
        DrawTexturePro(maskTexture.texture, src, dst, origin, 0.0f, WHITE);
    }
};

// int main() {
//     SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
//     InitWindow(1280, 720, "BMO Rig - GOLDEN MASTER + TONGUE");
//     SetTargetFPS(60);

//     ParametricMouth mouth;
//     mouth.Init({640, 360});
//     mouth.outputScale = 1.0f;
    
//     float logTimer = 0.0f;

//     while (!WindowShouldClose()) {
//         float dt = GetFrameTime();
        
//         if (IsKeyDown(KEY_Q)) mouth.target.open += 2.0f * dt;
//         if (IsKeyDown(KEY_A)) mouth.target.open -= 2.0f * dt;
//         if (IsKeyDown(KEY_W)) mouth.target.width += 1.0f * dt;
//         if (IsKeyDown(KEY_S)) mouth.target.width -= 1.0f * dt;
//         if (IsKeyDown(KEY_E)) mouth.target.curve += 2.0f * dt; 
//         if (IsKeyDown(KEY_D)) mouth.target.curve -= 2.0f * dt; 
//         if (IsKeyDown(KEY_T)) mouth.target.teethY += 2.0f * dt;
//         if (IsKeyDown(KEY_G)) mouth.target.teethY -= 2.0f * dt;
//         if (IsKeyDown(KEY_R)) mouth.target.squeeze += 2.0f * dt;
//         if (IsKeyDown(KEY_F)) mouth.target.squeeze -= 2.0f * dt;
//         if (IsKeyPressed(KEY_SPACE)) mouth.usePhysics = !mouth.usePhysics;

//         if (IsKeyDown(KEY_Y)) mouth.target.tongueUp += 2.0f * dt;
//         if (IsKeyDown(KEY_H)) mouth.target.tongueUp -= 2.0f * dt;
//         if (IsKeyDown(KEY_U)) mouth.target.tongueWidth += 1.0f * dt;
//         if (IsKeyDown(KEY_J)) mouth.target.tongueWidth -= 1.0f * dt;

//         if (IsKeyDown(KEY_UP)) mouth.outputScale += 1.0f * dt;
//         if (IsKeyDown(KEY_DOWN)) mouth.outputScale -= 1.0f * dt;
//         mouth.outputScale = Clamp(mouth.outputScale, 0.5f, 3.0f);

//         if (IsKeyDown(KEY_Z)) mouth.debugTeethWidthRatio -= 0.5f * dt;
//         if (IsKeyDown(KEY_X)) mouth.debugTeethWidthRatio += 0.5f * dt;
//         if (IsKeyDown(KEY_C)) mouth.debugTeethGap -= 20.0f * dt;
//         if (IsKeyDown(KEY_V)) mouth.debugTeethGap += 20.0f * dt;

//         mouth.debugTeethWidthRatio = Clamp(mouth.debugTeethWidthRatio, 0.1f, 0.95f);
//         mouth.debugTeethGap = Clamp(mouth.debugTeethGap, 0.0f, 100.0f);

//         if (IsKeyPressed(KEY_ONE))   mouth.target = { 0.05f, 1.17f, 1.0f, 0.0f, -1.0f, 0.0f, 0.65f }; 
//         if (IsKeyPressed(KEY_TWO))   mouth.target = { 1.0f, 0.6f, 0.8f, 0.0f, 0.2f, 0.3f, 0.7f }; 
//         if (IsKeyPressed(KEY_THREE)) mouth.target = { 0.6f, 0.5f, 0.2f, 0.0f, -1.0f, 0.85f, 0.6f }; 
//         if (IsKeyPressed(KEY_FOUR))  mouth.target = { 0.5f, 0.8f, -0.2f, 0.0f, 0.0f, 0.1f, 0.8f }; 

//         mouth.UpdatePhysics(dt);
//         mouth.GenerateGeometry();
        
//         BeginDrawing();
//         ClearBackground({201, 228, 195, 255}); 
//         mouth.Draw();
        
//         DrawText("Controls: Q/A(Open) W/S(Width) E/D(Curve) R/F(Squeeze) T/G(TeethY)", 20, 20, 20, DARKGRAY);
//         DrawText("Tongue: Y/H(Up) U/J(Width) | Teeth: Z/X(Width) C/V(Gap)", 20, 50, 20, DARKGRAY);
//         DrawText("Size: UP/DOWN | Presets: 1-4", 20, 80, 20, DARKGRAY);
        
//         // logTimer += dt;
//         // if (logTimer > 0.2f) {
//         //      printf("Tgt: O:%.2f W:%.2f C:%.2f TUp:%.2f TW:%.2f\n", 
//         //        mouth.target.open, mouth.target.width, mouth.target.curve, 
//         //        mouth.target.tongueUp, mouth.target.tongueWidth);
//         //      logTimer = 0.0f;
//         // }

//         EndDrawing();
//     }
//     mouth.Unload();
//     CloseWindow();
//     return 0;
// }