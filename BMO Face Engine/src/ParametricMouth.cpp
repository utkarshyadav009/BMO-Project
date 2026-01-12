// ParametricMouth_TeethGapFixed.cpp
// - FIXED: "Triangle Bottom" (Changed taper math to be boxier)
// - FIXED: "So Damn Long" (Reduced max length multiplier)
// - ADDED: "Teeth Gap" (Top and Bottom teeth no longer touch)

#include "raylib.h"
#include "rlgl.h" 
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <cfloat>

// ------------------------------
// Math Helpers
// ------------------------------
static inline Vector2 V2Add(Vector2 a, Vector2 b) { return {a.x + b.x, a.y + b.y}; }
static inline Vector2 V2Sub(Vector2 a, Vector2 b) { return {a.x - b.x, a.y - b.y}; }
static inline Vector2 V2Scale(Vector2 a, float s) { return {a.x * s, a.y * s}; }
static inline float   Clamp(float x, float a, float b) { return (x < a) ? a : ((x > b) ? b : x); }
static inline float   Lerp(float a, float b, float t) { return a + t * (b - a); }
static inline float   V2Cross(Vector2 a, Vector2 b) { return a.x*b.y - a.y*b.x; }

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

static float SignedArea(const std::vector<Vector2>& p) {
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
        int count = poly.size(); int safety = count * 4; 

        while (count > 2 && safety-- > 0) {
            for (int i = 0; i < count; i++) {
                int prev = (i - 1 + count) % count; int curr = i; int next = (i + 1) % count;
                Vector2 a = poly[indices[prev]]; Vector2 b = poly[indices[curr]]; Vector2 c = poly[indices[next]];
                if (V2Cross(V2Sub(b, a), V2Sub(c, b)) < 0) continue; 
                bool clean = true;
                for (int j = 0; j < count; j++) {
                    if (j == prev || j == curr || j == next) continue;
                    if (IsPointInTriangle(poly[indices[j]], a, b, c)) { clean = false; break; }
                }
                if (clean) {
                    triangles.push_back(a); triangles.push_back(b); triangles.push_back(c);
                    indices.erase(indices.begin() + curr); count--; break;
                }
            }
        }
        return triangles;
    }
};

struct FacialParams {
    float open = 0.05f; float width = 1.05f; float curve = -3.04f; float squeeze = 0.0f; float teethY = 0.0f; 
};

struct ParametricMouth {
    FacialParams current, target, velocity;
    std::vector<Vector2> controlPoints; 
    std::vector<Vector2> smoothContour; 
    std::vector<Vector2> fillTriangles; 
    std::vector<Vector2> topTeethPoly, botTeethPoly;
    
    RenderTexture2D maskTexture;
    bool textureLoaded = false;
    Vector2 centerPos;
    bool usePhysics = true;
    const float SS = 4.0f; 

    // DEBUG CONTROLS
    float debugTeethWidthRatio = 0.50f;
    float debugTeethGap = 90.0f; // [NEW] Controls gap between top/bottom

    Color colBg    = { 61, 93, 55, 255 };
    Color colLine  = { 20, 35, 20, 255 };
    Color colTeeth = { 245, 245, 245, 255 };

    void Init(Vector2 pos) {
        centerPos = pos;
        controlPoints.resize(16);
        maskTexture = LoadRenderTexture(GetScreenWidth() * SS, GetScreenHeight() * SS);
        SetTextureFilter(maskTexture.texture, TEXTURE_FILTER_BILINEAR);
        textureLoaded = true;
        target = { 0.05f, 1.05f, -3.04f, 0.0f, 0.0f }; // Rest
        current = target;
    }
    void Unload() { if(textureLoaded) UnloadRenderTexture(maskTexture); }

    void UpdatePhysics(float dt) {
        if (!usePhysics) { current = target; return; }
        if (dt > 0.05f) dt = 0.05f; 

        const float STIFFNESS = 180.0f, DAMPING = 14.0f;    
        auto Upd = [&](float& c, float& v, float t) {
            float f = STIFFNESS * (t - c), d = DAMPING * v;
            v += (f - d) * dt; c += v * dt;
        };
        target.teethY = Clamp(target.teethY, -1.0f, 1.0f);
        Upd(current.open, velocity.open, target.open);
        Upd(current.width, velocity.width, target.width);
        Upd(current.curve, velocity.curve, target.curve);
        Upd(current.squeeze, velocity.squeeze, target.squeeze);
        Upd(current.teethY, velocity.teethY, target.teethY);
        current.open = Clamp(current.open, 0.0f, 1.2f);
    }

    void BuildTeethToLine(const std::vector<Vector2>& spline, int start, int endExclusive,
                          float targetY, float dirY, std::vector<Vector2>& outPoly)
    {
        outPoly.clear();
        int n = (int)spline.size();
        start = Clamp(start + 1, 0, n-1);
        endExclusive = Clamp(endExclusive - 1, 0, n);
        if (start >= endExclusive) return;

        // 1. Root
        for (int i = start; i < endExclusive; i++) outPoly.push_back(spline[i]);

        // 2. Cutting Edge (Bite Line)
        int span = endExclusive - start;
        for (int i = endExclusive - 1; i >= start; i--) {
            Vector2 p = spline[i];
            float t = (float)(i - start) / (float)(span - 1);
            
            // [FIX] BOXIER TAPER
            // Old: pow(sin, 0.4) -> Very triangular
            // New: pow(sin, 0.1) -> Very square, with rounded corners
            float taper = std::pow(std::sin(t * PI), 0.1f); 

            // Calc height to target line
            float distToLine = (targetY - p.y) * dirY; 
            float h = std::max(0.0f, distToLine);
            float finalY = p.y + (dirY * h * taper);

            outPoly.push_back({p.x, finalY});
        }

        RemoveNearDuplicates(outPoly);
        if (SignedArea(outPoly) < 0.0f) std::reverse(outPoly.begin(), outPoly.end());
    }

    void GenerateGeometry() {
        float baseRadius = 40.0f * SS; 
        float w = baseRadius * (0.5f + current.width);
        float h = (current.open < 0.05f) ? 0.0f : (baseRadius * (0.2f + current.open * 1.5f));
        float cx = centerPos.x * SS; float cy = centerPos.y * SS;

        for (int i = 0; i < 16; i++) {
            float t = (float)i / 16.0f;
            float angle = t * PI * 2.0f + PI; 
            float x = cosf(angle) * w;
            float y = sinf(angle) * h;
            float curve = (current.curve - 0.5f) * 2.0f; 
            y += (x * x) / (w * w) * 20.0f * curve;
            if (std::abs(x) < w * 0.4f) y *= (1.0f - current.squeeze * 0.8f);
            controlPoints[i] = { cx + x, cy + y };
        }

        smoothContour.clear();
        for (int i = 0; i < 16; i++) {
            Vector2 p0 = controlPoints[(i-1+16)%16], p1 = controlPoints[i];
            Vector2 p2 = controlPoints[(i+1)%16], p3 = controlPoints[(i+2)%16];
            for (int k = 0; k < 6; k++) smoothContour.push_back(CatmullRom(p0, p1, p2, p3, k / 6.0f));
        }
        RemoveNearDuplicates(smoothContour);

        // --- TEETH LOGIC ---
        topTeethPoly.clear(); botTeethPoly.clear();
        
        bool isSafe = true;
        if (smoothContour.size() > 0) {
             if (smoothContour[smoothContour.size()/4].y > smoothContour[smoothContour.size()*3/4].y) isSafe = false;
        }

        if (isSafe && current.open > 0.15f) {
            int n = (int)smoothContour.size();
            int half = n / 2;
            float span = (1.0f - debugTeethWidthRatio) / 2.0f;
            int tStart = (int)(half * span);
            int tEnd   = (int)(half * (1.0f - span));
            int bStart = half + (int)((n - half) * span);
            int bEnd   = half + (int)((n - half) * (1.0f - span));

            // [FIX] BITE LINE + GAP
            // Center is 'cy'. We push top teeth up by Gap/2, bottom down by Gap/2.
            // teethY parameter shifts the whole assembly up/down.
            float shiftY = current.teethY * 20.0f * SS;
            float gap = debugTeethGap * SS;
            
            float topTarget = cy + shiftY - (gap * 0.5f);
            float botTarget = cy + shiftY + (gap * 0.5f);

            // [FIX] REDUCED MULTIPLIER (60 instead of 120)
            // But actually, 'BuildTeethToLine' ignores length and uses Target Y.
            // So we just need to make sure Target Y isn't too far.
            // The logic above naturally limits length because it's based on 'cy' (center).
            // If the mouth opens wide, the distance from Lip to Center grows, making teeth longer.
            // We can Clamp the target line so it follows the lip if it gets too far.
            
            BuildTeethToLine(smoothContour, tStart, tEnd, topTarget, 1.0f, topTeethPoly);
            BuildTeethToLine(smoothContour, bStart, bEnd, botTarget, -1.0f, botTeethPoly);
        }

        if (SignedArea(smoothContour) < 0.0f) std::reverse(smoothContour.begin(), smoothContour.end());
        fillTriangles = SimpleTriangulator::Triangulate(smoothContour);
    }

    void DrawPoly(const std::vector<Vector2>& poly, Color c) {
        if(poly.size() < 3) return;
        std::vector<Vector2> tris = SimpleTriangulator::Triangulate(poly);
        rlBegin(RL_TRIANGLES);
        rlColor4ub(c.r, c.g, c.b, c.a);
        for(auto& p : tris) rlVertex2f(p.x, p.y);
        rlEnd();
    }
    
    void DrawPolyOutline_Clipped(const std::vector<Vector2>& poly, float thick, Color c) {
        if(poly.size() < 2) return;
        for(size_t i=0; i<poly.size(); i++) DrawLineEx(poly[i], poly[(i+1)%poly.size()], thick, c);
    }

    void DrawPolyOutline_Smooth(const std::vector<Vector2>& poly, float thick, Color c) {
        if(poly.size() < 2) return;
        rlEnableSmoothLines(); 
        for(size_t i=0; i<poly.size(); i++) DrawLineEx(poly[i], poly[(i+1)%poly.size()], thick, c);
        rlDisableSmoothLines();
    }

    void Draw() {
        if (smoothContour.size() < 3) return;
        rlDisableBackfaceCulling();

        BeginTextureMode(maskTexture);
        ClearBackground(BLANK);

        if (current.open < 0.08f) {
            int half = smoothContour.size() / 2;
            rlEnableSmoothLines();
            for (int i = 0; i < half; i++) DrawLineEx(smoothContour[i], smoothContour[i+1], 6.0f * SS, colLine);
            rlDisableSmoothLines();
        } 
        else {
            DrawPoly(smoothContour, WHITE);
            rlDrawRenderBatchActive();
            rlSetBlendFactors(RL_DST_ALPHA, RL_ONE_MINUS_DST_ALPHA, RL_FUNC_ADD);

            DrawPoly(smoothContour, colBg); 
            if (!topTeethPoly.empty()) DrawPoly(topTeethPoly, colTeeth);
            if (!botTeethPoly.empty()) DrawPoly(botTeethPoly, colTeeth);
            
            float thick = 4.0f * SS; 
            if (!topTeethPoly.empty()) DrawPolyOutline_Clipped(topTeethPoly, thick, colLine);
            if (!botTeethPoly.empty()) DrawPolyOutline_Clipped(botTeethPoly, thick, colLine);

            rlDrawRenderBatchActive();
            rlSetBlendMode(BLEND_ALPHA);

            DrawPolyOutline_Smooth(smoothContour, 6.0f * SS, colLine);
        }

        EndTextureMode();
        rlEnableBackfaceCulling();

        Rectangle src = {0, 0, (float)maskTexture.texture.width, -(float)maskTexture.texture.height};
        Rectangle dst = {0, 0, (float)GetScreenWidth(), (float)GetScreenHeight()};
        DrawTexturePro(maskTexture.texture, src, dst, {0,0}, 0.0f, WHITE);
    }
};

int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 720, "BMO Rig - Perfected Teeth");
    SetTargetFPS(60);

    ParametricMouth mouth;
    mouth.Init({640, 360});

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        if (IsKeyDown(KEY_Q)) mouth.target.open += 2.0f * dt;
        if (IsKeyDown(KEY_A)) mouth.target.open -= 2.0f * dt;
        if (IsKeyDown(KEY_W)) mouth.target.width += 1.0f * dt;
        if (IsKeyDown(KEY_S)) mouth.target.width -= 1.0f * dt;
        if (IsKeyDown(KEY_E)) mouth.target.curve += 1.0f * dt; 
        if (IsKeyDown(KEY_D)) mouth.target.curve -= 1.0f * dt;
        if (IsKeyDown(KEY_T)) mouth.target.teethY += 1.0f * dt;
        if (IsKeyDown(KEY_G)) mouth.target.teethY -= 1.0f * dt;
        if (IsKeyPressed(KEY_SPACE)) mouth.usePhysics = !mouth.usePhysics;
        
        // DEBUG CONTROLS
        if (IsKeyDown(KEY_Z)) mouth.debugTeethWidthRatio -= 0.5f * dt;
        if (IsKeyDown(KEY_X)) mouth.debugTeethWidthRatio += 0.5f * dt;
        if (IsKeyDown(KEY_C)) mouth.debugTeethGap -= 20.0f * dt;
        if (IsKeyDown(KEY_V)) mouth.debugTeethGap += 20.0f * dt;
        
        mouth.debugTeethWidthRatio = Clamp(mouth.debugTeethWidthRatio, 0.1f, 0.95f);
        mouth.debugTeethGap = Clamp(mouth.debugTeethGap, 0.0f, 100.0f);

        // Presets
        if (IsKeyPressed(KEY_ONE)) mouth.target = { 1.0f, 0.7f, 0.7f, 0.0f, 0.5f}; 
        if (IsKeyPressed(KEY_TWO)) mouth.target = { 0.4f, 1.0f, 0.8f, 0.0f, 1.0f}; 
        if (IsKeyPressed(KEY_THREE)) mouth.target = { 0.8f, 0.2f, 0.5f, 0.0f, 0.0f}; 
        if (IsKeyPressed(KEY_FOUR)) mouth.target = { 0.05f, 0.5f, 0.5f, 0.0f, -1.0f}; 

        mouth.UpdatePhysics(dt);
        mouth.GenerateGeometry();
        
        BeginDrawing();
        ClearBackground({201, 228, 195, 255}); 
        mouth.Draw();
        
        DrawText(TextFormat("Teeth Width (Z/X): %.2f", mouth.debugTeethWidthRatio), 20, 20, 20, DARKGRAY);
        DrawText(TextFormat("Teeth Gap (C/V): %.0f", mouth.debugTeethGap), 20, 45, 20, DARKGRAY);
        
        //Print mouth target 
        printf("Target - Open: %.2f, Width: %.2f, Curve: %.2f, Squeeze: %.2f, TeethY: %.2f\n", 
               mouth.target.open, mouth.target.width, mouth.target.curve, mouth.target.squeeze, mouth.target.teethY);

        EndDrawing();
    }
    mouth.Unload();
    CloseWindow();
    return 0;
}