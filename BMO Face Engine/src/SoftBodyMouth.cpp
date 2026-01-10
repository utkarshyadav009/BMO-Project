// SoftBodyMouth_AuthorTool_Fixed.cpp
// FIXED VERSION:
// 1. Rendering uses "Triangle Fan" (Robust, never disappears)
// 2. Locked points can be dragged to reposition anchors
// 3. High contrast visuals

#include "raylib.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>

// ------------------------------
// Math Helpers
// ------------------------------
static inline Vector2 V2Add(Vector2 a, Vector2 b) { return {a.x + b.x, a.y + b.y}; }
static inline Vector2 V2Sub(Vector2 a, Vector2 b) { return {a.x - b.x, a.y - b.y}; }
static inline Vector2 V2Mul(Vector2 a, float s)   { return {a.x * s, a.y * s}; }
static inline float   V2Len(Vector2 v)            { return std::sqrt(v.x*v.x + v.y*v.y); }
static inline float   Clamp(float x, float a, float b) { return (x < a) ? a : ((x > b) ? b : x); }

// Area for physics
static float PolygonArea(const std::vector<Vector2>& pts) {
    double a = 0.0;
    int n = (int)pts.size();
    if (n < 3) return 0.0f;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        a += (double)pts[i].x * (double)pts[j].y - (double)pts[j].x * (double)pts[i].y;
    }
    return (float)(0.5 * a);
}

// Point in Poly (for clicking inside)
static bool PointInPoly(Vector2 p, const std::vector<Vector2>& poly) {
    bool c = false;
    int n = (int)poly.size();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (((poly[i].y > p.y) != (poly[j].y > p.y)) &&
            (p.x < (poly[j].x - poly[i].x) * (p.y - poly[i].y) / (poly[j].y - poly[i].y + 1e-9f) + poly[i].x))
            c = !c;
    }
    return c;
}

// Catmull-Rom Spline for smooth lips
static Vector2 CatmullRom(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    auto tj = [](float ti, Vector2 a, Vector2 b) {
        return ti + std::sqrt(std::max(V2Len(V2Sub(b, a)), 1e-6f));
    };
    float t0 = 0.0f;
    float t1 = tj(t0, p0, p1);
    float t2 = tj(t1, p1, p2);
    float t3 = tj(t2, p2, p3);
    float tt = t1 + (t2 - t1) * t;

    auto lerp = [](Vector2 a, Vector2 b, float u) { return V2Add(a, V2Mul(V2Sub(b, a), u)); };
    Vector2 A1 = lerp(p0, p1, (tt - t0) / (t1 - t0));
    Vector2 A2 = lerp(p1, p2, (tt - t1) / (t2 - t1));
    Vector2 A3 = lerp(p2, p3, (tt - t2) / (t3 - t2));
    Vector2 B1 = lerp(A1, A2, (tt - t0) / (t2 - t0));
    Vector2 B2 = lerp(A2, A3, (tt - t1) / (t3 - t1));
    return lerp(B1, B2, (tt - t1) / (t2 - t1));
}

// ------------------------------
// Core Structures
// ------------------------------
struct BlobPoint {
    Vector2 pos{};
    Vector2 prev{};
    Vector2 disp{};
    int dispW = 0;
    bool locked = false;

    void Acc(Vector2 d) {
        if (locked) return; // Physics cannot move a locked point
        disp = V2Add(disp, d);
        dispW++;
    }

    void ApplyDisp(float maxStep) {
        if (locked) { disp = {0,0}; dispW = 0; return; }
        if (dispW > 0) {
            Vector2 avg = V2Mul(disp, 1.0f / (float)dispW);
            float L = V2Len(avg);
            if (L > maxStep) avg = V2Mul(avg, maxStep / (L + 1e-6f));
            pos = V2Add(pos, avg);
            disp = {0,0}; dispW = 0;
        }
    }
};

// ------------------------------
// The Mouth Blob System
// ------------------------------
struct SoftBlob {
    std::vector<BlobPoint> pts;
    
    // Physics Config
    float targetArea = 1.0f;
    float chordLen = 10.0f;
    float diagLen = 10.0f;
    int solverIters = 16;
    float edgeStiff = 0.90f;
    float diagStiff = 0.25f;
    float areaStiff = 0.18f;
    float damping = 0.92f;

    // Interaction
    int grabbedIndex = -1;
    bool draggingInside = false;
    float radius = 100.0f;

    // Templates System
    std::vector<Vector2> neutralAbs; // The T-Pose
    std::vector<std::vector<Vector2>> slots; // Stores OFFSETS
    int activeSlot = 0;

    void Init(Vector2 center, int N, float r) {
        radius = r;
        pts.resize(N);
        for (int i = 0; i < N; i++) {
            float ang = (2.0f * PI * i / N) - (PI/2.0f);
            pts[i].pos = { center.x + cosf(ang)*r, center.y + sinf(ang)*r };
            pts[i].prev = pts[i].pos;
            pts[i].locked = false;
        }

        float circ = 2.0f * PI * r;
        chordLen = circ / N;
        diagLen = chordLen * 2.0f;
        targetArea = r * r * PI;

        // Init Template Slots
        neutralAbs.resize(N);
        for(int i=0; i<N; i++) neutralAbs[i] = pts[i].pos;
        slots.assign(9, std::vector<Vector2>(N, {0,0}));
    }

    // --- Input Handling ---
    void HandleInput() {
        Vector2 m = GetMousePosition();
        static Vector2 lastM = m;
        Vector2 delta = V2Sub(m, lastM);
        lastM = m;

        // 1. RIGHT CLICK: Toggle Lock
        if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
            float bestD = 1e9f;
            int best = -1;
            for(int i=0; i<(int)pts.size(); i++) {
                float d = V2Len(V2Sub(pts[i].pos, m));
                if (d < bestD) { bestD = d; best = i; }
            }
            // Increase grab radius to make clicking easier
            if (best != -1 && bestD < 30.0f) {
                pts[best].locked = !pts[best].locked;
                // Kill velocity so it stops instantly
                pts[best].prev = pts[best].pos; 
            }
        }

        // 2. LEFT CLICK: Grab Point OR Drag Inside
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            grabbedIndex = -1;
            float bestD = 1e9f;
            int best = -1;
            for(int i=0; i<(int)pts.size(); i++) {
                float d = V2Len(V2Sub(pts[i].pos, m));
                if (d < bestD) { bestD = d; best = i; }
            }
            
            if (best != -1 && bestD < 30.0f) {
                grabbedIndex = best;
            } else {
                std::vector<Vector2> poly;
                for(auto& p : pts) poly.push_back(p.pos);
                if (PointInPoly(m, poly)) draggingInside = true;
            }
        }

        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            grabbedIndex = -1;
            draggingInside = false;
        }

        // 3. DRAGGING LOGIC
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (grabbedIndex != -1) {
                // FORCE position even if locked (Kinematic move)
                pts[grabbedIndex].pos = m;
                pts[grabbedIndex].prev = m; 
            } else if (draggingInside) {
                // Brush effect
                for(auto& p : pts) {
                    if (p.locked) continue; // Don't move locked points with brush
                    float d = V2Len(V2Sub(p.pos, m));
                    if (d < radius * 1.2f) {
                        float strength = 1.0f - (d / (radius * 1.2f));
                        p.Acc(V2Mul(delta, strength * 1.5f));
                    }
                }
            }
        }
    }

    // --- Physics ---
    void Step() {
        // Verlet
        for(auto& p : pts) {
            if (p.locked) { p.prev = p.pos; continue; }
            Vector2 vel = V2Mul(V2Sub(p.pos, p.prev), damping);
            p.prev = p.pos;
            p.pos = V2Add(p.pos, vel);
        }

        // Constraints
        for(int k=0; k<solverIters; k++) {
            // Edges
            int N = pts.size();
            for(int i=0; i<N; i++) {
                int next = (i+1)%N;
                Vector2 d = V2Sub(pts[next].pos, pts[i].pos);
                float len = V2Len(d);
                if (len > 0.001f) {
                    float diff = (len - chordLen) / len;
                    Vector2 corr = V2Mul(d, diff * 0.5f * edgeStiff);
                    pts[i].Acc(corr);
                    pts[next].Acc(V2Mul(corr, -1.0f));
                }
            }
            // Diagonals
            for(int i=0; i<N; i++) {
                int next = (i+2)%N;
                Vector2 d = V2Sub(pts[next].pos, pts[i].pos);
                float len = V2Len(d);
                if (len > 0.001f) {
                    float diff = (len - diagLen) / len;
                    Vector2 corr = V2Mul(d, diff * 0.5f * diagStiff);
                    pts[i].Acc(corr);
                    pts[next].Acc(V2Mul(corr, -1.0f));
                }
            }
            
            // Area Preservation
            std::vector<Vector2> poly;
            Vector2 c = {0,0};
            for(auto& p : pts) { poly.push_back(p.pos); c = V2Add(c, p.pos); }
            c = V2Mul(c, 1.0f/N);
            float area = std::abs(PolygonArea(poly));
            if (area < 1.0f) area = 1.0f;
            float s = std::sqrt(targetArea / area);
            
            for(auto& p : pts) {
                if(p.locked) continue;
                Vector2 dir = V2Sub(p.pos, c);
                Vector2 target = V2Add(c, V2Mul(dir, s));
                Vector2 force = V2Mul(V2Sub(target, p.pos), areaStiff * 0.1f);
                p.Acc(force);
            }

            for(auto& p : pts) p.ApplyDisp(6.0f);
        }
    }

    // --- Save/Load Logic ---
    void CaptureSlot(int slot) {
        if (slot < 0 || slot >= 9) return;
        for(int i=0; i<(int)pts.size(); i++) {
            slots[slot][i] = V2Sub(pts[i].pos, neutralAbs[i]);
        }
        TraceLog(LOG_INFO, "Captured Slot %d", slot);
    }
    // NEW: Resize the entire blob (Simulate muscle contraction)
    void ResizeBlob(float scale) {
        // 1. Find center
        Vector2 c = {0,0};
        for(auto& p : pts) c = V2Add(c, p.pos);
        c = V2Mul(c, 1.0f/pts.size());

        // 2. Scale positions and physics constants
        for(auto& p : pts) {
            Vector2 dir = V2Sub(p.pos, c);
            p.pos = V2Add(c, V2Mul(dir, scale));
            p.prev = p.pos; // Reset velocity so it doesn't explode
        }

        // 3. Update physics constraints so it stays this size
        chordLen *= scale;
        diagLen *= scale;
        targetArea *= (scale * scale);
    }
    void SaveToCSV() {
        FILE* f = fopen("mouth_templates.csv", "w");
        if(!f) return;
        fprintf(f, "N,%d\n", (int)pts.size());
        fprintf(f, "neutral");
        for(auto& v : neutralAbs) fprintf(f, ",%.3f,%.3f", v.x, v.y);
        fprintf(f, "\n");
        for(int s=0; s<9; s++) {
            fprintf(f, "slot%d", s);
            for(auto& v : slots[s]) fprintf(f, ",%.3f,%.3f", v.x, v.y);
            fprintf(f, "\n");
        }
        fclose(f);
        TraceLog(LOG_INFO, "Saved CSV");
    }

    // --- Rendering ---
    void Draw() {
        // 1. Fill (The Fix: Triangle Fan)
        // This is robust. It draws from center to every edge.
        // It won't disappear even if concave.
        Vector2 center = {0,0};
        for(auto& p : pts) center = V2Add(center, p.pos);
        center = V2Mul(center, 1.0f/pts.size());
        
        Color mouthColor = { 57, 99, 55, 255 }; // Dark Red
        
        for(int i=0; i<(int)pts.size(); i++) {
            int next = (i+1)%pts.size();
            // Draw double-sided to prevent culling issues
            DrawTriangle(center, pts[i].pos, pts[next].pos, mouthColor); 
            DrawTriangle(center, pts[next].pos, pts[i].pos, mouthColor); 
        }

        // 2. Smooth Outline
        std::vector<Vector2> smooth;
        for(int i=0; i<(int)pts.size(); i++) {
            Vector2 p0 = pts[(i-1+pts.size())%pts.size()].pos;
            Vector2 p1 = pts[i].pos;
            Vector2 p2 = pts[(i+1)%pts.size()].pos;
            Vector2 p3 = pts[(i+2)%pts.size()].pos;
            for(int k=0; k<5; k++) smooth.push_back(CatmullRom(p0,p1,p2,p3, k/5.0f));
        }
        for(size_t i=0; i<smooth.size(); i++) {
            DrawLineEx(smooth[i], smooth[(i+1)%smooth.size()], 6.0f, {0, 0, 0, 255}); // Red Lips
        }

        // 3. Points
        for(int i=0; i<(int)pts.size(); i++) {
            Color c = pts[i].locked ? RED : GREEN;
            if (i == grabbedIndex) c = YELLOW;
            
            float r = pts[i].locked ? 7.0f : 4.0f;
            DrawCircleV(pts[i].pos, r, c);
            DrawCircleLines((int)pts[i].pos.x, (int)pts[i].pos.y, r, BLACK);
        }
    }
};

int main() {
    InitWindow(1024, 768, "Mouth Tool: Authoring");
    SetTargetFPS(60);
    SoftBlob blob;
    blob.Init({512, 384}, 20, 100.0f);

    while (!WindowShouldClose()) {
        // Key Input
        for(int k=0; k<9; k++) if(IsKeyPressed(KEY_ZERO + k)) blob.activeSlot = k;
        if(IsKeyPressed(KEY_ENTER)) blob.CaptureSlot(blob.activeSlot);
        if(IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_S)) blob.SaveToCSV();

        blob.HandleInput();
        blob.Step();
        if (IsKeyDown(KEY_LEFT_BRACKET)) blob.ResizeBlob(0.99f); // Shrink (Contract)
        if (IsKeyDown(KEY_RIGHT_BRACKET)) blob.ResizeBlob(1.01f); // Grow (Relax)
        BeginDrawing();
        ClearBackground({201, 228, 195, 255}); // Your background color

        blob.Draw();
        
        // 1. Current Status (Large)
        DrawText(TextFormat("CURRENT SLOT: %d", blob.activeSlot), 20, 20, 30, DARKGRAY);

        // 2. Mouse Instructions
        DrawText("MOUSE: Right-Click to Lock | Left-Click to Drag", 20, 60, 20, DARKGRAY);

        // 3. NEW: Save/Capture Instructions (Blue for visibility)
        DrawText("KEYS:  [0-9] Select Slot  |  [ENTER] Capture Shape  |  [CTRL+S] Save to File", 20, 90, 20, DARKBLUE);
        DrawText("KEYS: [ / ] Shrink/Grow | [0-9] Slot | [ENTER] Capture | [CTRL+S] Save", 20, 120, 20, DARKBLUE);
        EndDrawing();
    }
    CloseWindow();
    return 0;
}