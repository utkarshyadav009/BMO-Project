// SoftBodyMouth_AuthorTool_Polished.cpp
// - Fixed Rendering Order (Tongue over Teeth)
// - Fixed Masking (Mask shape now matches Spline shape exactly)
// - Physics OFF by default (Sculptor Mode)
// - Dynamic Teeth Height
// - Thicker Tongue

#include "raylib.h"
#include "rlgl.h" 
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
static inline float   V2Angle(Vector2 v)          { return atan2f(v.y, v.x); }
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

// Catmull-Rom Spline for smooth curves
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
        if (locked) return;
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
    std::vector<BlobPoint> pts;      // The Lips
    std::vector<BlobPoint> tongue;   // The Snake Tongue Chain
    
    // Physics Config
    bool physicsEnabled = false; // DEFAULT OFF FOR SCULPTING
    
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
    int grabbedTongueIndex = -1; 
    bool draggingInside = false;
    float radius = 100.0f;

    // Templates System
    std::vector<Vector2> neutralAbs;
    std::vector<std::vector<Vector2>> slots;
    std::vector<Vector2> neutralTongue;
    std::vector<std::vector<Vector2>> tongueSlots;
    int activeSlot = 0;

    // Rendering
    RenderTexture2D mouthMaskTex;
    
    // COLORS (Preserved)
    Color mouthBgCol = { 61, 93, 55, 255 };
    Color tongueCol  = { 152, 161, 101, 255 };
    Color teethCol   = WHITE;
    Color lipCol     = { 0, 0, 0, 255 };

    void Init(Vector2 center, int N, float r) {
        radius = r;
        pts.resize(N);
        for (int i = 0; i < N; i++) {
            float ang = (2.0f * PI * i / N) - (PI/2.0f);
            pts[i].pos = { center.x + cosf(ang)*r, center.y + sinf(ang)*r };
            pts[i].prev = pts[i].pos;
            pts[i].locked = false;
        }

        // Initialize Snake Tongue
        tongue.resize(6);
        for(int i=0; i<6; i++) {
            tongue[i].pos = { center.x, center.y + 60.0f - (i * 15.0f) }; 
            tongue[i].prev = tongue[i].pos;
            tongue[i].locked = (i == 0); 
        }

        float circ = 2.0f * PI * r;
        chordLen = circ / N;
        diagLen = chordLen * 2.0f;
        targetArea = r * r * PI;

        // Init Template Storage
        neutralAbs.resize(N);
        for(int i=0; i<N; i++) neutralAbs[i] = pts[i].pos;
        slots.assign(9, std::vector<Vector2>(N, {0,0}));

        neutralTongue.resize(tongue.size());
        for(int i=0; i<(int)tongue.size(); i++) neutralTongue[i] = tongue[i].pos;
        tongueSlots.assign(9, std::vector<Vector2>(tongue.size(), {0,0}));

        mouthMaskTex = LoadRenderTexture(GetScreenWidth(), GetScreenHeight());
    }

    void ResizeBlob(float scale) {
        Vector2 c = {0,0};
        for(auto& p : pts) c = V2Add(c, p.pos);
        c = V2Mul(c, 1.0f/pts.size());
        for(auto& p : pts) {
            Vector2 dir = V2Sub(p.pos, c);
            p.pos = V2Add(c, V2Mul(dir, scale));
            p.prev = p.pos;
        }
        chordLen *= scale;
        diagLen *= scale;
        targetArea *= (scale * scale);
    }

    // --- Input Handling ---
    void HandleInput() {
        Vector2 m = GetMousePosition();
        static Vector2 lastM = m;
        Vector2 delta = V2Sub(m, lastM);
        lastM = m;

        physicsEnabled = IsKeyDown(KEY_SPACE); // Hold SPACE for physics

        if (IsKeyDown(KEY_LEFT_BRACKET)) ResizeBlob(0.99f);
        if (IsKeyDown(KEY_RIGHT_BRACKET)) ResizeBlob(1.01f);

        // Right Click: Toggle Lock (Lips Only)
        if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)) {
            float bestD = 1e9f; int best = -1;
            for(int i=0; i<(int)pts.size(); i++) {
                float d = V2Len(V2Sub(pts[i].pos, m));
                if (d < bestD) { bestD = d; best = i; }
            }
            if (best != -1 && bestD < 30.0f) {
                pts[best].locked = !pts[best].locked;
                pts[best].prev = pts[best].pos; 
            }
        }

        // Left Click: Grab
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            grabbedIndex = -1;
            grabbedTongueIndex = -1;
            float grabR = 25.0f;

            float bestTD = 1e9f; int bestT = -1;
            for(int i=0; i<(int)tongue.size(); i++) {
                float d = V2Len(V2Sub(tongue[i].pos, m));
                if (d < bestTD) { bestTD = d; bestT = i; }
            }
            if (bestT != -1 && bestTD < grabR) {
                grabbedTongueIndex = bestT;
            } else {
                float bestD = 1e9f; int best = -1;
                for(int i=0; i<(int)pts.size(); i++) {
                    float d = V2Len(V2Sub(pts[i].pos, m));
                    if (d < bestD) { bestD = d; best = i; }
                }
                if (best != -1 && bestD < grabR) {
                    grabbedIndex = best;
                } else {
                    std::vector<Vector2> poly;
                    for(auto& p : pts) poly.push_back(p.pos);
                    if (PointInPoly(m, poly)) draggingInside = true;
                }
            }
        }

        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            grabbedIndex = -1; 
            grabbedTongueIndex = -1;
            draggingInside = false;
        }

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (grabbedTongueIndex != -1) {
                tongue[grabbedTongueIndex].pos = m;
                tongue[grabbedTongueIndex].prev = m;
            }
            else if (grabbedIndex != -1) {
                pts[grabbedIndex].pos = m;
                pts[grabbedIndex].prev = m; 
            } 
            else if (draggingInside) {
                for(auto& p : pts) {
                    if (!p.locked) p.pos = V2Add(p.pos, delta);
                    p.prev = p.pos;
                }
                 for(auto& t : tongue) {
                    if (!t.locked) t.pos = V2Add(t.pos, delta);
                    t.prev = t.pos;
                }
            }
        }
    }

    // --- Physics Step ---
    void Step() {
        if (!physicsEnabled) return; 

        // 1. Lips
        for(auto& p : pts) {
            if (p.locked) { p.prev = p.pos; continue; }
            Vector2 vel = V2Mul(V2Sub(p.pos, p.prev), damping);
            p.prev = p.pos; p.pos = V2Add(p.pos, vel);
        }
        
        // 2. Tongue
        Vector2 c = {0,0}; for(auto& p : pts) c = V2Add(c, p.pos); 
        c = V2Mul(c, 1.0f/pts.size());
        tongue[0].pos = { c.x, c.y + 40 }; 
        tongue[0].locked = true;

        for(auto& p : tongue) {
            if (p.locked) { p.prev = p.pos; continue; }
            Vector2 vel = V2Mul(V2Sub(p.pos, p.prev), 0.85f);
            p.prev = p.pos; p.pos = V2Add(p.pos, vel);
        }

        // 3. Constraints
        for(int k=0; k<solverIters; k++) {
            int N = pts.size();
            for(int i=0; i<N; i++) {
                int next = (i+1)%N;
                Vector2 d = V2Sub(pts[next].pos, pts[i].pos);
                float len = V2Len(d);
                if (len > 0.001f) {
                    float diff = (len - chordLen) / len;
                    Vector2 corr = V2Mul(d, diff * 0.5f * edgeStiff);
                    pts[i].Acc(corr); pts[next].Acc(V2Mul(corr, -1.0f));
                }
            }
            std::vector<Vector2> poly; Vector2 center = {0,0};
            for(auto& p : pts) { poly.push_back(p.pos); center = V2Add(center, p.pos); }
            center = V2Mul(center, 1.0f/N);
            float area = std::abs(PolygonArea(poly));
            if (area < 1.0f) area = 1.0f;
            float s = std::sqrt(targetArea / area);
            for(auto& p : pts) {
                if(p.locked) continue;
                Vector2 dir = V2Sub(p.pos, center);
                Vector2 target = V2Add(center, V2Mul(dir, s));
                p.Acc(V2Mul(V2Sub(target, p.pos), areaStiff * 0.1f));
            }
            
            float segLen = 22.0f; 
            for(int i=0; i<(int)tongue.size()-1; i++) {
                Vector2 d = V2Sub(tongue[i+1].pos, tongue[i].pos);
                float len = V2Len(d);
                if (len > 0.001f) {
                    float diff = (len - segLen) / len;
                    Vector2 corr = V2Mul(d, diff * 0.5f);
                    if (!tongue[i].locked) tongue[i].pos = V2Add(tongue[i].pos, corr);
                    if (!tongue[i+1].locked) tongue[i+1].pos = V2Sub(tongue[i+1].pos, corr);
                }
            }
            for(auto& p : pts) p.ApplyDisp(6.0f);
        }
    }

    void CaptureSlot(int slot) {
        if (slot < 0 || slot >= 9) return;
        for(int i=0; i<(int)pts.size(); i++) slots[slot][i] = V2Sub(pts[i].pos, neutralAbs[i]);
        for(int i=0; i<(int)tongue.size(); i++) tongueSlots[slot][i] = V2Sub(tongue[i].pos, neutralTongue[i]);
        TraceLog(LOG_INFO, "Captured Slot %d", slot);
    }

    void SaveToCSV() {
        FILE* f = fopen("mouth_templates.csv", "w");
        if(!f) return;
        fprintf(f, "N,%d,T,%d\n", (int)pts.size(), (int)tongue.size());
        fprintf(f, "neutral"); 
        for(auto& v : neutralAbs) fprintf(f, ",%.3f,%.3f", v.x, v.y); 
        for(auto& v : neutralTongue) fprintf(f, ",%.3f,%.3f", v.x, v.y);
        fprintf(f, "\n");
        for(int s=0; s<9; s++) {
            fprintf(f, "slot%d", s); 
            for(auto& v : slots[s]) fprintf(f, ",%.3f,%.3f", v.x, v.y); 
            for(auto& v : tongueSlots[s]) fprintf(f, ",%.3f,%.3f", v.x, v.y); 
            fprintf(f, "\n");
        }
        fclose(f);
        TraceLog(LOG_INFO, "Saved CSV with Tongue Data");
    }

// --- Rendering ---
    void Draw() {
        Vector2 center = {0,0};
        for(auto& p : pts) center = V2Add(center, p.pos);
        center = V2Mul(center, 1.0f/pts.size());

        // Teeth Logic
        Vector2 topLip = pts[0].pos;
        Vector2 botLip = pts[pts.size()/2].pos;
        float mouthOpening = V2Len(V2Sub(topLip, botLip));
        float teethHeight = (mouthOpening - 20.0f) * 0.5f; 
        teethHeight = Clamp(teethHeight, 0.0f, 30.0f);

        // 1. GENERATE SMOOTH SPLINE
        std::vector<Vector2> smooth;
        for(int i=0; i<(int)pts.size(); i++) {
            Vector2 p0 = pts[(i-1+pts.size())%pts.size()].pos;
            Vector2 p1 = pts[i].pos;
            Vector2 p2 = pts[(i+1)%pts.size()].pos;
            Vector2 p3 = pts[(i+2)%pts.size()].pos;
            for(int k=0; k<5; k++) smooth.push_back(CatmullRom(p0,p1,p2,p3, k/5.0f));
        }

        // 2. RENDER MASK
        BeginTextureMode(mouthMaskTex);
        ClearBackground(BLANK);

        // A. Draw Mask using SMOOTH POINTS
        Vector2 maskCenter = {0,0};
        for(auto& p : smooth) maskCenter = V2Add(maskCenter, p);
        maskCenter = V2Mul(maskCenter, 1.0f/smooth.size());

        for(size_t i=0; i<smooth.size(); i++) {
            DrawTriangle(maskCenter, smooth[i], smooth[(i+1)%smooth.size()], WHITE);
            DrawTriangle(maskCenter, smooth[(i+1)%smooth.size()], smooth[i], WHITE);
        }

        // [CHANGE 1] EXPAND THE MASK
        // We draw the outline in WHITE here. This tells the alpha channel:
        // "It is okay to draw pixels on the border too."
        for(size_t i=0; i<smooth.size(); i++) {
            DrawLineEx(smooth[i], smooth[(i+1)%smooth.size()], 8.0f, WHITE);
        }

        // B. Lock Alpha
        rlDrawRenderBatchActive();
        rlSetBlendFactors(RL_DST_ALPHA, RL_ONE_MINUS_DST_ALPHA, RL_FUNC_ADD);

        // C. Contents (Background)
        for(size_t i=0; i<smooth.size(); i++) {
            DrawTriangle(maskCenter, smooth[i], smooth[(i+1)%smooth.size()], mouthBgCol);
            DrawTriangle(maskCenter, smooth[(i+1)%smooth.size()], smooth[i], mouthBgCol);
        }

        // [CHANGE 2] DRAW OUTLINE HERE (Inside the Sandwich)
        // Now we draw the actual color outline. Because we are before the tongue,
        // the tongue will cover this line.
        for(size_t i=0; i<smooth.size(); i++) {
            DrawLineEx(smooth[i], smooth[(i+1)%smooth.size()], 8.0f, lipCol);
        }

        if (mouthOpening > 10.0f) {
            // D. TEETH (Drawn on top of Outline)
            if (teethHeight > 1.0f) {
                Vector2 topT = pts[0].pos;
                float topAng = atan2f((pts[1].pos.y - pts[(int)pts.size()-1].pos.y),
                                      (pts[1].pos.x - pts[(int)pts.size()-1].pos.x));

                Vector2 botT = pts[(int)pts.size()/2].pos;
                float botAng = atan2f((pts[(int)pts.size()/2 + 1].pos.y - pts[(int)pts.size()/2 - 1].pos.y),
                                      (pts[(int)pts.size()/2 + 1].pos.x - pts[(int)pts.size()/2 - 1].pos.x));

                rlPushMatrix();
                rlTranslatef(topT.x, topT.y + 10, 0); 
                rlRotatef(topAng * RAD2DEG, 0, 0, 1);
                DrawRectangleRounded({-50, -teethHeight * 0.5f, 100, teethHeight}, 0.4f, 6, teethCol);
                DrawRectangleRoundedLines({-50, -teethHeight * 0.5f, 100, teethHeight}, 0.4f, 6, BLACK); 
                rlPopMatrix();
                
                rlPushMatrix();
                rlTranslatef(botT.x, botT.y - 10, 0);
                rlRotatef(botAng * RAD2DEG, 0, 0, 1);
                DrawRectangleRounded({-45, -teethHeight * 0.5f, 90, teethHeight*0.8f}, 0.3f, 6, teethCol);
                DrawRectangleRoundedLines({-45, -teethHeight * 0.5f, 90, teethHeight*0.8f}, 0.3f, 6, BLACK);
                rlPopMatrix();
            }

            // E. TONGUE (Drawn on top of Teeth AND Outline)
            std::vector<Vector2> tSpline;
            for(int i=0; i<(int)tongue.size()-1; i++) {
                Vector2 p0 = tongue[std::max(0, i-1)].pos;
                Vector2 p1 = tongue[i].pos;
                Vector2 p2 = tongue[i+1].pos;
                Vector2 p3 = tongue[std::min((int)tongue.size()-1, i+2)].pos;
                for(int k=0; k<15; k++) tSpline.push_back(CatmullRom(p0,p1,p2,p3, k/15.0f));
            }
            
            for(int i=0; i<(int)tSpline.size(); i++) {
                float rad = 38.0f - (i * 0.05f); 
                DrawCircleV(tSpline[i], rad, tongueCol);
            }
            for(int i=0; i<(int)tSpline.size()-1; i++) {
                DrawLineEx(tSpline[i], tSpline[i+1], 3.0f, Fade(BLACK, 0.10f));
            }
        }

        rlDrawRenderBatchActive();
        rlSetBlendMode(BLEND_ALPHA);
        EndTextureMode();

        // 3. Draw Texture
        DrawTextureRec(mouthMaskTex.texture, 
                      (Rectangle){0, 0, (float)mouthMaskTex.texture.width, -(float)mouthMaskTex.texture.height},
                      (Vector2){0,0}, WHITE);

        // [CHANGE 3] REMOVED OUTLINE DRAWING FROM HERE
        // (It's now inside the texture, so we don't draw it again)

        // 5. Debug UI
        for(int i=0; i<(int)pts.size(); i++) {
            Color c = pts[i].locked ? RED : GREEN;
            if (i == grabbedIndex) c = YELLOW;
            DrawCircleV(pts[i].pos, 3.0f, c);
        }
        if (mouthOpening > 10.0f) {
            for(int i=0; i<(int)tongue.size(); i++) {
                 DrawCircleLines((int)tongue[i].pos.x, (int)tongue[i].pos.y, 6, Fade(BLUE, 0.6f));
            }
        }
    }
};

int main() {
    InitWindow(1280, 720, "Mouth Sculptor Polished");
    SetTargetFPS(60);
    SoftBlob blob;
    blob.Init({640, 360}, 24, 80.0f);

    while (!WindowShouldClose()) {
        for(int k=0; k<9; k++) if(IsKeyPressed(KEY_ZERO + k)) blob.activeSlot = k;
        if(IsKeyPressed(KEY_ENTER)) blob.CaptureSlot(blob.activeSlot);
        if(IsKeyDown(KEY_LEFT_CONTROL) && IsKeyPressed(KEY_S)) blob.SaveToCSV();

        blob.HandleInput();
        blob.Step();

        BeginDrawing();
        ClearBackground({201, 228, 195, 255});

        blob.Draw();
        
        DrawText(TextFormat("SLOT: %d", blob.activeSlot), 20, 20, 30, DARKGRAY);
        DrawText("SCULPT MODE", 20, 60, 20, DARKGREEN);
        DrawText("Teeth/Tongue appear when mouth opens", 20, 85, 20, DARKBLUE);
        
        EndDrawing();
    }
    UnloadRenderTexture(blob.mouthMaskTex);
    CloseWindow();
    return 0;
}