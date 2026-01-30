// BMO_BlobTalkMouth.cpp
// - Eyes/blink/moods stay sprite-based
// - Expression mouths stay sprite-based (currentMood + "_mouth")
// - Talking mouth uses a VOLUME-PRESERVING SOFT-BODY “BLOB” ring (Verlet + constraints)
// - Single CSV logger written to: GetWorkingDirectory()/procedural_mouth_log.csv (path printed in console)

#include "raylib.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>

using json = nlohmann::json;

// ---------------------------------------------------------
// 0) CSV LOGGER (single logger)
// ---------------------------------------------------------
struct MouthLogger {
    FILE* f = nullptr;
    int frame = 0;

    void Open() {
        std::string path = std::string(GetWorkingDirectory()) + "/procedural_mouth_log.csv";
        TraceLog(LOG_INFO, "MOUTH LOG PATH: %s", path.c_str());

        f = fopen(path.c_str(), "w");
        if (!f) {
            TraceLog(LOG_ERROR, "Failed to open mouth log file!");
            return;
        }

        fprintf(f, "frame,mode,viseme,open,openT,wide,wideT,round,roundT,smile,smileT,width,height,area,areaT,fill,teeth\n");
        fflush(f);
    }

    void Close() {
        if (f) fclose(f);
        f = nullptr;
    }

    void Log(const char* mode, char viseme,
             float open, float openT,
             float wide, float wideT,
             float round, float roundT,
             float smile, float smileT,
             float width, float height,
             float area, float areaT,
             bool fill, bool teeth)
    {
        if (!f) return;
        if ((frame++ % 2) != 0) return;

        fprintf(f, "%d,%s,%c,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%d,%d\n",
                frame, mode, viseme,
                open, openT,
                wide, wideT,
                round, roundT,
                smile, smileT,
                width, height,
                area, areaT,
                fill ? 1 : 0,
                teeth ? 1 : 0);
        fflush(f);
    }
};

static MouthLogger gMouthLog;

// ---------------------------------------------------------
// 1) ASSET LOADER
// ---------------------------------------------------------
struct SpriteAtlas {
    Texture2D texture{};
    std::unordered_map<std::string, Rectangle> frames;

    void Load(const char* imagePath, const char* jsonPath) {
        texture = LoadTexture(imagePath);

        std::ifstream f(jsonPath);
        if (!f.is_open()) {
            TraceLog(LOG_ERROR, "JSON not found: %s", jsonPath);
            return;
        }
        json data = json::parse(f);

        if (data.contains("textures")) {
            for (auto& t : data["textures"]) {
                for (auto& fr : t["frames"]) {
                    std::string name = fr["filename"];
                    frames[name] = {
                        (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                        (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                    };
                }
            }
        }
        TraceLog(LOG_INFO, "Atlas Loaded: %d frames", (int)frames.size());
    }

    void DrawFrame(const std::string& name, int x, int y, int screenW, int screenH,
                   Vector2 offset, Vector2 scale, Color tint)
    {
        auto it = frames.find(name);
        if (it == frames.end()) return;

        Rectangle src = it->second;

        float destW = screenW * scale.x;
        float destH = screenH * scale.y;

        Rectangle dest = {
            (float)x + (screenW - destW) / 2.0f + offset.x,
            (float)y + (screenH - destH) / 2.0f + offset.y,
            destW,
            destH
        };

        DrawTexturePro(texture, src, dest, {0,0}, 0.0f, tint);
    }

    void Unload() { UnloadTexture(texture); }
};

// ---------------------------------------------------------
// 2) LIP SYNC PARSER (char viseme)
// ---------------------------------------------------------
struct LipSyncSystem {
    struct Cue { float time; char shape; };
    std::vector<Cue> cues;

    void Load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            TraceLog(LOG_ERROR, "TSV Missing: %s", path.c_str());
            return;
        }

        float t; std::string s;
        cues.clear();
        while (f >> t >> s) {
            cues.push_back({t, s.empty() ? 'X' : s[0]});
        }
        TraceLog(LOG_INFO, "LipSync Loaded: %d cues", (int)cues.size());
    }

    char GetShapeForTime(float time) const {
        if (cues.empty()) return 'X';
        char shape = 'X';
        for (const auto& cue : cues) {
            if (cue.time > time) break;
            shape = cue.shape;
        }
        return shape;
    }
};

// ---------------------------------------------------------
// 3) SPRING
// ---------------------------------------------------------
struct Spring { float val = 0.0f, vel = 0.0f, target = 0.0f; };

static void UpdateSpring(Spring& s, float k, float d, float dt) {
    float f = (s.target - s.val) * k;
    s.vel = (s.vel + f * dt) * d;
    s.val += s.vel * dt;
}

static float Clamp01(float x) { return (x < 0.0f) ? 0.0f : (x > 1.0f ? 1.0f : x); }
static float Lerp(float a, float b, float t) { return a + (b-a)*t; }

// ---------------------------------------------------------
// 4) BLOB MOUTH (Verlet + edge constraints + area preserve + anchors)
// ---------------------------------------------------------
static Color HexToColor(unsigned int rgb) {
    return Color{
        (unsigned char)((rgb >> 16) & 0xFF),
        (unsigned char)((rgb >> 8) & 0xFF),
        (unsigned char)((rgb) & 0xFF),
        255
    };
}

static Vector2 V2Add(Vector2 a, Vector2 b) { return {a.x+b.x, a.y+b.y}; }
static Vector2 V2Sub(Vector2 a, Vector2 b) { return {a.x-b.x, a.y-b.y}; }
static Vector2 V2Scale(Vector2 a, float s) { return {a.x*s, a.y*s}; }

static float V2Len(Vector2 v) { return sqrtf(v.x*v.x + v.y*v.y); }
static Vector2 V2Norm(Vector2 v) {
    float L = V2Len(v);
    if (L <= 1e-6f) return {0,0};
    return { v.x/L, v.y/L };
}
static Vector2 V2PerpCCW(Vector2 v) { return { -v.y, v.x }; }

static float PolygonAreaShoelace(const std::vector<Vector2>& pts) {
    // signed area (shoelace). We'll keep sign; for a CCW loop it’s positive.
    double a = 0.0;
    int n = (int)pts.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        a += (double)pts[i].x * (double)pts[j].y - (double)pts[j].x * (double)pts[i].y;
    }
    return (float)(0.5 * a);
}

static float PolygonPerimeter(const std::vector<Vector2>& pts) {
    float p = 0.0f;
    int n = (int)pts.size();
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        p += V2Len(V2Sub(pts[j], pts[i]));
    }
    return p;
}

// Catmull-Rom point (uniform)
static Vector2 CatmullRom(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float t2 = t*t;
    float t3 = t2*t;
    return {
        0.5f * ( (2*p1.x) + (-p0.x + p2.x)*t + (2*p0.x - 5*p1.x + 4*p2.x - p3.x)*t2 + (-p0.x + 3*p1.x - 3*p2.x + p3.x)*t3 ),
        0.5f * ( (2*p1.y) + (-p0.y + p2.y)*t + (2*p0.y - 5*p1.y + 4*p2.y - p3.y)*t2 + (-p0.y + 3*p1.y - 3*p2.y + p3.y)*t3 )
    };
}

struct MouthPoint {
    Vector2 pos{};
    Vector2 prev{};
    Vector2 disp{};
    int dispW = 0;

    void Accumulate(Vector2 d) {
        disp = V2Add(disp, d);
        dispW++;
    }
    void ApplyDisplacement() {
        if (dispW > 0) {
            disp = V2Scale(disp, 1.0f/(float)dispW);
            pos = V2Add(pos, disp);
            disp = {0,0};
            dispW = 0;
        }
    }
};

struct BlobMouth {
    // High-level controls (0..1)
    Spring open;   // jaw open
    Spring wide;   // corner spread
    Spring round;  // O-ish
    Spring smile;  // 0..1 -> remap to [-1..+1]

    // Geometry
    std::vector<MouthPoint> pts;
    int N = 0;

    int idxLeft = 0;
    int idxRight = 0;
    int idxTop = 0;
    int idxBottom = 0;

    float restRadius = 40.0f;
    float chordLen = 10.0f;

    // Volume preserve
    float targetArea = 1.0f;
    float areaStiffness = 0.30f;   // how strongly we correct area per solver iteration

    // Constraints
    int solverIters = 10;
    float edgeStiffness = 1.0f;    // 1.0 means full correction
    float anchorStiffness = 0.80f; // 0..1 per iteration

    // Verlet
    float damping = 0.92f; // higher = more floaty, lower = more stable

    // Style (your palette)
    Color mouthFill  = HexToColor(0x396337);       // #396337
    Color outlineCol = {0,0,0,255};
    Color teethCol   = {255,255,255,255};
    float outlinePx  = 10.0f;

    // Debug/render
    int smoothSubdiv = 6; // samples per segment for Catmull-Rom

    // Anchors (computed each frame from viseme)
    Vector2 anchorCenter{};
    Vector2 aLeft{}, aRight{}, aTop{}, aBottom{};
    float lastWidth = 0.0f;
    float lastHeight = 0.0f;

    void Init(Vector2 center, int numPoints, float radius, float puffiness) {
        anchorCenter = center;
        N = (numPoints < 8) ? 8 : numPoints;
        restRadius = radius;

        pts.assign(N, {});

        // Build an initial CCW loop starting at top (like the frog code does)
        for (int i = 0; i < N; i++) {
            float ang = (2.0f * PI * (float)i / (float)N) - (PI/2.0f);
            Vector2 off = { cosf(ang)*restRadius, sinf(ang)*restRadius };
            pts[i].pos  = V2Add(center, off);
            pts[i].prev = pts[i].pos;
        }

        // chord length from circumference
        float circumference = 2.0f * PI * restRadius;
        chordLen = circumference / (float)N;

        // target area
        targetArea = (restRadius * restRadius * PI) * puffiness;

        // indices: based on our angle ordering (top at i=0)
        idxTop = 0;
        idxRight = (int)(N * 0.25f) % N;
        idxBottom = (int)(N * 0.50f) % N;
        idxLeft = (int)(N * 0.75f) % N;

        open.val = open.target = 0.06f;
        wide.val = wide.target = 0.35f;
        round.val = round.target = 0.05f;
        smile.val = smile.target = 0.75f;
    }

    void SetTargetsFromViseme(char v, Vector2 center, Vector2 scale, bool isTalking) {
        anchorCenter = center;

        // defaults
        float tOpen=0.06f, tWide=0.35f, tRound=0.05f, tSmile=0.75f;

        if (isTalking) {
            switch (v) {
                case 'A': tOpen=0.85f; tWide=0.55f; tRound=0.10f; tSmile=0.65f; break;
                case 'E': tOpen=0.35f; tWide=0.95f; tRound=0.00f; tSmile=0.85f; break;
                case 'C':
                case 'D': tOpen=0.55f; tWide=0.55f; tRound=0.05f; tSmile=0.70f; break;
                case 'B':
                case 'F':
                case 'G':
                case 'H': tOpen=0.10f; tWide=0.30f; tRound=0.00f; tSmile=0.65f; break;
                case 'X':
                default:  tOpen=0.06f; tWide=0.35f; tRound=0.05f; tSmile=0.75f; break;
            }
        } else {
            // rest mouth: soft “U” by default
            tOpen=0.05f; tWide=0.45f; tRound=0.00f; tSmile=0.78f;
        }

        open.target  = Clamp01(tOpen);
        wide.target  = Clamp01(tWide);
        round.target = Clamp01(tRound);
        smile.target = Clamp01(tSmile);

        // build anchor targets from smoothed values (use .target? no; use .val after Update())
        // we update springs separately; this function just stores targets.
        (void)v;
        (void)scale;
    }

    void UpdateSprings(float dt) {
        UpdateSpring(open,  40.0f, 0.80f, dt);
        UpdateSpring(wide,  40.0f, 0.80f, dt);
        UpdateSpring(round, 40.0f, 0.80f, dt);
        UpdateSpring(smile, 40.0f, 0.80f, dt);
    }

    void StepPhysics(float dt, Vector2 scale) {
        // Determine anchor targets from current (smoothed) control values
        float o = open.val;
        float w = wide.val;
        float r = round.val;
        float s = (smile.val * 2.0f - 1.0f); // [-1..1]

        // Base mouth dimensions (tune)
        float baseW = 150.0f * scale.x;
        float baseH = 85.0f  * scale.y;

        float W = baseW * Lerp(0.55f, 1.10f, w) * (1.0f - 0.25f * r);
        float H = baseH * Lerp(0.05f, 1.05f, o) * (1.0f + 0.40f * r);

        // Smile lifts corners, also subtly lifts bottom for “U”
        float cornerUp = -18.0f * s * scale.y;

        // Anchor positions
        aLeft   = { anchorCenter.x - W*0.50f, anchorCenter.y + cornerUp };
        aRight  = { anchorCenter.x + W*0.50f, anchorCenter.y + cornerUp };
        aTop    = { anchorCenter.x,          anchorCenter.y - H*0.35f };
        aBottom = { anchorCenter.x,          anchorCenter.y + H*0.55f + (12.0f * s) };

        lastWidth = W;
        lastHeight = (aBottom.y - aTop.y);

        // Update targetArea to loosely follow viseme (this is the “volume control”)
        // Smaller when closed, larger when open; also bump for round.
        float openAreaMult = Lerp(0.35f, 1.15f, o);
        float roundMult    = 1.0f + 0.25f * r;
        float desiredArea  = (restRadius*restRadius*PI) * openAreaMult * roundMult;
        // Smoothly chase desired area to avoid “breathing”
        targetArea = Lerp(targetArea, desiredArea, 0.15f);

        // ---- Verlet integration ----
        // We’re not doing gravity for mouth; just inertia + damping
        for (int i = 0; i < N; i++) {
            Vector2 cur = pts[i].pos;
            Vector2 vel = V2Scale(V2Sub(pts[i].pos, pts[i].prev), damping);
            pts[i].pos  = V2Add(pts[i].pos, vel);
            pts[i].prev = cur;
        }

        // ---- constraint solver ----
        for (int iter = 0; iter < solverIters; iter++) {
            // 1) Edge (distance) constraints (keep rim from stretching/compressing too much)
            for (int i = 0; i < N; i++) {
                int j = (i + 1) % N;

                Vector2 d = V2Sub(pts[j].pos, pts[i].pos);
                float dist = V2Len(d);
                if (dist <= 1e-6f) continue;

                float diff = dist - chordLen;
                // Correct both over and under length
                Vector2 dir = V2Scale(d, 1.0f / dist);
                Vector2 corr = V2Scale(dir, (diff * 0.5f) * edgeStiffness);

                pts[i].Accumulate(corr);
                pts[j].Accumulate(V2Scale(corr, -1.0f));
            }

            // 2) Area preservation (“pressure”)
            // Build a temporary list of positions
            std::vector<Vector2> loop;
            loop.reserve(N);
            for (int i = 0; i < N; i++) loop.push_back(pts[i].pos);

            float curArea = PolygonAreaShoelace(loop);
            // Ensure area sign consistency (if loop is CW, flip normal direction)
            float sign = (curArea >= 0.0f) ? 1.0f : -1.0f;
            float areaAbs = fabsf(curArea);

            float perimeter = PolygonPerimeter(loop);
            perimeter = (perimeter < 1e-3f) ? 1e-3f : perimeter;

            float error = (targetArea - areaAbs);
            float offsetMag = (error / perimeter) * areaStiffness; // “puffiness push”

            for (int i = 0; i < N; i++) {
                int ip = (i == 0) ? (N - 1) : (i - 1);
                int in = (i + 1) % N;

                Vector2 secant = V2Sub(pts[in].pos, pts[ip].pos);
                Vector2 nrm = V2PerpCCW(secant);
                nrm = V2Norm(nrm);

                // sign keeps dilation correct if polygon is CW
                Vector2 push = V2Scale(nrm, offsetMag * sign);
                pts[i].Accumulate(push);
            }

            // 3) Anchor constraints (drive shape)
            // pull 4 key points toward their targets
            auto PullTo = [&](int idx, Vector2 target) {
                Vector2 d = V2Sub(target, pts[idx].pos);
                pts[idx].Accumulate(V2Scale(d, anchorStiffness));
            };
            PullTo(idxLeft,   aLeft);
            PullTo(idxRight,  aRight);
            PullTo(idxTop,    aTop);
            PullTo(idxBottom, aBottom);

            // apply displacement
            for (int i = 0; i < N; i++) pts[i].ApplyDisplacement();
        }
    }

    // Build smooth outline points (Catmull-Rom)
    void BuildSmooth(std::vector<Vector2>& out) const {
        out.clear();
        out.reserve(N * smoothSubdiv);

        // Use positions as control points
        for (int i = 0; i < N; i++) {
            int i0 = (i - 1 + N) % N;
            int i1 = i;
            int i2 = (i + 1) % N;
            int i3 = (i + 2) % N;

            Vector2 p0 = pts[i0].pos;
            Vector2 p1 = pts[i1].pos;
            Vector2 p2 = pts[i2].pos;
            Vector2 p3 = pts[i3].pos;

            for (int s = 0; s < smoothSubdiv; s++) {
                float t = (float)s / (float)smoothSubdiv;
                out.push_back(CatmullRom(p0, p1, p2, p3, t));
            }
        }
    }

    void Draw(char viseme, bool debugMode) const {
        // Determine “open” in pixels using top/bottom points
        float mouthH = fabsf(pts[idxBottom].pos.y - pts[idxTop].pos.y);
        const float FILL_OPEN_PX  = 12.0f;
        const float TEETH_OPEN_PX = 28.0f;

        bool doFill  = (mouthH >= FILL_OPEN_PX);
        bool doTeeth = (mouthH >= TEETH_OPEN_PX);

        // Logging
        // Compute current area (abs)
        std::vector<Vector2> loop;
        loop.reserve(N);
        for (int i = 0; i < N; i++) loop.push_back(pts[i].pos);
        float areaAbs = fabsf(PolygonAreaShoelace(loop));

        gMouthLog.Log(debugMode ? "debug_blob" : "blob",
                      viseme,
                      open.val, open.target,
                      wide.val, wide.target,
                      round.val, round.target,
                      smile.val, smile.target,
                      lastWidth, lastHeight,
                      areaAbs, targetArea,
                      doFill, doTeeth);

        // Build smoothed polyline
        std::vector<Vector2> poly;
        BuildSmooth(poly);

        // If “closed”, draw a clean stroke only (benchmark vibe)
        if (!doFill) {
            for (int i = 0; i < (int)poly.size(); i++) {
                int j = (i + 1) % (int)poly.size();
                DrawLineEx(poly[i], poly[j], outlinePx, outlineCol);
            }
            return;
        }

        // Fill interior
        Vector2 c = {0,0};
        for (auto& p : poly) { c.x += p.x; c.y += p.y; }
        c.x /= (float)poly.size();
        c.y /= (float)poly.size();

        for (int i = 0; i < (int)poly.size(); i++) {
            int j = (i + 1) % (int)poly.size();
            DrawTriangle(c, poly[i], poly[j], mouthFill);
        }

        // Teeth band: sample points along the “upper arc”
        if (doTeeth) {
            // Collect upper-arc indices from left -> right passing through top
            std::vector<Vector2> upper;
            upper.reserve(N);

            int i = idxLeft;
            while (i != idxRight) {
                upper.push_back(pts[i].pos);
                i = (i + 1) % N;
                // stop if we somehow loop too long (safety)
                if ((int)upper.size() > N+2) break;
            }
            upper.push_back(pts[idxRight].pos);

            // find two points near 15% and 85%
            auto SampleUpper = [&](float t)->Vector2 {
                if (upper.size() < 2) return pts[idxTop].pos;
                float f = t * (float)(upper.size() - 1);
                int a = (int)floorf(f);
                int b = a + 1;
                if (b >= (int)upper.size()) b = (int)upper.size() - 1;
                float tt = f - (float)a;
                return {
                    Lerp(upper[a].x, upper[b].x, tt),
                    Lerp(upper[a].y, upper[b].y, tt)
                };
            };

            Vector2 tL = SampleUpper(0.15f);
            Vector2 tR = SampleUpper(0.85f);

            float bandH = Clamp01((mouthH - TEETH_OPEN_PX) / 60.0f) * (mouthH * 0.28f);
            bandH = fmaxf(bandH, 6.0f);

            // Push down inside the mouth
            Vector2 q0 = { tL.x, tL.y + bandH*0.15f };
            Vector2 q1 = { tR.x, tR.y + bandH*0.15f };
            Vector2 q2 = { tR.x, tR.y + bandH };
            Vector2 q3 = { tL.x, tL.y + bandH };

            DrawTriangle(q0, q1, q2, teethCol);
            DrawTriangle(q0, q2, q3, teethCol);

            // Divider line like your sprites
            DrawLineEx(q0, q1, 4.0f, outlineCol);
        }

        // Outline last
        for (int i = 0; i < (int)poly.size(); i++) {
            int j = (i + 1) % (int)poly.size();
            DrawLineEx(poly[i], poly[j], outlinePx, outlineCol);
        }

        // Optional: debug points
        // for (int i = 0; i < N; i++) DrawCircleV(pts[i].pos, 3, RED);
        // DrawCircleV(pts[idxLeft].pos, 6, BLUE);
        // DrawCircleV(pts[idxRight].pos, 6, BLUE);
        // DrawCircleV(pts[idxTop].pos, 6, YELLOW);
        // DrawCircleV(pts[idxBottom].pos, 6, GREEN);
    }
};

// ---------------------------------------------------------
// 5) BMO PUPPET (sprites for eyes + expression mouth; blob only for talk)
// ---------------------------------------------------------
struct BMO {
    SpriteAtlas atlas;
    LipSyncSystem lips;

    std::string currentMood = "face_happy_standard";

    Spring scaleY, scaleX;
    Vector2 lookOffset = {0,0};

    Vector2 shakeOffset = {0,0};
    float pulseScale = 1.0f;

    float blinkTimer = 0.0f, nextBlink = 3.0f;
    bool isBlinking = false;

    bool debugMode = false;
    int debugIndex = 0;
    std::vector<char> debugVisemes = { 'A','B','C','D','E','F','G','H','X' };

    BlobMouth talkMouth;
    bool mouthInited = false;

    void Init(int W, int H) {
        atlas.Load("assets/BMO_Animation_LipSyncSprite.png",
                   "assets/BMO_Animation_Lipsync.json");
        lips.Load("assets/output.tsv");

        scaleY.val = scaleY.target = 1.0f;
        scaleX.val = scaleX.target = 1.0f;

        // Init blob mouth centered like your sprite mouth (screen center + y offset handled in draw)
        Vector2 screenCenter = { (float)W*0.5f, (float)H*0.5f };
        talkMouth.Init(screenCenter, 24, 42.0f, 1.25f);
        mouthInited = true;
    }

    bool ForceSpriteMouth() const {
        // keep iconic expression mouths as sprites
        return (currentMood.find("angry_shout") != std::string::npos) ||
               (currentMood.find("crying") != std::string::npos) ||
               (currentMood.find("kawaii") != std::string::npos) ||
               (currentMood.find("excited_stars") != std::string::npos);
    }

    void Update(float dt, float audioTime, bool isPlaying, int W, int H) {
        // DEBUG toggle
        if (IsKeyPressed(KEY_TAB)) debugMode = !debugMode;
        if (debugMode) {
            if (IsKeyPressed(KEY_RIGHT)) { debugIndex++; if (debugIndex >= (int)debugVisemes.size()) debugIndex = 0; }
            if (IsKeyPressed(KEY_LEFT))  { debugIndex--; if (debugIndex < 0) debugIndex = (int)debugVisemes.size() - 1; }
        }

        // Mood FX
        float time = (float)GetTime();
        shakeOffset = {0, 0};
        pulseScale = 1.0f;

        if (currentMood.find("angry") != std::string::npos || currentMood.find("wail") != std::string::npos) {
            shakeOffset.x = (float)GetRandomValue(-3, 3);
            shakeOffset.y = (float)GetRandomValue(-3, 3);
        }
        else if (currentMood.find("cry") != std::string::npos || currentMood.find("worried") != std::string::npos) {
            shakeOffset.y = sinf(time * 8.0f) * 5.0f;
        }
        else if (currentMood.find("star") != std::string::npos || currentMood.find("excited") != std::string::npos || currentMood.find("kawaii") != std::string::npos) {
            float pulse = sinf(time * 12.0f) * 0.05f;
            pulseScale = 1.0f + pulse;
        }

        // Blinking
        bool canBlink = (currentMood.find("closed") == std::string::npos) &&
                        (currentMood.find("winc") == std::string::npos) &&
                        !debugMode;

        if (canBlink) {
            blinkTimer += dt;
            if (blinkTimer >= nextBlink) {
                isBlinking = true;
                if (blinkTimer >= nextBlink + 0.15f) {
                    isBlinking = false;
                    blinkTimer = 0.0f;
                    nextBlink = GetRandomValue(20, 60) / 10.0f;
                }
            }
        } else {
            isBlinking = false;
        }

        // Mouse look
        Vector2 mouse = GetMousePosition();
        Vector2 center = { (float)GetScreenWidth()/2, (float)GetScreenHeight()/2 };
        Vector2 dir = { mouse.x - center.x, mouse.y - center.y };
        float dist = sqrtf(dir.x*dir.x + dir.y*dir.y);

        if (dist > 0) {
            float move = fminf(dist * 0.05f, 15.0f);
            lookOffset = { (dir.x/dist)*move, (dir.y/dist)*move };
        }

        // Squash/stretch
        float breath = sinf(time * 2.5f) * 0.008f;
        scaleY.target = 1.0f + breath;
        scaleX.target = 1.0f - breath * 0.5f;

        if (isPlaying && !debugMode) {
            char v = lips.GetShapeForTime(audioTime);
            if (v == 'D' || v == 'C') {
                scaleY.target += 0.03f;
                scaleX.target -= 0.015f;
            }
        }

        UpdateSpring(scaleY, 150.0f, 0.6f, dt);
        UpdateSpring(scaleX, 150.0f, 0.6f, dt);

        // Blob mouth update only when used
        bool talkingNow = isPlaying || debugMode;
        bool useBlobTalk = talkingNow && !ForceSpriteMouth();

        if (useBlobTalk && mouthInited) {
            char v = debugMode ? debugVisemes[debugIndex] : lips.GetShapeForTime(audioTime);

            Vector2 screenCenter = { (float)W*0.5f, (float)H*0.5f };
            Vector2 mouthOffset = { shakeOffset.x, -10.0f + shakeOffset.y };
            Vector2 mouthCenter = V2Add(screenCenter, mouthOffset);

            Vector2 finalScale = { scaleX.val * pulseScale, scaleY.val * pulseScale };

            talkMouth.SetTargetsFromViseme(v, mouthCenter, finalScale, true);
            talkMouth.UpdateSprings(dt);
            talkMouth.StepPhysics(dt, finalScale);
        } else if (mouthInited) {
            // still relax the mouth a bit in the background so it doesn’t freeze weirdly
            Vector2 screenCenter = { (float)W*0.5f, (float)H*0.5f };
            Vector2 mouthOffset = { shakeOffset.x, -10.0f + shakeOffset.y };
            Vector2 mouthCenter = V2Add(screenCenter, mouthOffset);
            Vector2 finalScale = { scaleX.val * pulseScale, scaleY.val * pulseScale };

            talkMouth.SetTargetsFromViseme('X', mouthCenter, finalScale, false);
            talkMouth.UpdateSprings(dt);
            talkMouth.StepPhysics(dt, finalScale);
        }
    }

    void Draw(int W, int H, float audioTime, bool isPlaying) {
        std::string eyesTex = isBlinking ? "face_happy_closed_eyes_eyes" : currentMood + "_eyes";

        Vector2 finalScale = { scaleX.val * pulseScale, scaleY.val * pulseScale };
        Vector2 eyesFinalOffset = { lookOffset.x + shakeOffset.x, lookOffset.y + shakeOffset.y };

        // Draw eyes
        atlas.DrawFrame(eyesTex, 0, 0, W, H, eyesFinalOffset, finalScale, WHITE);

        // Mouth placement
        Vector2 screenCenter = { (float)W*0.5f, (float)H*0.5f };
        Vector2 mouthOffset = { shakeOffset.x, -10.0f + shakeOffset.y };
        Vector2 mouthCenter = V2Add(screenCenter, mouthOffset);

        bool talkingNow = isPlaying || debugMode;
        bool useBlobTalk = talkingNow && !ForceSpriteMouth();

        if (useBlobTalk && mouthInited) {
            char v = debugMode ? debugVisemes[debugIndex] : (isPlaying ? lips.GetShapeForTime(audioTime) : 'X');
            (void)mouthCenter; // mouthCenter already baked into physics via Update
            talkMouth.Draw(v, debugMode);
        } else {
            // Expression mouth sprite
            std::string mouthTex = currentMood + "_mouth";
            atlas.DrawFrame(mouthTex, 0, 0, W, H, mouthOffset, finalScale, WHITE);

            // log sprite mode too
            gMouthLog.Log("sprite_expr", 'X',
                          talkMouth.open.val, talkMouth.open.target,
                          talkMouth.wide.val, talkMouth.wide.target,
                          talkMouth.round.val, talkMouth.round.target,
                          talkMouth.smile.val, talkMouth.smile.target,
                          0, 0,
                          0, talkMouth.targetArea,
                          false, false);
        }

        if (debugMode) {
            DrawText("DEBUG MODE", 20, H - 60, 20, YELLOW);
            DrawText(TextFormat("Viseme: %c", debugVisemes[debugIndex]), 20, H - 30, 20, WHITE);
        }
    }
};

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    const int W = 1280, H = 720;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(W, H, "BMO Engine: Sprite Expressions + BLOB Talk Mouth");
    InitAudioDevice();
    SetTargetFPS(60);

    gMouthLog.Open();

    BMO bmo;
    bmo.Init(W, H);

    Music voice = LoadMusicStream("assets/song.wav");
    bool isPlaying = false;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        UpdateMusicStream(voice);

        // Controls
        if (IsKeyPressed(KEY_SPACE)) {
            if (IsMusicStreamPlaying(voice)) { StopMusicStream(voice); isPlaying = false; }
            else { PlayMusicStream(voice); isPlaying = true; }
        }

        // Mood switching
        if (IsKeyPressed(KEY_H)) bmo.currentMood = "face_happy_standard";
        if (IsKeyPressed(KEY_S)) bmo.currentMood = "face_sad_standard";
        if (IsKeyPressed(KEY_A)) bmo.currentMood = "face_angry_shout";
        if (IsKeyPressed(KEY_W)) bmo.currentMood = "face_crying_wail";
        if (IsKeyPressed(KEY_C)) bmo.currentMood = "face_crying_tears";
        if (IsKeyPressed(KEY_E)) bmo.currentMood = "face_excited_stars";
        if (IsKeyPressed(KEY_K)) bmo.currentMood = "face_kawaii_sparkle";

        float time = GetMusicTimePlayed(voice);
        if (!IsMusicStreamPlaying(voice)) time = 0.0f;

        bmo.Update(dt, time, isPlaying, W, H);

        BeginDrawing();
        ClearBackground({ 0xC9, 0xE4, 0xC3, 0xFF }); // #c9e4c3
        bmo.Draw(W, H, time, isPlaying);

        if (!bmo.debugMode) {
            DrawText("SPACE: Audio | TAB: Debug | Left/Right: Viseme", 20, 20, 20, DARKGRAY);
            DrawText("H: Happy | S: Sad | A: Angry | C: Cry | E: Excited | K: Kawaii", 20, 50, 20, DARKGRAY);
        }
        EndDrawing();
    }

    UnloadMusicStream(voice);
    gMouthLog.Close();
    bmo.atlas.Unload();

    CloseAudioDevice();
    CloseWindow();
    return 0;
}
