// ShaderParametricMouth.cpp
// SHADER-BASED IMPLEMENTATION
// Fixes: Scaled Outlines & Rounded Teeth

#include "raylib.h"
#include "rlgl.h" 
#include "raymath.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream> 

// ------------------------------
// Math & Geometry Helpers
// ------------------------------
static inline Vector2 V2Add(Vector2 a, Vector2 b) { return {a.x + b.x, a.y + b.y}; }
static inline Vector2 V2Sub(Vector2 a, Vector2 b) { return {a.x - b.x, a.y - b.y}; }
static inline Vector2 V2Scale(Vector2 a, float s) { return {a.x * s, a.y * s}; }
static inline int ClampInt(int v, int lo, int hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }
static inline float V2Dist(Vector2 a, Vector2 b) { return hypotf(b.x - a.x, b.y - a.y); }

// static Vector4 ColorNormalize(Color c) {
//     return { c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f };
// }

static Vector2 CatmullRom(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float t2 = t * t; float t3 = t2 * t;
    float v0 = ((-t3) + (2 * t2) - t) * 0.5f;
    float v1 = ((3 * t3) - (5 * t2) + 2) * 0.5f;
    float v2 = ((-3 * t3) + (4 * t2) + t) * 0.5f;
    float v3 = (t3 - t2) * 0.5f;
    return { (p0.x * v0) + (p1.x * v1) + (p2.x * v2) + (p3.x * v3),
             (p0.y * v0) + (p1.y * v1) + (p2.y * v2) + (p3.y * v3) };
}

// ------------------------------
// Geometry Processing
// ------------------------------
static float GetPolyPerimeter(const std::vector<Vector2>& poly, bool closed) {
    float len = 0.0f;
    for (size_t i = 0; i < poly.size(); i++) {
        if (!closed && i == poly.size() - 1) break;
        len += V2Dist(poly[i], poly[(i + 1) % poly.size()]);
    }
    return len;
}

static void ResamplePoly(const std::vector<Vector2>& input, float* outputFlat, int maxPts, int& outCount, float scale, Vector2 offset, bool closedLoop) {
    if (input.empty()) { outCount = 0; return; }

    if (input.size() <= (size_t)maxPts) {
        outCount = (int)input.size();
        for(int i=0; i<outCount; i++) {
            outputFlat[i*2]     = (input[i].x - offset.x) * scale;
            outputFlat[i*2 + 1] = (input[i].y - offset.y) * scale;
        }
        return;
    }

    outCount = maxPts;
    float totalLen = GetPolyPerimeter(input, closedLoop);
    float step = totalLen / (float)(closedLoop ? maxPts : maxPts - 1);
    
    outputFlat[0] = (input[0].x - offset.x) * scale;
    outputFlat[1] = (input[0].y - offset.y) * scale;
    int ptsEmitted = 1;
    float currentDist = 0.0f;
    float nextSample = step;
    
    for (size_t i = 0; i < input.size(); i++) {
        if (ptsEmitted >= maxPts) break;
        if (!closedLoop && i == input.size() - 1) break;

        Vector2 p1 = input[i];
        Vector2 p2 = input[(i + 1) % input.size()];
        float segLen = V2Dist(p1, p2);
        
        if (segLen < 1e-6f) { currentDist += segLen; continue; }
        
        while (nextSample <= currentDist + segLen) {
            float t = (nextSample - currentDist) / segLen;
            float px = p1.x + (p2.x - p1.x) * t;
            float py = p1.y + (p2.y - p1.y) * t;
            
            outputFlat[ptsEmitted*2]     = (px - offset.x) * scale;
            outputFlat[ptsEmitted*2 + 1] = (py - offset.y) * scale;
            ptsEmitted++;
            nextSample += step;
            if (ptsEmitted >= maxPts) break;
        }
        currentDist += segLen;
    }
}

static Rectangle GetBoundingBox(const std::vector<Vector2>& poly) {
    if(poly.empty()) return {0,0,0,0};
    float minX = 1e10f, maxX = -1e10f, minY = 1e10f, maxY = -1e10f;
    for(auto& p : poly) {
        if(p.x < minX) minX = p.x; if(p.x > maxX) maxX = p.x;
        if(p.y < minY) minY = p.y; if(p.y > maxY) maxY = p.y;
    }
    return {minX, minY, maxX - minX, maxY - minY};
}

// ------------------------------
// Helpers
// ------------------------------
static void RemoveNearDuplicates(std::vector<Vector2>& poly, float eps=0.5f) {
    if (poly.size() < 2) return;
    std::vector<Vector2> out; out.push_back(poly[0]);
    for (size_t i=1; i<poly.size(); i++){
        if (V2Dist(poly[i], out.back()) > eps) out.push_back(poly[i]);
    }
    if (out.size() > 2 && V2Dist(out.front(), out.back()) <= eps) out.pop_back();
    poly = out;
}

static void RotateContourToLeftmost(std::vector<Vector2>& c) {
    if (c.empty()) return;
    int best = 0;
    for (int i = 1; i < (int)c.size(); i++) {
        if (c[i].x < c[best].x || (c[i].x == c[best].x && c[i].y < c[best].y)) best = i;
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

// ------------------------------
// MAIN CLASS
// ------------------------------
struct FacialParams {
    float open = 0.05f; float width = 0.5f; float curve = 0.0f; 
    float squeezeTop = 0.0f; float squeezeBottom = 0.0f; 
    float teethY = 0.0f; float tongueUp = 0.0f; float tongueX = 0.0f;
    float tongueWidth = 0.65f; float asymmetry = 0.0f; float squareness = 0.0f;
    float teethWidth = 0.50f; float teethGap = 45.0f; float scale = 1.0f; float outlineThickness = 1.5f;
    float sigma = 0.45f; float power = 6.0f; float maxLiftValue = 0.55f;
};

static FacialParams MakeDefaultParams() {
    FacialParams p{};
    p.open = 0.05f; p.width = 0.5f; p.curve = 0.2f; p.scale = 1.0f;
    p.teethWidth = 0.5f; p.teethGap = 45.0f; p.tongueWidth = 0.65f;
    p.teethY = -1.0f; p.squeezeTop = 0.0f; p.squeezeBottom = 0.0f;
    p.tongueUp = 0.0f; p.tongueX = 0.0f; p.asymmetry = 0.0f;
    p.squareness = 0.0f; p.sigma = 0.45f; p.power = 6.0f; p.maxLiftValue = 0.55f;
    p.outlineThickness = 1.5f;
    return p;
}

struct ParametricMouth {
    FacialParams current, target, velocity;

    std::vector<Vector2> controlPoints; 
    std::vector<Vector2> smoothContour; 
    std::vector<Vector2> topTeethPoly, botTeethPoly;
    std::vector<Vector2> tonguePoly;
    
    Shader sdfShader;
    bool shaderLoaded = false;
    
    int locMouthPts, locMouthCnt;
    int locTopPts, locTopCnt;
    int locBotPts, locBotCnt;
    int locTonguePts, locTongueCnt;
    int locRes, locPadding, locScale, locOutlineThickness;
    int locColBg, locColLine, locColTeeth, locColTongue;

    Vector2 centerPos; 
    bool usePhysics = true;
    const float GEO_SIZE = 1024.0f;
    const float SS = 4.0f; 

    Color colBg     = { 57, 99, 55, 255 };
    Color colLine   = { 20, 35, 20, 255 };
    Color colTeeth  = { 245, 245, 245, 255 };
    Color colTongue = { 162, 178, 106, 255 }; 

    void Init(Vector2 pos) {
        velocity = {}; 
        centerPos = pos;
        
        controlPoints.resize(16);
        smoothContour.reserve(512);
        topTeethPoly.reserve(64);
        botTeethPoly.reserve(64);
        tonguePoly.reserve(64);
        
        // Robust Shader Loading
        const char* filename = "mouth_es.fs";
        std::string path = std::string(GetApplicationDirectory()) + filename;
        if (!FileExists(path.c_str())) path = filename;

        if (!FileExists(path.c_str())) {
             std::cout << "\n[CRITICAL] SHADER FILE MISSING AT: " << path << std::endl;
        } else {
             sdfShader = LoadShader(0, path.c_str());
             if (sdfShader.id > 0) {
                 shaderLoaded = true;
                 locMouthPts = GetShaderLocation(sdfShader, "uMouthPts");
                 locMouthCnt = GetShaderLocation(sdfShader, "uMouthCount");
                 locTopPts   = GetShaderLocation(sdfShader, "uTopTeethPts");
                 locTopCnt   = GetShaderLocation(sdfShader, "uTopTeethCount");
                 locBotPts   = GetShaderLocation(sdfShader, "uBotTeethPts");
                 locBotCnt   = GetShaderLocation(sdfShader, "uBotTeethCount");
                 locTonguePts= GetShaderLocation(sdfShader, "uTonguePts");
                 locTongueCnt= GetShaderLocation(sdfShader, "uTongueCount");
                 
                 locRes      = GetShaderLocation(sdfShader, "uResolution");
                 locPadding  = GetShaderLocation(sdfShader, "uPadding");
                 locScale    = GetShaderLocation(sdfShader, "uScale"); // [NEW] Get Scale Loc
                 locOutlineThickness = GetShaderLocation(sdfShader, "uOutlineThickness");
                 
                 locColBg    = GetShaderLocation(sdfShader, "uColBg");
                 locColLine  = GetShaderLocation(sdfShader, "uColLine");
                 locColTeeth = GetShaderLocation(sdfShader, "uColTeeth");
                 locColTongue= GetShaderLocation(sdfShader, "uColTongue");
                 std::cout << "[Mouth] SUCCESS: Shader Loaded. ID: " << sdfShader.id << std::endl;
             }
        }
        target = MakeDefaultParams();
        current = target;
    }

    void Unload() { if(shaderLoaded) UnloadShader(sdfShader); }
    void resetPosition(Vector2 pos) { centerPos = pos; }
    

    void UpdatePhysics(float dt) {
        target.open = Clamp(target.open, 0.0f, 1.2f);
        target.scale = Clamp(target.scale, 0.5f, 4.0f);

        if (!usePhysics) { current = target; return; }
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
        Upd(current.sigma, velocity.sigma, target.sigma);
        Upd(current.power, velocity.power, target.power);
        Upd(current.maxLiftValue, velocity.maxLiftValue, target.maxLiftValue);
        Upd(current.teethY, velocity.teethY, target.teethY);
        Upd(current.tongueUp, velocity.tongueUp, target.tongueUp);
        Upd(current.tongueX, velocity.tongueX, target.tongueX);
        Upd(current.tongueWidth, velocity.tongueWidth, target.tongueWidth);
        Upd(current.asymmetry, velocity.asymmetry, target.asymmetry);
        Upd(current.squareness, velocity.squareness, target.squareness);
        Upd(current.teethWidth, velocity.teethWidth, target.teethWidth);
        Upd(current.teethGap,   velocity.teethGap,   target.teethGap);
        Upd(current.scale,     velocity.scale,     target.scale);
        Upd(current.outlineThickness, velocity.outlineThickness, target.outlineThickness);
    }

    void GenerateGeometry() {
        float cx = GEO_SIZE * 0.5f; 
        float cy = GEO_SIZE * 0.5f;
        float baseRadius = 40.0f * SS; 
        float w = baseRadius * (0.5f + current.width);
        float h = (current.open < 0.08f) ? 0.0f : (baseRadius * (0.2f + current.open * 1.5f));

        for (int i = 0; i < 16; i++) {
            float t = (float)i / 16.0f;
            float angle = t * PI * 2.0f + PI; 
            float x = cosf(angle) * w;
            
            // 1. BASE SHAPE (Superellipse with Flattening)
            float rawSin = sinf(angle);
            bool isTop = (i <= 8); // Indices 0-8 are Top Arch
            
            // [NEW] D-SHAPE LOGIC
            // Asymmetry now controls "Flatness" instead of just vertical shift.
            // +Asymmetry = Flat Top (Talking)
            // -Asymmetry = Flat Bottom (Smile)
            float flatness = 0.0f;
            if (current.asymmetry > 0.0f && isTop) flatness = current.asymmetry; 
            if (current.asymmetry < 0.0f && !isTop) flatness = -current.asymmetry;

            // Apply flatness
            float effectiveH = h;
            float wave = std::abs(rawSin);
            
            if (flatness > 0.01f) {
                // Squashing: Reduce height of this arch
                effectiveH *= (1.0f - flatness * 0.6f); 
                
                // Boxiness: Make the corners sharper on the flat side
                // This creates the "D" shape corners
                wave = std::pow(wave, 1.0f - (flatness * 0.5f)); 
            }

            // Global Squareness
            float sqPower = 1.0f - (current.squareness * 0.8f);
            float shapedSin = std::pow(wave, sqPower);
            
            // Re-apply sign
            float sign = (rawSin >= 0.0f) ? 1.0f : -1.0f;
            float y = shapedSin * effectiveH * sign;
            
            // 2. BEND (Global Curve)
            // [NEW] Don't bend the flat part, or it becomes a banana
            float bendFactor = 15.0f * SS; 
            float normalizedX = x / w;
            float rawBend = (normalizedX * normalizedX) * bendFactor * current.curve;
            
            float bendMult = 1.0f;
            if (flatness > 0.0f) bendMult = 1.0f - flatness; // Reduce bend on flat side
            
            y -= rawBend * bendMult; 

            // 3. SQUEEZE (Center lift) - smooth bean bump
            float activeSqueeze = isTop ? current.squeezeTop : current.squeezeBottom;
                    
            // Normalize position across width (0 at center, 1 at corners)
            float tX = std::abs(x) / (w + 1e-6f);
            tX = Clamp(tX, 0.0f, 1.0f);
                    
            // Super-Gaussian bump: 1 at center -> smoothly to 0 at edges
            // sigma controls how wide the lifted region is.
            // power controls how flat/round the top of the bump is.
            // float sigma = 0.42f;      // try 0.45..0.70
            // float power = 6.5f;       // 2 = gaussian, 4/6 = rounder "U"
            float u = tX / current.sigma;
            float influence = expf(-powf(u, current.power));  // very smooth
                    
            // Additive displacement (NOT scaling y) to preserve curvature
            // Tie magnitude to h so it behaves consistently as mouth opens.
            float maxLift = (h * current.maxLiftValue);              // try 0.45..0.75
            float lift = activeSqueeze * influence * maxLift;
                    
            // Direction: top arch should go DOWN, bottom arch should go UP
            // Using rawSin sign is robust with your topology.
            float archSign = (rawSin >= 0.0f) ? 1.0f : -1.0f;
            y -= lift * archSign;

            
            controlPoints[i] = { cx + x, cy + y };
        }

        // [FIX] Increased subdivisions from 8 to 16 for ultra-smooth sides
        smoothContour.clear();
        for (int i = 0; i < 16; i++) {
            Vector2 p0 = controlPoints[(i-1+16)%16];
            Vector2 p1 = controlPoints[i];
            Vector2 p2 = controlPoints[(i+1)%16];
            Vector2 p3 = controlPoints[(i+2)%16];
            for (int k = 0; k < 16; k++) { 
                smoothContour.push_back(CatmullRom(p0, p1, p2, p3, (float)k/16.0f));
            }
        }
        RemoveNearDuplicates(smoothContour);
        NormalizeContourForTeeth(smoothContour);

        topTeethPoly.clear(); botTeethPoly.clear(); tonguePoly.clear();
        
        if (current.open > 0.10f && !smoothContour.empty()) {
            float minY = 1e10, maxY = -1e10, minX = 1e10, maxX = -1e10;
            for (const auto& p : smoothContour) {
                if(p.y < minY) minY = p.y; if(p.y > maxY) maxY = p.y;
                if(p.x < minX) minX = p.x; if(p.x > maxX) maxX = p.x;
            }
            
            // [FIX] ROUNDER TONGUE GEOMETRY
            float shiftX = current.tongueX * (w * 0.4f);
            float tongueCX = cx + shiftX;
            float tongueW = (maxX - minX) * 0.5f * current.tongueWidth;
            float tongueTip = Lerp(maxY, minY, current.tongueUp);
            float tongueBase = maxY;
            
            const int tongueSegs = 32;
            tonguePoly.clear();
            for (int i = 0; i <= tongueSegs; i++) {
                float t = (float)i / (float)tongueSegs; 
                float tx = -cosf(t * PI); 
                float ty = sinf(t * PI);  
                
                float px = tongueCX + (tx * tongueW);
                float py = Lerp(tongueBase, tongueTip, ty);
                tonguePoly.push_back({px, py});
            }
            tonguePoly.push_back({tongueCX + tongueW, tongueBase + 10});
            tonguePoly.push_back({tongueCX - tongueW, tongueBase + 10});
            
            // ... Inside GenerateGeometry ...
            // 1. Calculate Bounds
            float teethW = (maxX - minX) * current.teethWidth * 0.5f;
            float gap = current.teethGap * SS;
            float midY = cy + (current.teethY * 20.0f * SS);

            // CPU Culling Checks
            float mouthTop = minY; 
            float mouthBottom = maxY;
            float topTeethEdge = midY - gap/2;
            float botTeethEdge = midY + gap/2;

            // 2. TOP TEETH (Flat line - no side edges when fully extended)
            if(gap != 400 && topTeethEdge > mouthTop + 5.0f) 
            {
                // When teethWidth is at full (1.0), create a flat horizontal line
                // Otherwise, create a box that tapers with width
                if(current.teethWidth >= 0.95f) {
                    // Fully extended: flat line of teeth across the mouth
                    topTeethPoly = {
                        {minX, minY},         // Top Left (mouth edge)
                        {maxX, minY},         // Top Right (mouth edge)
                        {maxX, topTeethEdge}, // Bottom Right
                        {minX, topTeethEdge}  // Bottom Left
                    };
                } else {
                    // Partial width: tapered box
                    topTeethPoly = {
                        {cx - teethW, minY},         // Top Left
                        {cx + teethW, minY},         // Top Right
                        {cx + teethW, topTeethEdge}, // Bottom Right
                        {cx - teethW, topTeethEdge}  // Bottom Left
                    };
                }
            }

            // 3. BOTTOM TEETH (Flat line - no side edges when fully extended)
            if(gap != 400 && botTeethEdge < mouthBottom - 5.0f)
            {
                // When teethWidth is at full (1.0), create a flat horizontal line
                if(current.teethWidth >= 0.95f) {
                    // Fully extended: flat line of teeth across the mouth
                    botTeethPoly = {
                        {minX, botTeethEdge}, // Top Left (mouth edge)
                        {maxX, botTeethEdge}, // Top Right (mouth edge)
                        {maxX, maxY},         // Bottom Right
                        {minX, maxY}          // Bottom Left
                    };
                } else {
                    // Partial width: tapered box
                    botTeethPoly = {
                        {cx - teethW, botTeethEdge}, // Top Left
                        {cx + teethW, botTeethEdge}, // Top Right
                        {cx + teethW, maxY},         // Bottom Right
                        {cx - teethW, maxY}          // Bottom Left
                    };
                }
            }
            // // 1. Calculate Bounds
            // float teethW = (maxX - minX) * current.teethWidth * 0.5f;
            // float gap = current.teethGap * SS;
            // float midY = cy + (current.teethY * 20.0f * SS);

            // // CPU Culling Checks
            // float mouthTop = minY; 
            // float mouthBottom = maxY;
            // float topTeethEdge = midY - gap/2;
            // float botTeethEdge = midY + gap/2;

            // // 2. TOP TEETH (Simple Box)
            // // We only send the 4 corners. The shader will turn this into a round pill.
            // if(gap != 400 && topTeethEdge > mouthTop + 5.0f) 
            // {
            //     topTeethPoly = {
            //         {cx - teethW, minY},         // Top Left
            //         {cx + teethW, minY},         // Top Right
            //         {cx + teethW, topTeethEdge}, // Bottom Right
            //         {cx - teethW, topTeethEdge}  // Bottom Left
            //     };
            // }

            // // 3. BOTTOM TEETH (Simple Box)
            // if(gap != 400 && botTeethEdge < mouthBottom - 5.0f)
            // {
            //     botTeethPoly = {
            //         {cx - teethW, botTeethEdge}, // Top Left
            //         {cx + teethW, botTeethEdge}, // Top Right
            //         {cx + teethW, maxY},         // Bottom Right
            //         {cx - teethW, maxY}          // Bottom Left
            //     };
            // }
        }
    }

    void Draw() {
        if (!shaderLoaded) return;
        if (smoothContour.empty()) return;

        const float paddingPx = 8.0f; 
        float unitScale = (256.0f / GEO_SIZE) * current.scale;
        if (unitScale < 0.0001f) unitScale = 0.0001f;

        Rectangle boundsPhys = GetBoundingBox(smoothContour);
        float paddingPhys = paddingPx / unitScale;
        boundsPhys.x -= paddingPhys;
        boundsPhys.y -= paddingPhys;
        boundsPhys.width += paddingPhys * 2.0f;
        boundsPhys.height += paddingPhys * 2.0f;

        float physRefLeft = centerPos.x - (GEO_SIZE * 0.5f * unitScale);
        float physRefTop  = centerPos.y - (GEO_SIZE * 0.5f * unitScale);
        
        Rectangle screenRect = {
            physRefLeft + (boundsPhys.x * unitScale),
            physRefTop  + (boundsPhys.y * unitScale),
            boundsPhys.width * unitScale,
            boundsPhys.height * unitScale
        };

        const int MAX_PTS = 64;
        float flatMouth[MAX_PTS * 2];
        float flatTop[MAX_PTS * 2];
        float flatBot[MAX_PTS * 2];
        float flatTongue[MAX_PTS * 2];
        
        int cntMouth=0, cntTop=0, cntBot=0, cntTongue=0;
        Vector2 offset = {boundsPhys.x, boundsPhys.y};
        
        ResamplePoly(smoothContour, flatMouth, MAX_PTS, cntMouth, unitScale, offset, true);
        ResamplePoly(topTeethPoly, flatTop, MAX_PTS, cntTop, unitScale, offset, true);
        ResamplePoly(botTeethPoly, flatBot, MAX_PTS, cntBot, unitScale, offset, true);
        ResamplePoly(tonguePoly, flatTongue, MAX_PTS, cntTongue, unitScale, offset, true);

        BeginShaderMode(sdfShader);
            SetShaderValueV(sdfShader, locMouthPts, flatMouth, SHADER_UNIFORM_VEC2, MAX_PTS);
            SetShaderValue(sdfShader, locMouthCnt, &cntMouth, SHADER_UNIFORM_INT);
            SetShaderValueV(sdfShader, locTopPts, flatTop, SHADER_UNIFORM_VEC2, MAX_PTS);
            SetShaderValue(sdfShader, locTopCnt, &cntTop, SHADER_UNIFORM_INT);
            SetShaderValueV(sdfShader, locBotPts, flatBot, SHADER_UNIFORM_VEC2, MAX_PTS);
            SetShaderValue(sdfShader, locBotCnt, &cntBot, SHADER_UNIFORM_INT);
            SetShaderValueV(sdfShader, locTonguePts, flatTongue, SHADER_UNIFORM_VEC2, MAX_PTS);
            SetShaderValue(sdfShader, locTongueCnt, &cntTongue, SHADER_UNIFORM_INT);
            
            float res[2] = { screenRect.width, screenRect.height };
            SetShaderValue(sdfShader, locRes, res, SHADER_UNIFORM_VEC2);
            SetShaderValue(sdfShader, locPadding, &paddingPx, SHADER_UNIFORM_FLOAT);
            // [NEW] Send current scale to Shader for outline correction
            SetShaderValue(sdfShader, locScale, &current.scale, SHADER_UNIFORM_FLOAT);
            SetShaderValue(sdfShader, locOutlineThickness, &current.outlineThickness, SHADER_UNIFORM_FLOAT);
            
            Vector4 cBg = ColorNormalize(colBg);
            Vector4 cLn = ColorNormalize(colLine);
            Vector4 cTh = ColorNormalize(colTeeth);
            Vector4 cTg = ColorNormalize(colTongue);
            
            SetShaderValue(sdfShader, locColBg, &cBg, SHADER_UNIFORM_VEC4);
            SetShaderValue(sdfShader, locColLine, &cLn, SHADER_UNIFORM_VEC4);
            SetShaderValue(sdfShader, locColTeeth, &cTh, SHADER_UNIFORM_VEC4);
            SetShaderValue(sdfShader, locColTongue, &cTg, SHADER_UNIFORM_VEC4);

            rlBegin(RL_QUADS);
                rlColor4ub(255, 255, 255, 255);
                rlTexCoord2f(0.0f, 0.0f); rlVertex2f(screenRect.x, screenRect.y);
                rlTexCoord2f(0.0f, 1.0f); rlVertex2f(screenRect.x, screenRect.y + screenRect.height);
                rlTexCoord2f(1.0f, 1.0f); rlVertex2f(screenRect.x + screenRect.width, screenRect.y + screenRect.height);
                rlTexCoord2f(1.0f, 0.0f); rlVertex2f(screenRect.x + screenRect.width, screenRect.y);
            rlEnd();
        EndShaderMode();
    }
};