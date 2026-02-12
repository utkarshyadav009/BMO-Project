// FinalFaceShader.cpp
// UNIFIED: Multi-Shader Architecture (Eyes + Mouth)
// Features: Hot-Reload, Physics, Pixelation, Global Scaling, SDF Geometry

#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "utility.h"
#include "FaceData.h"
// --------------------------------------------------------
// MATH UTILS (From Mouth Implementation)
// --------------------------------------------------------
namespace MathUtils {
    static inline float V2Dist(Vector2 a, Vector2 b) { return hypotf(b.x - a.x, b.y - a.y); }
    
    static Vector2 CatmullRom(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
        float t2 = t * t; float t3 = t2 * t;
        float v0 = ((-t3) + (2 * t2) - t) * 0.5f;
        float v1 = ((3 * t3) - (5 * t2) + 2) * 0.5f;
        float v2 = ((-3 * t3) + (4 * t2) + t) * 0.5f;
        float v3 = (t3 - t2) * 0.5f;
        return { (p0.x * v0) + (p1.x * v1) + (p2.x * v2) + (p3.x * v3),
                 (p0.y * v0) + (p1.y * v1) + (p2.y * v2) + (p3.y * v3) };
    }

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
    
    static int ClampInt(int v, int lo, int hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }
    static Vector4 ColorNormalize(Color c) { return { c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f }; }

    //Not using this function anymore, keeping it cause its cool 
    //Wrote this function because I wanted to have sharp corners 
    //but now I am using a built in raylib rect 
    static void BuildRoundedRectContour(std::vector<Vector2>& out, float cx, float cy,float halfW, float halfH, float radius, int arcSeg = 8)
    {
        out.clear();

        radius = Clamp(radius, 0.0f, std::min(halfW, halfH));
        if (radius <= 1e-5f) {
            // Sharp rectangle (flat ends)
            out.push_back({cx - halfW, cy - halfH});
            out.push_back({cx + halfW, cy - halfH});
            out.push_back({cx + halfW, cy + halfH});
            out.push_back({cx - halfW, cy + halfH});
            return;
        }

        const float left   = cx - halfW;
        const float right  = cx + halfW;
        const float top    = cy - halfH;
        const float bottom = cy + halfH;

        // Corner centers
        Vector2 cTR{right - radius, top + radius};
        Vector2 cTL{left + radius,  top + radius};
        Vector2 cBL{left + radius,  bottom - radius};
        Vector2 cBR{right - radius, bottom - radius};

        auto arc = [&](Vector2 c, float a0, float a1) {
            for (int i = 0; i <= arcSeg; ++i) {
                float t = (float)i / (float)arcSeg;
                float a = Lerp(a0, a1, t);
                out.push_back({c.x + cosf(a) * radius, c.y + sinf(a) * radius});
            }
        };

        // Clockwise: TR -> TL -> BL -> BR
        arc(cTR, -PI * 0.5f, 0.0f);
        arc(cBR, 0.0f, PI * 0.5f);
        arc(cBL, PI * 0.5f, PI);
        arc(cTL, PI, PI * 1.5f);

        MathUtils::RemoveNearDuplicates(out);
    }

    static inline float Smoothstep(float a, float b, float x)
    {
        float t = Clamp((x - a) / (b - a + 1e-6f), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }


}

// --------------------------------------------------------
// CONFIG
// --------------------------------------------------------
namespace Config {
    // Shader Filenames
    const char* SHADER_EYE       = "eyes_es.fs";
    const char* SHADER_BROW      = "brow_es.fs";
    const char* SHADER_TEAR      = "tears_es.fs";
    const char* SHADER_MOUTH     = "mouth_es.fs";
    const char* SHADER_PIXELIZER = "pixelizer_es.fs";
    const char* SHADER_DIR       = "src/";

    // Animation & Physics (Eyes)
    const float BLINK_MIN_DELAY = 2.0f;
    const float BLINK_MAX_DELAY = 5.0f;
    const float BLINK_HOLD_DURATION = 0.10f;
    const float BLINK_CLOSE_THRESHOLD = 0.2f;
    const float BLINK_OPEN_THRESHOLD = 0.9f;

    // Drawing
    const float REF_EYE_SIZE = 100.0f;
    const float EYE_SCALE_FACTOR = 2.7f;
    const float EYE_HEIGHT_OFFSET = 150.0f;
}

// --------------------------------------------------------
// PHYSICS STRUCTS
// --------------------------------------------------------
struct Spring {
    float val, vel, target;
    float stiffness, damping;

    void Reset(float initial) {
        val = initial; target = initial; vel = 0.0f;
    }

    void Update(float dt) {
        float f = stiffness * (target - val);
        vel = (vel + f * dt) * damping;
        val += vel * dt;
    }
};

// --------------------------------------------------------
// SHADER WRAPPER (Unified)
// --------------------------------------------------------
struct ShaderAsset {
    Shader shader;
    std::string filePath;
    long lastModTime;
    
    // Cached Uniform Locations (Union of all possible locations)
    struct Locations {
        // Common
        int resolution, time, color;
        // Eye
        int shape, bend, thickness, side, spiral, distort, stress, gloom, squareness;
        // Brow
        int browLen, browY, browAngle, browBendOffset;
        // Tears
        int scale, tearLevel, blushMode, showBlush, blushCol, tearCol;
        // Mouth
        int mouthPts, mouthCnt, topPts, topCnt, botPts, botCnt, tonguePts, tongueCnt;
        int padding, outlineThickness;
        int colBg, colLine, colTeeth, colTongue, stressLines;
        // Pixel
        int locPixelSize, locRenderSize;
    } locs;

    void Load(const char* filename, const char* dir) {
        if (FileExists(filename)) filePath = filename;
        else if (FileExists((std::string(dir) + filename).c_str())) filePath = std::string(dir) + filename;
        else filePath = filename; 

        shader = LoadShader(0, filePath.c_str());
        lastModTime = GetFileModTime(filePath.c_str());
        if(shader.id == 0) std::cout << "[ShaderAsset] Failed to load: " << filePath << std::endl;
    }

    void ReloadIfChanged() {
        long currentModTime = GetFileModTime(filePath.c_str());
        if (currentModTime > lastModTime) {
            std::cout << "[HOT RELOAD] Reloading shader: " << filePath << std::endl;
            Shader newShader = LoadShader(0, filePath.c_str());
            
            if (newShader.id != rlGetShaderIdDefault()) {
                UnloadShader(shader);
                shader = newShader;
                lastModTime = currentModTime;
                TraceLog(LOG_INFO, "HOT RELOAD: Success (%s)", filePath.c_str());
            } else {
                TraceLog(LOG_WARNING, "HOT RELOAD: Compile failed (%s)", filePath.c_str());
            }
        }
    }
    
    void Unload() { UnloadShader(shader); }
};

// // --------------------------------------------------------
// // PARAMETERS
// // --------------------------------------------------------
// struct EyeParams {
//     // Eye Shape
//     float eyeShapeID = 0.0f; float bend = 0.0f; float eyeThickness = 4.0f;
//     float eyeSide = 0.0f; float scaleX = 1.0f; float scaleY = 1.0f;
//     float angle = 0.0f; float spacing = 612.0f; float squareness = 0.0f;

//     // FX
//     float stressLevel = 0.0f; float gloomLevel = 0.0f;
//     float spiralSpeed = -1.2f;

//     // Look
//     float lookX = 0.0f; float lookY = 62.50f;

//     // Brow
//     bool showBrow = false; bool useLowerBrow = false;
//     float eyebrowThickness = 4.0f; float eyebrowLength = 1.0f;
//     float eyebrowSpacing = 0.0f; float eyebrowX = 0.0f; float eyebrowY = 0.0f;
//     float browScale = 1.0f; float browSide = 1.0f; float browAngle = 0.0f;
//     float browBend = 0.0f; float browBendOffset = 0.85f;

//     // Tears/Blush
//     bool showTears = false; bool showBlush = false;
//     float tearsLevel = 0.0f; int blushMode = 0; float blushScale = 1.0f;
//     float blushX = 0.0f; float blushY = 0.0f; float blushSpacing = 0.0f;

//     // Global
//     float pixelation = 1.0f; 
// };

// struct MouthParams {
//     float open = 0.05f; float width = 0.5f; float curve = 0.0f; float mouthAngle = 0.0f; 
//     float squeezeTop = 0.0f; float squeezeBottom = 0.0f; 
//     float teethY = 0.0f; float tongueUp = 0.0f; float tongueX = 0.0f;
//     float tongueWidth = 0.65f; float asymmetry = 0.0f; float squareness = 0.0f;
//     float teethWidth = 0.50f; float teethGap = 45.0f; float scale = 1.0f; float outlineThickness = 1.5f;
//     float sigma = 0.45f; float power = 6.0f; float maxLiftValue = 0.55f;
//     float lookX = 0.0f; float lookY = 0.0f; float stressLines = 0.0f; bool showInnerMouth =true; bool isThreeShape; bool isDShape; bool isSlashShape;
// };

// --------------------------------------------------------
// FACE SYSTEM (UNIFIED)
// --------------------------------------------------------
struct FaceSystem {
    // --- Resources ---
    ShaderAsset shEye;
    ShaderAsset shBrow;
    ShaderAsset shTears;
    ShaderAsset shMouth;
    ShaderAsset shPixel;
    RenderTexture2D canvas;
    Font mouthFont;

    // --- Colors ---
    const Color COL_STAR   = { 255, 184, 0, 255 };
    const Color COL_HEART  = { 220, 1, 1, 255 };
    const Color COL_BLUSH  = { 255, 105, 180, 150 };
    const Color COL_TEAR   = { 100, 180, 255, 220 };
    Color colMouthBg       = { 57, 99, 55, 255 };
    Color colMouthLine     = { 20, 35, 20, 255 };
    Color colMouthTeeth    = { 245, 245, 245, 255 };
    Color colMouthTongue   = { 162, 178, 106, 255 }; 

    // --- Eye Physics ---
    Spring sScaleX = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
    Spring sScaleY = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
    Spring sLookX  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};
    Spring sLookY  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};
    float blinkTimer = 0.0f;
    float nextBlinkTime = 3.0f;
    int blinkPhase = 0; // 0=Open, 1=Closing, 2=Closed, 3=Opening

    // --- Mouth Physics & Geometry ---
    MouthParams mCurrent, mVelocity; // Target is passed in Draw
    std::vector<Vector2> mouthCtrlPts; 
    std::vector<Vector2> mouthContour; 
    std::vector<Vector2> topTeethPoly, botTeethPoly, tonguePoly;
    const float MOUTH_GEO_SIZE = 1024.0f;
    const float MOUTH_SS = 4.0f; 

    // --- System State ---
    bool usePhysics = true;
    bool debugBoxes = false;
    float hotReloadTimer = 0.0f;
    float currentGlobalScale = 1.0f;

    // ----------------------------------------------------
    // INIT
    // ----------------------------------------------------
    void Init() {
        // Load Shaders
        shEye.Load(Config::SHADER_EYE, Config::SHADER_DIR);     RefreshLocEye();
        shBrow.Load(Config::SHADER_BROW, Config::SHADER_DIR);   RefreshLocBrow();
        shTears.Load(Config::SHADER_TEAR, Config::SHADER_DIR);  RefreshLocTear();
        shPixel.Load(Config::SHADER_PIXELIZER, Config::SHADER_DIR); RefreshLocPixel();
        shMouth.Load(Config::SHADER_MOUTH, Config::SHADER_DIR); RefreshLocMouth();

        // Canvas
        canvas = LoadRenderTexture(GetScreenHeight(), GetScreenHeight());
        mouthFont = LoadFontEx("assets/Roboto-Regular.ttf",512,NULL,0);
        SetTextureFilter(mouthFont.texture, TEXTURE_FILTER_TRILINEAR);

        // Initialize Mouth Lists
        mouthCtrlPts.resize(16);
        mouthContour.reserve(512);
        topTeethPoly.reserve(64);
        botTeethPoly.reserve(64);
        tonguePoly.reserve(64);

        // Initialize Mouth Physics Defaults
        mCurrent = {};
        mCurrent.open = 0.05f; mCurrent.width = 0.5f; mCurrent.curve = 0.2f; mCurrent.scale = 1.0f;
        mCurrent.teethWidth = 0.5f; mCurrent.teethGap = 45.0f; mCurrent.tongueWidth = 0.65f;
        mCurrent.teethY = -1.0f; mCurrent.sigma = 0.45f; mCurrent.power = 6.0f; mCurrent.maxLiftValue = 0.55f;
        mCurrent.outlineThickness = 1.5f;
    }

    void Unload() {
        shEye.Unload(); shBrow.Unload(); shTears.Unload(); shPixel.Unload(); shMouth.Unload();
        UnloadRenderTexture(canvas);
        UnloadFont(mouthFont);
    }

    // ----------------------------------------------------
    // UPDATE
    // ----------------------------------------------------
    void Update(float dt, const EyeParams& eParams, const MouthParams& mParams) {
        currentGlobalScale = GlobalScaler.scale;

        // Hot Reload
        hotReloadTimer += dt;
        if (hotReloadTimer > 1.0f) { CheckHotReload(); hotReloadTimer = 0.0f; }

        // --- EYE PHYSICS ---
        if (!usePhysics) {
            sScaleX.val = eParams.scaleX; sScaleY.val = eParams.scaleY;
            sLookX.val = eParams.lookX; sLookY.val = eParams.lookY;
            blinkPhase = 0;
            mCurrent = mParams; // Snap mouth
            GenerateMouthGeometry();
        } else {
            UpdateEyesPhysics(dt, eParams);
            UpdateMouthPhysics(dt, mParams);
        }
    }

    void UpdateEyesPhysics(float dt, const EyeParams& p) {
        // Blink Logic
        bool canBlink = (p.eyeShapeID < 0.5f || p.eyeShapeID > 3.5f);
        if (canBlink) {
            blinkTimer += dt;
            if (blinkTimer > nextBlinkTime && blinkPhase == 0) {
                blinkPhase = 1; blinkTimer = 0.0f;
                nextBlinkTime = (float)GetRandomValue((int)(Config::BLINK_MIN_DELAY * 10), (int)(Config::BLINK_MAX_DELAY * 10)) / 10.0f;
            }
        } else { blinkPhase = 0; }

        switch (blinkPhase) {
            case 1: // Closing
                sScaleY.target = 0.0f;
                if (sScaleY.val < Config::BLINK_CLOSE_THRESHOLD) { blinkPhase = 2; blinkTimer = 0.0f; }
                break;
            case 2: // Closed
                sScaleY.val = 1.0f; sScaleY.target = 1.0f; 
                if (blinkTimer > Config::BLINK_HOLD_DURATION) { blinkPhase = 3; sScaleY.val = 0.0f; }
                break;
            case 3: // Opening
                sScaleY.target = p.scaleY;
                if (sScaleY.val > p.scaleY * Config::BLINK_OPEN_THRESHOLD) { blinkPhase = 0; }
                break;
        }

        // Breathing
        if (blinkPhase == 0) {
             float breath = (p.scaleY > 0.8f) ? sinf((float)GetTime() * 2.0f) * 0.02f : 0.0f;
             sScaleY.target = p.scaleY + breath;
             sScaleX.target = p.scaleX - breath;
        }
        
        sLookX.target = p.lookX; sLookY.target = p.lookY;
        sScaleX.Update(dt); sScaleY.Update(dt); sLookX.Update(dt); sLookY.Update(dt);
    }

    void UpdateMouthPhysics(float dt, const MouthParams& target) {
        // Clamping strict dt for mouth physics stability
        float step = (dt > 0.05f) ? 0.05f : dt;
        
        const float STIFFNESS = 180.0f;
        const float DAMPING   = 14.0f;    
        auto Upd = [&](float& c, float& v, float t) {
            float f = STIFFNESS * (t - c);
            float d = DAMPING * v;
            v += (f - d) * step; 
            c += v * step;
        };

        // Update all mouth params
        Upd(mCurrent.open, mVelocity.open, Clamp(target.open, 0.0f, 1.2f));
        Upd(mCurrent.width, mVelocity.width, target.width);
        Upd(mCurrent.curve, mVelocity.curve, target.curve);
        Upd(mCurrent.squeezeTop, mVelocity.squeezeTop, target.squeezeTop);
        Upd(mCurrent.squeezeBottom, mVelocity.squeezeBottom, target.squeezeBottom);
        Upd(mCurrent.sigma, mVelocity.sigma, target.sigma);
        Upd(mCurrent.power, mVelocity.power, target.power);
        Upd(mCurrent.maxLiftValue, mVelocity.maxLiftValue, target.maxLiftValue);
        Upd(mCurrent.teethY, mVelocity.teethY, target.teethY);
        Upd(mCurrent.tongueUp, mVelocity.tongueUp, target.tongueUp);
        Upd(mCurrent.tongueX, mVelocity.tongueX, target.tongueX);
        Upd(mCurrent.tongueWidth, mVelocity.tongueWidth, target.tongueWidth);
        Upd(mCurrent.asymmetry, mVelocity.asymmetry, target.asymmetry);
        Upd(mCurrent.squareness, mVelocity.squareness, target.squareness);
        Upd(mCurrent.teethWidth, mVelocity.teethWidth, target.teethWidth);
        Upd(mCurrent.teethGap,   mVelocity.teethGap,   target.teethGap);
        Upd(mCurrent.scale,     mVelocity.scale,     Clamp(target.scale, 0.5f, 4.0f));
        Upd(mCurrent.outlineThickness, mVelocity.outlineThickness, target.outlineThickness);
        Upd(mCurrent.lookX, mVelocity.lookX, target.lookX); 
        Upd(mCurrent.lookY, mVelocity.lookY, target.lookY);
        Upd(mCurrent.stressLines, mVelocity.stressLines, target.stressLines);
        mCurrent.showInnerMouth = target.showInnerMouth;
        mCurrent.isThreeShape   = target.isThreeShape;   
        mCurrent.isDShape       = target.isDShape;       
        mCurrent.isSlashShape   = target.isSlashShape;
        GenerateMouthGeometry();
    }

    void GenerateMouthGeometry() {
        float cx = MOUTH_GEO_SIZE * 0.5f; 
        float cy = MOUTH_GEO_SIZE * 0.5f;
        float baseRadius = 40.0f * MOUTH_SS; 
        float w = baseRadius * (0.5f + mCurrent.width);
        float h = (mCurrent.open < 0.08f) ? 0.0f : (baseRadius * (0.2f + mCurrent.open * 1.5f));

        for (int i = 0; i < 16; i++) {
            float t = (float)i / 16.0f;
            float angle = t * PI * 2.0f + PI; 
            
            // --- PRE-CALCULATE SINE/COSINE ---
            float rawCos = cosf(angle);
            float rawSin = sinf(angle);
            
            // --- CHANGE 2: APPLY SQUARENESS TO WIDTH (X) ---
            // Just like you did for Y, we power the X component to push it to the edges.
            float sqPower = 1.0f - (mCurrent.squareness * 0.75f);
            
            float xWave = std::abs(rawCos);
            float shapedCos = std::pow(xWave, sqPower);
            float xSign = (rawCos >= 0.0f) ? 1.0f : -1.0f;
            float x = shapedCos * w * xSign; // New Squared X
            
            bool isTop = (i <= 8); 
            
            float flatness = 0.0f;
            if (mCurrent.asymmetry > 0.0f && isTop) flatness = mCurrent.asymmetry; 
            if (mCurrent.asymmetry < 0.0f && !isTop) flatness = -mCurrent.asymmetry;
            
            float effectiveH = h;
            float wave = std::abs(rawSin);
            if (flatness > 0.01f) {
                effectiveH *= (1.0f - flatness * 0.6f); 
                wave = std::pow(wave, 1.0f - (flatness * 0.5f)); 
            }
        
            // Your existing Y Squareness logic
            float shapedSin = std::pow(wave, sqPower);
            float sign = (rawSin >= 0.0f) ? 1.0f : -1.0f;
            float y = shapedSin * effectiveH * sign;
            
            float bendFactor = 15.0f * MOUTH_SS; 
            float normalizedX = x / (w + 1e-6f);
            float rawBend = 0.0f;
            // Adjust curve power slightly so it doesn't deform the box too much
            rawBend = (normalizedX * normalizedX) * bendFactor * mCurrent.curve;
            
            float bendMult = (flatness > 0.0f) ? (1.0f - flatness) : 1.0f;
            y -= rawBend * bendMult; 
        
            float activeSqueeze = isTop ? mCurrent.squeezeTop : mCurrent.squeezeBottom;
            float tX = Clamp(std::abs(x) / (w + 1e-6f), 0.0f, 1.0f);
            float u = tX / mCurrent.sigma;
            float influence = expf(-powf(u, mCurrent.power));
            float maxLift = (h * mCurrent.maxLiftValue);
            float archSign = (rawSin >= 0.0f) ? 1.0f : -1.0f;
            y -= activeSqueeze * influence * maxLift * archSign;
        
            mouthCtrlPts[i] = { cx + x, cy + y };
        }   

        mouthContour.clear();

        //Adding squared mouth stuff lol 
        //if(mCurrent.open<0.01 && mCurrent)
        for (int i = 0; i < 16; i++) {
            Vector2 p0 = mouthCtrlPts[(i-1+16)%16];
            Vector2 p1 = mouthCtrlPts[i];
            Vector2 p2 = mouthCtrlPts[(i+1)%16];
            Vector2 p3 = mouthCtrlPts[(i+2)%16];
            for (int k = 0; k < 16; k++) { 
                float t = (float)k/16.0f;
                Vector2 finalPos = MathUtils::CatmullRom(p0, p1, p2, p3, t);
                mouthContour.push_back(finalPos);
            }
        }
        MathUtils::RemoveNearDuplicates(mouthContour);
        MathUtils::NormalizeContourForTeeth(mouthContour);

        topTeethPoly.clear(); botTeethPoly.clear(); tonguePoly.clear();
        
        if (mCurrent.showInnerMouth && mCurrent.open > 0.10f && !mouthContour.empty()) {
            Rectangle bounds = MathUtils::GetBoundingBox(mouthContour);
            
            float shiftX = mCurrent.tongueX * (w * 0.4f);
            float tongueCX = cx + shiftX;
            float tongueW = bounds.width * 0.5f * mCurrent.tongueWidth;
            float tongueTip = Lerp(bounds.y + bounds.height, bounds.y, mCurrent.tongueUp);
            float tongueBase = bounds.y + bounds.height;
            
            for (int i = 0; i <= 32; i++) {
                float t = (float)i / 32.0f; 
                float tx = -cosf(t * PI); float ty = sinf(t * PI);  
                tonguePoly.push_back({tongueCX + (tx * tongueW), Lerp(tongueBase, tongueTip, ty)});
            }
            tonguePoly.push_back({tongueCX + tongueW, tongueBase + 10});
            tonguePoly.push_back({tongueCX - tongueW, tongueBase + 10});

            float teethW = bounds.width * mCurrent.teethWidth * 0.5f;
            float gap = mCurrent.teethGap * MOUTH_SS;
            float midY = cy + (mCurrent.teethY * 20.0f * MOUTH_SS);
            float topTeethEdge = midY - gap/2;
            float botTeethEdge = midY + gap/2;

            if(gap != 400 && topTeethEdge > bounds.y + 5.0f) {
                float tw = (mCurrent.teethWidth >= 0.95f) ? bounds.width * 0.6f : teethW;
                float tx_min = (mCurrent.teethWidth >= 0.95f) ? bounds.x : cx - tw;
                float tx_max = (mCurrent.teethWidth >= 0.95f) ? bounds.x + bounds.width : cx + tw;
                topTeethPoly = { {tx_min, bounds.y}, {tx_max, bounds.y}, {tx_max, topTeethEdge}, {tx_min, topTeethEdge} };
            }
            if(gap != 400 && botTeethEdge < bounds.y + bounds.height - 5.0f) {
                float tw = (mCurrent.teethWidth >= 0.95f) ? bounds.width * 0.6f : teethW;
                float tx_min = (mCurrent.teethWidth >= 0.95f) ? bounds.x : cx - tw;
                float tx_max = (mCurrent.teethWidth >= 0.95f) ? bounds.x + bounds.width : cx + tw;
                botTeethPoly = { {tx_min, botTeethEdge}, {tx_max, botTeethEdge}, {tx_max, bounds.y + bounds.height}, {tx_min, bounds.y + bounds.height} };
            }
        }
    }

    // ----------------------------------------------------
    // DRAWING
    // ----------------------------------------------------
    void Draw(Vector2 eyeCenter, Vector2 mouthCenter, EyeParams eP, MouthParams mP,Color eyeColor) {
        // Resize Canvas if screen changes
        if (canvas.texture.width != GetScreenWidth() || canvas.texture.height != GetScreenHeight()) {
            UnloadRenderTexture(canvas);
            canvas = LoadRenderTexture(GetScreenWidth(), GetScreenHeight());
        }

        // --- RENDER TO CANVAS ---
        BeginTextureMode(canvas);
            ClearBackground(BLANK);
            
            // 1. Draw Eyes (Stacked)
            DrawEyesCombined(eyeCenter, eP, eyeColor);
            
            // 2. Draw Mouth
            // Mouth renders directly into the texture. We must handle its position manually
            // since it calculates geometry around a local origin (GEO_SIZE/2).
            rlSetTexture(0); // Ensure clean state
            rlPushMatrix();
                if (fabsf(mP.mouthAngle) > 0.01f) {
                    rlTranslatef(mouthCenter.x, mouthCenter.y, 0);
                    rlRotatef(mP.mouthAngle, 0, 0, 1);
                    rlTranslatef(-mouthCenter.x, -mouthCenter.y, 0);
                }
                else if (fabsf(eP.angle) > 0.01f) {
                    rlTranslatef(eyeCenter.x, eyeCenter.y, 0);
                    rlRotatef(eP.angle, 0, 0, 1);
                    rlTranslatef(-eyeCenter.x, -eyeCenter.y, 0);
                }
                DrawMouthInternal(mouthCenter);
            rlPopMatrix();

        EndTextureMode();

        // --- FINAL COMPOSITE ---
        BeginShaderMode(shPixel.shader);
            float pixelSize = eP.pixelation;
            float renderSize[2] = { (float)canvas.texture.width, (float)canvas.texture.height };
            SetShaderValue(shPixel.shader, shPixel.locs.locPixelSize, &pixelSize, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shPixel.shader, shPixel.locs.locRenderSize, renderSize, SHADER_UNIFORM_VEC2);

            Rectangle src = { 0, 0, (float)canvas.texture.width, -(float)canvas.texture.height };
            Rectangle dest = { 0, 0, (float)GetScreenWidth(), (float)GetScreenHeight() };
            DrawTexturePro(canvas.texture, src, dest, {0,0}, 0.0f, WHITE);
        EndShaderMode();
    }

    // --- Eye Sub-Routines ---
    void DrawEyesCombined(Vector2 centerPos, EyeParams p, Color c) {
        float scaledSize = GlobalScaler.S(Config::REF_EYE_SIZE);
        float scaledW = scaledSize * Config::EYE_SCALE_FACTOR;
        float scaledH = scaledSize * Config::EYE_SCALE_FACTOR;
        float scaledHeightOffset = GlobalScaler.S(Config::EYE_HEIGHT_OFFSET);
        float scaledSpacing = GlobalScaler.S(p.spacing);

        rlPushMatrix();
        if(fabsf(p.angle) > 0.01f) {
            rlTranslatef(centerPos.x, centerPos.y, 0);
            rlRotatef(p.angle, 0, 0, 1);
            rlTranslatef(-centerPos.x, -centerPos.y, 0);
        }   

        Rectangle leftRect = { centerPos.x - (scaledSpacing * 0.5f) - (scaledW * 0.5f), centerPos.y - scaledHeightOffset - (scaledH * 0.5f), scaledW, scaledH };
        Rectangle rightRect = { centerPos.x + (scaledSpacing * 0.5f) - (scaledW * 0.5f), centerPos.y - scaledHeightOffset - (scaledH * 0.5f), scaledW, scaledH };

        // Color Overrides
        bool isStar = (p.eyeShapeID > 3.5 && p.eyeShapeID < 4.5) || (p.eyeShapeID > 7.5 && p.eyeShapeID < 8.5);
        bool isHeart = (p.eyeShapeID > 4.5 && p.eyeShapeID < 5.5);
        if (isStar) c = COL_STAR; else if (isHeart) c = COL_HEART;

        EyeParams leftP = p; leftP.browSide = -1.0f; leftP.eyeSide = 1.0f;
        DrawSingleEyeStack(leftRect, leftP, c);

        EyeParams rightP = p; rightP.browSide = 1.0f; rightP.eyeSide = -1.0f;
        DrawSingleEyeStack(rightRect, rightP, c);
        rlPopMatrix();
    }

    void DrawSingleEyeStack(Rectangle rect, EyeParams p, Color c) {
        Vector2 originalRes = { rect.width, rect.height };
        
        // Physics Transform
        Rectangle eyeRect = rect;
        eyeRect.x += GlobalScaler.S(sLookX.val);
        eyeRect.y += GlobalScaler.S(sLookY.val);
        
        float oldW = eyeRect.width, oldH = eyeRect.height;
        eyeRect.width *= sScaleX.val; eyeRect.height *= sScaleY.val;
        eyeRect.x += (oldW - eyeRect.width) * 0.5f; eyeRect.y += (oldH - eyeRect.height) * 0.5f;

        if(p.eyeShapeID>=12.0f) eyeRect.height *= 2.0f; // Special handling for "::" eyes

        if(p.tearsLevel < 0.4f) {
            DrawEyeLayer(eyeRect, originalRes, p, c);
            DrawTearsAndBlush(eyeRect, p);
            DrawBrows(eyeRect, rect, p);
        } else {
            DrawTearsAndBlush(eyeRect, p);
            DrawEyeLayer(eyeRect, originalRes, p, c);
            DrawBrows(eyeRect, rect, p);
        }
    }

    void DrawEyeLayer(Rectangle rect, Vector2 originalRes, EyeParams p, Color color) {
        rlSetTexture(0);
        BeginShaderMode(shEye.shader);
        SetCommonUniforms(shEye.shader, shEye.locs.resolution, shEye.locs.time, shEye.locs.color, {0,0,originalRes.x, originalRes.y}, color);
        
        float scaledThick = GlobalScaler.S(p.eyeThickness);
        float finalShape = (blinkPhase == 2) ? 1.0f : p.eyeShapeID;
        SetShaderValue(shEye.shader, shEye.locs.shape, &finalShape, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.bend, &p.bend, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.thickness, &scaledThick, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.side, &p.eyeSide, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.spiral, &p.spiralSpeed, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.stress, &p.stressLevel, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.gloom, &p.gloomLevel, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.squareness, &p.squareness, SHADER_UNIFORM_FLOAT);
        DrawQuad(rect, color);
        EndShaderMode();
        if(debugBoxes) DrawRectangleLinesEx(rect, 1, RED);
    }

    void DrawBrows(Rectangle eyeRect, Rectangle originalRect, EyeParams p) {
        if (!p.showBrow) return;
        Rectangle browRect = originalRect;
        browRect.width  = originalRect.width * 2.0f * p.browScale * p.eyebrowLength; 
        browRect.height = originalRect.height * 0.5f * p.browScale;
        browRect.x = originalRect.x + GlobalScaler.S(sLookX.val);
        browRect.y = originalRect.y + GlobalScaler.S(sLookY.val) - (originalRect.height * 0.5f);
        browRect.x += (originalRect.width - browRect.width) * 0.5f;
        browRect.y += (originalRect.height - browRect.height) * 0.5f;
        browRect.x += GlobalScaler.S((p.eyebrowX * 20.0f) + (p.eyebrowSpacing * 20.0f * p.browSide));
        //browRect.x += GlobalScaler.S(p.eyebrowX * 20.0f) + (GlobalScaler.S(p.eyebrowSpacing * 20.0f) * p.browSide);
        browRect.y += GlobalScaler.S(p.eyebrowY * 20.0f);

        rlSetTexture(0);
        BeginShaderMode(shBrow.shader);
        SetCommonUniforms(shBrow.shader, shBrow.locs.resolution, -1, shBrow.locs.color, browRect, BLACK);
        float scaledThick = GlobalScaler.S(p.eyebrowThickness);
        SetShaderValue(shBrow.shader, shBrow.locs.bend, &p.browBend, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.thickness, &scaledThick, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.browLen, &p.eyebrowLength, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.side, &p.browSide, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.browAngle, &p.browAngle, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.browBendOffset, &p.browBendOffset, SHADER_UNIFORM_FLOAT);
        DrawQuad(browRect, BLACK);
        EndShaderMode();

        // Lower Brow
        if (p.useLowerBrow) {
             // Logic simplified for brevity but preserves original transform
            EyeParams lowerP = p; lowerP.eyebrowY = 0.0f; lowerP.bend = -p.bend;
            lowerP.eyebrowLength = 0.6f; lowerP.eyebrowThickness = 7.15f;
            Rectangle lowerRect = eyeRect;
            lowerRect.width  = eyeRect.width * 2.5f * p.browScale * lowerP.eyebrowLength;
            lowerRect.height = eyeRect.height * (1.0f + fabsf(p.bend));
            lowerRect.x = eyeRect.x + (eyeRect.width - lowerRect.width) * 0.5f;
            lowerRect.y = (eyeRect.y + eyeRect.height * 0.5f + eyeRect.height * 0.5f) - (lowerRect.height * 0.53f) - eyeRect.height * 0.35f;
            lowerRect.x += GlobalScaler.S(48.6235f);

            rlSetTexture(0);
            BeginShaderMode(shBrow.shader);
            SetCommonUniforms(shBrow.shader, shBrow.locs.resolution, -1, shBrow.locs.color, lowerRect, BLACK);
            float lThick = GlobalScaler.S(lowerP.eyebrowThickness);
            SetShaderValue(shBrow.shader, shBrow.locs.bend, &lowerP.browBend, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow.shader, shBrow.locs.thickness, &lThick, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow.shader, shBrow.locs.browLen, &lowerP.eyebrowLength, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow.shader, shBrow.locs.side, &p.browSide, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow.shader, shBrow.locs.browAngle, &p.browAngle, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow.shader, shBrow.locs.browBendOffset, &p.browBendOffset, SHADER_UNIFORM_FLOAT);
            DrawQuad(lowerRect, BLACK);
            EndShaderMode();
            if(debugBoxes) DrawRectangleLinesEx(lowerRect, 1, YELLOW);

        }
        if(debugBoxes) DrawRectangleLinesEx(browRect, 1, YELLOW);
    }

    void DrawTearsAndBlush(Rectangle eyeRect, EyeParams p) {
        if (!p.showTears && !p.showBlush) return;
        
        if (p.showTears) {
            float tearWidth = eyeRect.width + GlobalScaler.S(130.0f);
            Rectangle tearRect = { eyeRect.x + (eyeRect.width * 0.5f) - (tearWidth * 0.5f), eyeRect.y + (eyeRect.height * 0.5f) - 17.0f, tearWidth, (float)GetScreenHeight() };
            DrawTearLayer(tearRect, p, false);
        }
        if (p.showBlush) {
            float blushSize = eyeRect.width * 0.6f * p.blushScale;
            float cx = eyeRect.x + (eyeRect.width * 0.5f) + (GlobalScaler.S(150.0f) * -p.eyeSide) + GlobalScaler.S(p.blushX * 20.0f) + (GlobalScaler.S(p.blushSpacing * 10.0f) * -p.eyeSide);
            float cy = eyeRect.y + (eyeRect.height * 0.5f) + GlobalScaler.S(200.0f) + GlobalScaler.S(p.blushY * 20.0f);
            Rectangle blushRect = { cx - (blushSize * 0.5f), cy - (blushSize * 0.5f), blushSize, blushSize };
            DrawTearLayer(blushRect, p, true);
        }

    }

    void DrawTearLayer(Rectangle rect, EyeParams p, bool isBlush) {
        rlSetTexture(0);
        BeginShaderMode(shTears.shader);
        SetCommonUniforms(shTears.shader, shTears.locs.resolution, shTears.locs.time, -1, rect, WHITE);
        int blushToggle = isBlush ? 1 : 0;
        int showBlush = (isBlush && p.showBlush) ? 1 : (p.showBlush ? 1 : 0); // Logic preserve
        float pink[4] = { COL_BLUSH.r/255.0f, COL_BLUSH.g/255.0f, COL_BLUSH.b/255.0f, COL_BLUSH.a/255.0f };
        float blue[4] = { COL_TEAR.r/255.0f, COL_TEAR.g/255.0f, COL_TEAR.b/255.0f, COL_TEAR.a/255.0f };
        SetShaderValue(shTears.shader, shTears.locs.scale, &currentGlobalScale, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shTears.shader, shTears.locs.tearLevel, &p.tearsLevel, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shTears.shader, shTears.locs.blushMode, &p.blushMode, SHADER_UNIFORM_INT);
        SetShaderValue(shTears.shader, shTears.locs.side, &p.eyeSide, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shTears.shader, shTears.locs.showBlush, &showBlush, SHADER_UNIFORM_INT);
        SetShaderValue(shTears.shader, shTears.locs.blushCol, pink, SHADER_UNIFORM_VEC4);
        SetShaderValue(shTears.shader, shTears.locs.tearCol, blue, SHADER_UNIFORM_VEC4);
        DrawQuad(rect, WHITE);
        EndShaderMode();
        if(debugBoxes) DrawRectangleLinesEx(rect, 1, BLACK);

    }

    // --- Mouth Sub-Routines ---
    void DrawMouthInternal(Vector2 centerPos) {
        if (mCurrent.isThreeShape) {
            const char* text = "3"; // Or ":3" if you want the full kitty face
            // 1. Calculate Size
            // We scale the font size by your physics 'scale'
            // 'GlobalScaler.scale' ensures it resizes with the window
            float fontSize = 150.0f * mCurrent.scale * GlobalScaler.scale;
            float spacing = 2.0f;

            // 2. Measure text to center it perfectly
            Vector2 textSize = MeasureTextEx(mouthFont, text, fontSize, spacing);
            Vector2 textOrigin = { textSize.x * 0.5f, textSize.y * 0.5f };

            // 3. Offset Correction
            // Apply your 'lookX/Y' physics offsets
            Vector2 drawPos = {
                centerPos.x + GlobalScaler.S(mCurrent.lookX),
                centerPos.y + GlobalScaler.S(mCurrent.lookY)
            };

            // 4. Draw it!
            // Note: Rotation is 0.0f because you already apply rlRotatef 
            // in the parent Draw() function before calling this.
            DrawTextPro(
                mouthFont,
                text,
                drawPos,
                textOrigin, // Pivot point (center of text)
                0.0f,       // Rotation (handled by parent matrix)
                fontSize,
                spacing,
                BLACK // Use your existing mouth color!
            );

            // Return early so we don't draw the shader mouth on top
            return;
        }
        if (mCurrent.isDShape) {
            const char* text = "D";
            // 1. Calculate Size
            // We scale the font size by your physics 'scale'
            // 'GlobalScaler.scale' ensures it resizes with the window
            float fontSize = 150.0f * mCurrent.scale * GlobalScaler.scale;
            float spacing = 2.0f;

            // 2. Measure text to center it perfectly
            Vector2 textSize = MeasureTextEx(mouthFont, text, fontSize, spacing);
            Vector2 textOrigin = { textSize.x * 0.5f, textSize.y * 0.5f };

            // 3. Offset Correction
            // Apply your 'lookX/Y' physics offsets
           Vector2 drawPos = {
                roundf(centerPos.x + GlobalScaler.S(mCurrent.lookX)),
                roundf(centerPos.y + GlobalScaler.S(mCurrent.lookY))
            };
            // 4. Draw it!
            // Note: Rotation is 0.0f because you already apply rlRotatef 
            // in the parent Draw() function before calling this.
            DrawTextPro(
                mouthFont,
                text,
                drawPos,
                textOrigin, // Pivot point (center of text)
                90.0f,       // Rotation (handled by parent matrix)
                fontSize,
                spacing,
                BLACK // Use your existing mouth color!
            );

            // Return early so we don't draw the shader mouth on top
            return;
        }

        if (mCurrent.isSlashShape) {
            float baseWeight = 4.0f;
            float thickness = (baseWeight * mCurrent.outlineThickness) * mCurrent.scale * GlobalScaler.scale;
            float length = (100.0f + (mCurrent.width * 300.0f)) * mCurrent.scale * GlobalScaler.scale;

            Vector2 center = {
                roundf(centerPos.x + GlobalScaler.S(mCurrent.lookX)),
                roundf(centerPos.y + GlobalScaler.S(mCurrent.lookY))
            };
            Rectangle rect = { center.x, center.y, length, thickness };
            Vector2 origin = { length * 0.5f, thickness * 0.5f }; // Pivot center

            DrawRectanglePro(rect, origin, 0.0f, BLACK);

            return;
        }
        if (mouthContour.empty()) return;

        float basePadding = 20.0f;

        // 2. Extra padding for the stress lines (cheeks)
        // We scale this by mCurrent.scale so the box grows when the mouth grows.
        // 120.0f is the rough radius of the stress lines from the center.
        float stressPadding = 50.0f * mCurrent.scale; 

        // 3. Total dynamic padding
        const float paddingPx = basePadding + stressPadding;
        // [UNIFICATION] Scale factor now integrates GlobalScaler
        float unitScale = (256.0f / MOUTH_GEO_SIZE) * mCurrent.scale * GlobalScaler.scale; 
        if (unitScale < 0.0001f) unitScale = 0.0001f;

        float offsetX = GlobalScaler.S(mCurrent.lookX);
        float offsetY = GlobalScaler.S(mCurrent.lookY);

        Rectangle boundsPhys = MathUtils::GetBoundingBox(mouthContour);
        float paddingPhys = paddingPx / unitScale;
        boundsPhys.x -= paddingPhys; boundsPhys.y -= paddingPhys;
        boundsPhys.width += paddingPhys * 2.0f; boundsPhys.height += paddingPhys * 2.0f;

        // Origin logic from Mouth: Center of GEO_SIZE maps to centerPos
        float physRefLeft = centerPos.x - (MOUTH_GEO_SIZE * 0.5f * unitScale) + offsetX;
        float physRefTop  = centerPos.y - (MOUTH_GEO_SIZE * 0.5f * unitScale) + offsetY;
        
        Rectangle screenRect = {
            physRefLeft + (boundsPhys.x * unitScale),
            physRefTop  + (boundsPhys.y * unitScale),
            boundsPhys.width * unitScale,
            boundsPhys.height * unitScale
        };

        const int MAX_PTS = 128;
        float flatMouth[MAX_PTS * 2], flatTop[MAX_PTS * 2], flatBot[MAX_PTS * 2], flatTongue[MAX_PTS * 2];
        int cntMouth=0, cntTop=0, cntBot=0, cntTongue=0;
        Vector2 offset = {boundsPhys.x, boundsPhys.y};
        
        MathUtils::ResamplePoly(mouthContour, flatMouth, MAX_PTS, cntMouth, unitScale, offset, true);
        MathUtils::ResamplePoly(topTeethPoly, flatTop, MAX_PTS, cntTop, unitScale, offset, true);
        MathUtils::ResamplePoly(botTeethPoly, flatBot, MAX_PTS, cntBot, unitScale, offset, true);
        MathUtils::ResamplePoly(tonguePoly, flatTongue, MAX_PTS, cntTongue, unitScale, offset, true);

        Vector4 cBg, cTh, cTg; 

        if (mCurrent.showInnerMouth) {
            cBg = MathUtils::ColorNormalize(colMouthBg);
            cTh = MathUtils::ColorNormalize(colMouthTeeth);
            cTg = MathUtils::ColorNormalize(colMouthTongue);
        } else {
            cBg = { 0.0f, 0.0f, 0.0f, 0.0f }; // Transparent
            cTh = { 0.0f, 0.0f, 0.0f, 0.0f };
            cTg = { 0.0f, 0.0f, 0.0f, 0.0f };
        }
        
        Vector4 cLn = MathUtils::ColorNormalize(colMouthLine);

        BeginShaderMode(shMouth.shader);
            SetShaderValueV(shMouth.shader, shMouth.locs.mouthPts, flatMouth, SHADER_UNIFORM_VEC2, MAX_PTS);
            SetShaderValue(shMouth.shader, shMouth.locs.mouthCnt, &cntMouth, SHADER_UNIFORM_INT);
            SetShaderValueV(shMouth.shader, shMouth.locs.topPts, flatTop, SHADER_UNIFORM_VEC2, MAX_PTS);
            SetShaderValue(shMouth.shader, shMouth.locs.topCnt, &cntTop, SHADER_UNIFORM_INT);
            SetShaderValueV(shMouth.shader, shMouth.locs.botPts, flatBot, SHADER_UNIFORM_VEC2, MAX_PTS);
            SetShaderValue(shMouth.shader, shMouth.locs.botCnt, &cntBot, SHADER_UNIFORM_INT);
            SetShaderValueV(shMouth.shader, shMouth.locs.tonguePts, flatTongue, SHADER_UNIFORM_VEC2, MAX_PTS);
            SetShaderValue(shMouth.shader, shMouth.locs.tongueCnt, &cntTongue, SHADER_UNIFORM_INT);
            
            float res[2] = { screenRect.width, screenRect.height };
            SetShaderValue(shMouth.shader, shMouth.locs.resolution, res, SHADER_UNIFORM_VEC2);
            SetShaderValue(shMouth.shader, shMouth.locs.padding, &paddingPx, SHADER_UNIFORM_FLOAT);
            // Combined scale for outline thickness correction
            float combinedScale = mCurrent.scale * GlobalScaler.scale;
            SetShaderValue(shMouth.shader, shMouth.locs.scale, &combinedScale, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shMouth.shader, shMouth.locs.outlineThickness, &mCurrent.outlineThickness, SHADER_UNIFORM_FLOAT);
            
            SetShaderValue(shMouth.shader, shMouth.locs.colBg, &cBg, SHADER_UNIFORM_VEC4);
            SetShaderValue(shMouth.shader, shMouth.locs.colLine, &cLn, SHADER_UNIFORM_VEC4);
            SetShaderValue(shMouth.shader, shMouth.locs.colTeeth, &cTh, SHADER_UNIFORM_VEC4);
            SetShaderValue(shMouth.shader, shMouth.locs.colTongue, &cTg, SHADER_UNIFORM_VEC4);
            SetShaderValue(shMouth.shader, shMouth.locs.stressLines, &mCurrent.stressLines, SHADER_UNIFORM_FLOAT);

            DrawQuad(screenRect, WHITE);
        EndShaderMode();
        if(debugBoxes) DrawRectangleLinesEx(screenRect, 1, GREEN);
    }

    // --- Helpers ---
    void SetCommonUniforms(Shader shader, int locRes, int locTime, int locCol, Rectangle rect, Color color) {
        float res[2] = { rect.width, rect.height };
        float time = (float)GetTime();
        float colVec[4] = { color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f };
        if (locRes != -1) SetShaderValue(shader, locRes, res, SHADER_UNIFORM_VEC2);
        if (locTime != -1) SetShaderValue(shader, locTime, &time, SHADER_UNIFORM_FLOAT);
        if (locCol != -1) SetShaderValue(shader, locCol, colVec, SHADER_UNIFORM_VEC4);
    }

    void DrawQuad(Rectangle rect, Color color) {
        rlBegin(RL_QUADS);
            rlColor4ub(color.r, color.g, color.b, color.a);
            rlTexCoord2f(0.0f, 0.0f); rlVertex2f(rect.x, rect.y);
            rlTexCoord2f(0.0f, 1.0f); rlVertex2f(rect.x, rect.y + rect.height);
            rlTexCoord2f(1.0f, 1.0f); rlVertex2f(rect.x + rect.width, rect.y + rect.height);
            rlTexCoord2f(1.0f, 0.0f); rlVertex2f(rect.x + rect.width, rect.y);
        rlEnd();
    }

    void CheckHotReload() {
        long t1 = shEye.lastModTime; long t2 = shBrow.lastModTime; long t3 = shTears.lastModTime; long t4 = shMouth.lastModTime;
        shEye.ReloadIfChanged();   if (shEye.lastModTime > t1) RefreshLocEye();
        shBrow.ReloadIfChanged();  if (shBrow.lastModTime > t2) RefreshLocBrow();
        shTears.ReloadIfChanged(); if (shTears.lastModTime > t3) RefreshLocTear();
        shMouth.ReloadIfChanged(); if (shMouth.lastModTime > t4) RefreshLocMouth();
        shPixel.ReloadIfChanged(); 
    }

    // --- Refresh Locations ---
    void RefreshLocEye() {
        shEye.locs.resolution = GetShaderLocation(shEye.shader, "uResolution");
        shEye.locs.time = GetShaderLocation(shEye.shader, "uTime");
        shEye.locs.color = GetShaderLocation(shEye.shader, "uColor");
        shEye.locs.shape = GetShaderLocation(shEye.shader, "uShapeID");
        shEye.locs.bend = GetShaderLocation(shEye.shader, "uBend");
        shEye.locs.thickness = GetShaderLocation(shEye.shader, "uThickness");
        shEye.locs.side = GetShaderLocation(shEye.shader, "uEyeSide");
        shEye.locs.spiral = GetShaderLocation(shEye.shader, "uSpiralSpeed");
        shEye.locs.stress = GetShaderLocation(shEye.shader, "uStressLevel");
        shEye.locs.gloom = GetShaderLocation(shEye.shader, "uGloomLevel");
        shEye.locs.squareness = GetShaderLocation(shEye.shader, "uSquareness");
    }
    void RefreshLocBrow() {
        shBrow.locs.resolution = GetShaderLocation(shBrow.shader, "uResolution");
        shBrow.locs.color = GetShaderLocation(shBrow.shader, "uColor");
        shBrow.locs.bend = GetShaderLocation(shBrow.shader, "uBend");
        shBrow.locs.thickness = GetShaderLocation(shBrow.shader, "uThickness");
        shBrow.locs.browLen = GetShaderLocation(shBrow.shader, "uEyeBrowLength");
        shBrow.locs.side = GetShaderLocation(shBrow.shader, "uBrowSide");
        shBrow.locs.browAngle = GetShaderLocation(shBrow.shader, "uAngle");
        shBrow.locs.browBendOffset = GetShaderLocation(shBrow.shader, "uBendOffset");
    }
    void RefreshLocTear() {
        shTears.locs.resolution = GetShaderLocation(shTears.shader, "uResolution");
        shTears.locs.scale = GetShaderLocation(shTears.shader, "uScale");
        shTears.locs.time = GetShaderLocation(shTears.shader, "uTime");
        shTears.locs.tearLevel = GetShaderLocation(shTears.shader, "uTearsLevel");
        shTears.locs.blushMode = GetShaderLocation(shTears.shader, "uBlushMode");
        shTears.locs.showBlush = GetShaderLocation(shTears.shader, "uShowBlush");
        shTears.locs.blushCol = GetShaderLocation(shTears.shader, "uBlushColor");
        shTears.locs.tearCol = GetShaderLocation(shTears.shader, "uTearColor");
        shTears.locs.side = GetShaderLocation(shTears.shader, "uSide");
    }
    void RefreshLocMouth() {
        shMouth.locs.mouthPts = GetShaderLocation(shMouth.shader, "uMouthPts");
        shMouth.locs.mouthCnt = GetShaderLocation(shMouth.shader, "uMouthCount");
        shMouth.locs.topPts = GetShaderLocation(shMouth.shader, "uTopTeethPts");
        shMouth.locs.topCnt = GetShaderLocation(shMouth.shader, "uTopTeethCount");
        shMouth.locs.botPts = GetShaderLocation(shMouth.shader, "uBotTeethPts");
        shMouth.locs.botCnt = GetShaderLocation(shMouth.shader, "uBotTeethCount");
        shMouth.locs.tonguePts = GetShaderLocation(shMouth.shader, "uTonguePts");
        shMouth.locs.tongueCnt = GetShaderLocation(shMouth.shader, "uTongueCount");
        shMouth.locs.resolution = GetShaderLocation(shMouth.shader, "uResolution");
        shMouth.locs.padding = GetShaderLocation(shMouth.shader, "uPadding");
        shMouth.locs.scale = GetShaderLocation(shMouth.shader, "uScale");
        shMouth.locs.outlineThickness = GetShaderLocation(shMouth.shader, "uOutlineThickness");
        shMouth.locs.colBg = GetShaderLocation(shMouth.shader, "uColBg");
        shMouth.locs.colLine = GetShaderLocation(shMouth.shader, "uColLine");
        shMouth.locs.colTeeth = GetShaderLocation(shMouth.shader, "uColTeeth");
        shMouth.locs.colTongue = GetShaderLocation(shMouth.shader, "uColTongue");
        shMouth.locs.stressLines = GetShaderLocation(shMouth.shader, "uStressLines");
    }
    void RefreshLocPixel() {
        shPixel.locs.locPixelSize = GetShaderLocation(shPixel.shader, "pixelSize");
        shPixel.locs.locRenderSize = GetShaderLocation(shPixel.shader, "renderSize");
    }
};