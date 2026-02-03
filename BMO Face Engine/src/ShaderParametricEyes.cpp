// ShaderParametricEyes.cpp
// REFINED: Multi-Shader Architecture (Eye, Brow, Tears)
// Preserves Physics, Scaling, and Color Overrides

#include "raylib.h"
#include "rlgl.h"
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "utility.h" // Access to GlobalScaler

// --------------------------------------------------------
// CONSTANTS & CONFIG
// --------------------------------------------------------
namespace Config {
    // Shader Filenames
    const char* SHADER_EYE   = "eyes_es.fs";
    const char* SHADER_BROW  = "brow_es.fs";
    const char* SHADER_TEAR  = "tears_es.fs";
    const char* SHADER_DIR   = "src/";

    // Animation & Physics
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
// EYE PARAMETERS
// --------------------------------------------------------
struct EyeParams {
    // --- Main Eye ---
    float eyeShapeID = 0.0f;     // 0-8=Std, 9=Kawaii, 10=Shocked
    float bend = 0.0f;
    float eyeThickness = 4.0f;
    float eyeSide = 0.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float spacing = 612.0f;

    // --- Eye Surface FX ---
    float stressLevel = 0.0f;    // 0-1: Angry lines
    float gloomLevel = 0.0f;     // 0-1: Shocked vertical lines
    int distortMode = 0;         // 1 = Squash/Stretch distortion
    float spiralSpeed = -1.2f;

    // --- Look Targets ---
    float lookX = 0.0f;
    float lookY = 62.50f;

    // --- Eyebrow ---
    bool showBrow = false;
    bool useLowerBrow = false;
    float eyebrowType = 0.0f;
    float eyebrowThickness = 4.0f;
    float eyebrowLength = 1.0f;
    float eyebrowSpacing = 0.0f;    
    float eyebrowX = 0.0f;
    float eyebrowY = 0.0f;
    float browScale = 1.0f;
    float browSide = 1.0f;       // 1.0 = left, -1.0 = right
    float browAngle = 0.0f;
    float browBend = 0.0f;
    float browBendOffset = 0.85f;

    // --- Tears & Blush ---
    bool showTears = false;
    bool showBlush = false;
    float tearsLevel = 0.0f;
    int blushMode = 0;
    float blushScale = 1.0f;
    float blushX = 0.0f;
    float blushY = 0.0f;
    float blushSpacing = 0.0f;
};

// --------------------------------------------------------
// SHADER WRAPPER (Hot Reload Support)
// --------------------------------------------------------
struct ShaderAsset {
    Shader shader;
    std::string filePath;
    long lastModTime;
    
    // Cached Uniform Locations
    struct Locations {
        int resolution, time, color;
        // Layer-specific uniforms stored in a generic way or specific map could be used,
        // but keeping it flat for simplicity in C++ struct mirroring.
        // Eye
        int shape, bend, thickness, side, spiral, distort, stress, gloom;
        // Brow
        int browType, browLen, browY, browAngle, browBendOffset;
        // Tears
        int scale, tearLevel, blushMode, showBlush, blushCol, tearCol;
    } locs;

    void Load(const char* filename, const char* dir) {
        // Resolve path
        if (FileExists(filename)) filePath = filename;
        else if (FileExists((std::string(dir) + filename).c_str())) filePath = std::string(dir) + filename;
        else filePath = filename; // Fallback

        shader = LoadShader(0, filePath.c_str());
        lastModTime = GetFileModTime(filePath.c_str());
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
    
    void Unload() {
        UnloadShader(shader);
    }
};

// --------------------------------------------------------
// MAIN CLASS
// --------------------------------------------------------
struct ParametricEyes {
    // Shaders
    ShaderAsset shEye;
    ShaderAsset shBrow;
    ShaderAsset shTears;

    // Colors
    const Color COL_STAR   = { 255, 184, 0, 255 };
    const Color COL_HEART  = { 220, 1, 1, 255 };
    const Color COL_BLUSH  = { 255, 105, 180, 150 };
    const Color COL_TEAR   = { 100, 180, 255, 220 };

    // Physics Springs
    Spring sScaleX = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
    Spring sScaleY = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
    Spring sLookX  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};
    Spring sLookY  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};

    // Blink State
    float blinkTimer = 0.0f;
    float nextBlinkTime = 3.0f;
    int blinkPhase = 0; // 0=Open, 1=Closing, 2=Closed, 3=Opening
    bool usePhysics = true;

    // Debug & Util
    float hotReloadTimer = 0.0f;
    float currentGlobalScale = 1.0f;
    bool debugBoxes = false;

    // ----------------------------------------------------
    // INIT
    // ----------------------------------------------------
    void Init() {
        // Load Shaders
        shEye.Load(Config::SHADER_EYE, Config::SHADER_DIR);
        RefreshLocationsEye();

        shBrow.Load(Config::SHADER_BROW, Config::SHADER_DIR);
        RefreshLocationsBrow();

        shTears.Load(Config::SHADER_TEAR, Config::SHADER_DIR);
        RefreshLocationsTear();
    }

    void Unload() {
        shEye.Unload();
        shBrow.Unload();
        shTears.Unload();
    }

    // ----------------------------------------------------
    // UPDATE
    // ----------------------------------------------------
    void Update(float dt, EyeParams& params) {
        currentGlobalScale = GlobalScaler.scale;

        // Hot Reload Check
        hotReloadTimer += dt;
        if (hotReloadTimer > 1.0f) {
            CheckHotReload();
            hotReloadTimer = 0.0f;
        }

        if (!usePhysics) {
            sScaleX.val = params.scaleX;
            sScaleY.val = params.scaleY;
            sLookX.val = params.lookX;
            sLookY.val = params.lookY;
            blinkPhase = 0;
            return;
        }

        UpdateBlinkLogic(dt, params);
        
        // Physics breathing
        if (blinkPhase == 0) {
             float breath = (params.scaleY > 0.8f) ? sinf((float)GetTime() * 2.0f) * 0.02f : 0.0f;
             sScaleY.target = params.scaleY + breath;
             sScaleX.target = params.scaleX - breath;
        }
        
        sLookX.target = params.lookX;
        sLookY.target = params.lookY;

        sScaleX.Update(dt); sScaleY.Update(dt);
        sLookX.Update(dt);  sLookY.Update(dt);
    }

    void UpdateBlinkLogic(float dt, const EyeParams& params) {
        // Auto Blink only for standard shapes
        bool canBlink = (params.eyeShapeID < 0.5f || params.eyeShapeID > 3.5f);

        if (canBlink) {
            blinkTimer += dt;
            if (blinkTimer > nextBlinkTime && blinkPhase == 0) {
                blinkPhase = 1;
                blinkTimer = 0.0f;
                nextBlinkTime = (float)GetRandomValue((int)(Config::BLINK_MIN_DELAY * 10), (int)(Config::BLINK_MAX_DELAY * 10)) / 10.0f;
            }
        } else {
            blinkPhase = 0;
        }

        // Blink State Machine
        switch (blinkPhase) {
            case 1: // Closing
                sScaleY.target = 0.0f;
                if (sScaleY.val < Config::BLINK_CLOSE_THRESHOLD) { 
                    blinkPhase = 2; 
                    blinkTimer = 0.0f; 
                }
                break;
            case 2: // Closed (Hold)
                sScaleY.val = 1.0f; sScaleY.target = 1.0f; // Force line thickness
                if (blinkTimer > Config::BLINK_HOLD_DURATION) { 
                    blinkPhase = 3; 
                    sScaleY.val = 0.0f; 
                }
                break;
            case 3: // Opening
                sScaleY.target = params.scaleY;
                if (sScaleY.val > params.scaleY * Config::BLINK_OPEN_THRESHOLD) {
                    blinkPhase = 0; 
                }
                break;
        }
    }

    // ----------------------------------------------------
    // DRAWING HELPERS
    // ----------------------------------------------------
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

    void DrawDebugRect(Rectangle rect, Color col) {
        if (!debugBoxes) return;
        DrawRectangleLinesEx(rect, 2.0f, col);
        Color fill = col; fill.a = 40;
        DrawRectangleRec(rect, fill);
    }

    // ----------------------------------------------------
    // LAYER DRAWING
    // ----------------------------------------------------
    void DrawEyeLayer(Rectangle rect, Vector2 originalRes, EyeParams p, Color color) {
        rlSetTexture(0);
        BeginShaderMode(shEye.shader);
        
        // Uniforms
        SetCommonUniforms(shEye.shader, shEye.locs.resolution, shEye.locs.time, shEye.locs.color, {0,0,originalRes.x, originalRes.y}, color);

        float scaledThick = GlobalScaler.S(p.eyeThickness);
        float finalShape = (blinkPhase == 2) ? 1.0f : p.eyeShapeID;

        SetShaderValue(shEye.shader, shEye.locs.shape, &finalShape, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.bend, &p.bend, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.thickness, &scaledThick, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.side, &p.eyeSide, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.spiral, &p.spiralSpeed, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.distort, &p.distortMode, SHADER_UNIFORM_INT);
        SetShaderValue(shEye.shader, shEye.locs.stress, &p.stressLevel, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shEye.shader, shEye.locs.gloom, &p.gloomLevel, SHADER_UNIFORM_FLOAT);

        DrawQuad(rect, color);
        EndShaderMode();
        DrawDebugRect(rect, RED);
    }

    void DrawBrowLayer(Rectangle rect, EyeParams p, Color color) {
        rlSetTexture(0);
        BeginShaderMode(shBrow.shader);

        SetCommonUniforms(shBrow.shader, shBrow.locs.resolution, -1, shBrow.locs.color, rect, color);

        float scaledThick = GlobalScaler.S(p.eyebrowThickness);
        
        SetShaderValue(shBrow.shader, shBrow.locs.browType, &p.eyebrowType, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.bend, &p.browBend, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.thickness, &scaledThick, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.browLen, &p.eyebrowLength, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.side, &p.browSide, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.browAngle, &p.browAngle, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shBrow.shader, shBrow.locs.browBendOffset, &p.browBendOffset, SHADER_UNIFORM_FLOAT);

        DrawQuad(rect, color);
        EndShaderMode();
        DrawDebugRect(rect, YELLOW);
    }

    void DrawTearsLayer(Rectangle rect, EyeParams p) {
        rlSetTexture(0);
        BeginShaderMode(shTears.shader);

        SetCommonUniforms(shTears.shader, shTears.locs.resolution, shTears.locs.time, -1, rect, WHITE);

        int blushToggle = p.showBlush ? 1 : 0;
        float pink[4] = { COL_BLUSH.r/255.0f, COL_BLUSH.g/255.0f, COL_BLUSH.b/255.0f, COL_BLUSH.a/255.0f };
        float blue[4] = { COL_TEAR.r/255.0f, COL_TEAR.g/255.0f, COL_TEAR.b/255.0f, COL_TEAR.a/255.0f };

        SetShaderValue(shTears.shader, shTears.locs.scale, &currentGlobalScale, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shTears.shader, shTears.locs.tearLevel, &p.tearsLevel, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shTears.shader, shTears.locs.blushMode, &p.blushMode, SHADER_UNIFORM_INT);
        SetShaderValue(shTears.shader, shTears.locs.side, &p.eyeSide, SHADER_UNIFORM_FLOAT);
        SetShaderValue(shTears.shader, shTears.locs.showBlush, &blushToggle, SHADER_UNIFORM_INT);
        SetShaderValue(shTears.shader, shTears.locs.blushCol, pink, SHADER_UNIFORM_VEC4);
        SetShaderValue(shTears.shader, shTears.locs.tearCol, blue, SHADER_UNIFORM_VEC4);

        DrawQuad(rect, WHITE);
        EndShaderMode();
        
        if (p.showBlush && !p.showTears) DrawDebugRect(rect, { 255, 105, 180, 150 });
        else DrawDebugRect(rect, { 255, 200, 0, 150 });
    }

    // ----------------------------------------------------
    // COMPONENT CALCULATIONS
    // ----------------------------------------------------
    void DrawTearsAndBlush(Rectangle eyeRect, EyeParams p) {
        if (!p.showTears && !p.showBlush) return;

        // 1. Tears
        if (p.showTears) {
            EyeParams tearParams = p;
            tearParams.showBlush = false;

            float tearWidth = eyeRect.width + GlobalScaler.S(130.0f);
            float startY = eyeRect.y + (eyeRect.height * 0.5f) - 5.0f;
            float tearHeight = (float)GetScreenHeight() - startY + 50.0f;

            Rectangle tearRect = {
                eyeRect.x + (eyeRect.width * 0.5f) - (tearWidth * 0.5f),
                startY,
                tearWidth,
                tearHeight
            };

            DrawTearsLayer(tearRect, tearParams);
        }

        // 2. Blush
        if (p.showBlush) {
            EyeParams blushParams = p;
            blushParams.showTears = false;

            float blushSize = eyeRect.width * 0.6f * p.blushScale;
            float xOffset = GlobalScaler.S(150.0f) * -p.eyeSide;
            float yOffset = GlobalScaler.S(200.0f);

            float centerX = eyeRect.x + (eyeRect.width * 0.5f) + xOffset;
            float centerY = eyeRect.y + (eyeRect.height * 0.5f) + yOffset;

            // Extra parameter adjustments
            centerX += GlobalScaler.S(p.blushX * 20.0f);
            centerY += GlobalScaler.S(p.blushY * 20.0f);
            centerX += GlobalScaler.S(p.blushSpacing * 10.0f) * -p.eyeSide;

            Rectangle blushRect = {
                centerX - (blushSize * 0.5f),
                centerY - (blushSize * 0.5f),
                blushSize,
                blushSize
            };

            DrawTearsLayer(blushRect, blushParams);
        }
    }

    void DrawBrows(Rectangle eyeRect, Rectangle originalRect, EyeParams p) {
        if (!p.showBrow) return;

        // Main Brow
        Rectangle browRect = originalRect;
        browRect.width  = originalRect.width * 2.0f * p.browScale * p.eyebrowLength; 
        browRect.height = originalRect.height * 0.5f * p.browScale;

        // Position via physics targets (Reference pixels -> Scaled)
        browRect.x = originalRect.x + GlobalScaler.S(sLookX.val);
        browRect.y = originalRect.y + GlobalScaler.S(sLookY.val) - (originalRect.height * 0.5f);

        // Center on previous rect
        browRect.x += (originalRect.width - browRect.width) * 0.5f;
        browRect.y += (originalRect.height - browRect.height) * 0.5f;


        // Manual Offsets
        browRect.x += GlobalScaler.S(p.eyebrowX * 20.0f);
        browRect.y += GlobalScaler.S(p.eyebrowY * 20.0f);
        browRect.x += GlobalScaler.S(p.eyebrowSpacing * 20.0f) * p.browSide;

        DrawBrowLayer(browRect, p, BLACK);

        // Lower Brow (Under-eye line)
        if (p.useLowerBrow) {
            EyeParams lowerP = p;
            lowerP.eyebrowY = 0.0f;
            lowerP.bend = -p.bend;
            lowerP.eyebrowLength = 1.37f;
            lowerP.eyebrowThickness = 7.15f;

            Vector2 eyeCenter = { 
                eyeRect.x + eyeRect.width * 0.5f,
                eyeRect.y + eyeRect.height * 0.5f 
            };
            float eyeBottomY = eyeCenter.y + (eyeRect.height * 0.5f);

            Rectangle lowerRect = eyeRect;
            lowerRect.width  = eyeRect.width * 2.5f * p.browScale * lowerP.eyebrowLength;
            lowerRect.height = eyeRect.height * (1.0f + fabsf(p.bend));
            lowerRect.x = eyeRect.x + (eyeRect.width - lowerRect.width) * 0.5f;
            
            // Critical alignment fix
            lowerRect.y = eyeBottomY - (lowerRect.height * 0.53f);
            lowerRect.y -= eyeRect.height * 0.3f;
            lowerRect.x += GlobalScaler.S(48.6235f);

            DrawBrowLayer(lowerRect, lowerP, BLACK);
        }
    }

    // ----------------------------------------------------
    // MAIN DRAW CALL
    // ----------------------------------------------------
    void DrawSingleEyeStack(Rectangle rect, EyeParams p, Color c) {
        Vector2 originalRes = { rect.width, rect.height };

        // 1. Physics Transform for Eye Ball
        Rectangle eyeRect = rect;
        eyeRect.x += GlobalScaler.S(sLookX.val);
        eyeRect.y += GlobalScaler.S(sLookY.val);
        
        float oldW = eyeRect.width;
        float oldH = eyeRect.height;
        eyeRect.width  *= sScaleX.val;
        eyeRect.height *= sScaleY.val;
        eyeRect.x += (oldW - eyeRect.width) * 0.5f;
        eyeRect.y += (oldH - eyeRect.height) * 0.5f;

        // 2. Render Order
        if(p.tearsLevel<0.4f)
        {
        DrawEyeLayer(eyeRect, originalRes, p, c);
        DrawTearsAndBlush(eyeRect, p);
        DrawBrows(eyeRect, rect, p);
        }
        else
        {
            DrawTearsAndBlush(eyeRect, p);
            DrawEyeLayer(eyeRect, originalRes, p, c);
            DrawBrows(eyeRect, rect, p);
        }
        
    }

    void Draw(Vector2 centerPos, EyeParams p, Color c) {
        float scaledSize = GlobalScaler.S(Config::REF_EYE_SIZE);
        float scaledW = scaledSize * Config::EYE_SCALE_FACTOR;
        float scaledH = scaledSize * Config::EYE_SCALE_FACTOR;
        float scaledHeightOffset = GlobalScaler.S(Config::EYE_HEIGHT_OFFSET);
        float scaledSpacing = GlobalScaler.S(p.spacing);

        Rectangle leftRect = { 
            centerPos.x - (scaledSpacing * 0.5f) - (scaledW * 0.5f), 
            centerPos.y - scaledHeightOffset - (scaledH * 0.5f), 
            scaledW, scaledH 
        };

        Rectangle rightRect = { 
            centerPos.x + (scaledSpacing * 0.5f) - (scaledW * 0.5f), 
            centerPos.y - scaledHeightOffset - (scaledH * 0.5f), 
            scaledW, scaledH 
        };

        // Color Overrides
        // Star Eyes (Gold) or Heart Eyes (Red)
        bool isStar = (p.eyeShapeID > 3.5 && p.eyeShapeID < 4.5) || (p.eyeShapeID > 7.5 && p.eyeShapeID < 8.5);
        bool isHeart = (p.eyeShapeID > 4.5 && p.eyeShapeID < 5.5);
        
        if (isStar) c = COL_STAR;
        else if (isHeart) c = COL_HEART;

        // Draw Left
        EyeParams leftParams = p;
        leftParams.browSide = -1.0f;
        leftParams.eyeSide = 1.0f;
        DrawSingleEyeStack(leftRect, leftParams, c);
        
        // Draw Right
        EyeParams rightParams = p;
        rightParams.browSide = 1.0f;
        rightParams.eyeSide = -1.0f;
        DrawSingleEyeStack(rightRect, rightParams, c);
    }

    // ----------------------------------------------------
    // UNIFORM CACHING (Boilerplate)
    // ----------------------------------------------------
    void RefreshLocationsEye() {
        shEye.locs.resolution = GetShaderLocation(shEye.shader, "uResolution");
        shEye.locs.time       = GetShaderLocation(shEye.shader, "uTime");
        shEye.locs.color      = GetShaderLocation(shEye.shader, "uColor");
        shEye.locs.shape      = GetShaderLocation(shEye.shader, "uShapeID");
        shEye.locs.bend       = GetShaderLocation(shEye.shader, "uBend");
        shEye.locs.thickness  = GetShaderLocation(shEye.shader, "uThickness");
        shEye.locs.side       = GetShaderLocation(shEye.shader, "uEyeSide");
        shEye.locs.spiral     = GetShaderLocation(shEye.shader, "uSpiralSpeed");
        shEye.locs.distort    = GetShaderLocation(shEye.shader, "uDistortMode");
        shEye.locs.stress     = GetShaderLocation(shEye.shader, "uStressLevel");
        shEye.locs.gloom      = GetShaderLocation(shEye.shader, "uGloomLevel");
    }

    void RefreshLocationsBrow() {
        shBrow.locs.resolution     = GetShaderLocation(shBrow.shader, "uResolution");
        shBrow.locs.color          = GetShaderLocation(shBrow.shader, "uColor");
        shBrow.locs.browType       = GetShaderLocation(shBrow.shader, "uEyebrowType");
        shBrow.locs.bend           = GetShaderLocation(shBrow.shader, "uBend");
        shBrow.locs.thickness      = GetShaderLocation(shBrow.shader, "uThickness");
        shBrow.locs.browLen        = GetShaderLocation(shBrow.shader, "uEyeBrowLength");
        shBrow.locs.side           = GetShaderLocation(shBrow.shader, "uBrowSide");
        shBrow.locs.browAngle      = GetShaderLocation(shBrow.shader, "uAngle");
        shBrow.locs.browBendOffset = GetShaderLocation(shBrow.shader, "uBendOffset");
    }

    void RefreshLocationsTear() {
        shTears.locs.resolution = GetShaderLocation(shTears.shader, "uResolution");
        shTears.locs.scale      = GetShaderLocation(shTears.shader, "uScale");
        shTears.locs.time       = GetShaderLocation(shTears.shader, "uTime");
        shTears.locs.tearLevel  = GetShaderLocation(shTears.shader, "uTearsLevel");
        shTears.locs.blushMode  = GetShaderLocation(shTears.shader, "uBlushMode");
        shTears.locs.showBlush  = GetShaderLocation(shTears.shader, "uShowBlush");
        shTears.locs.blushCol   = GetShaderLocation(shTears.shader, "uBlushColor");
        shTears.locs.tearCol    = GetShaderLocation(shTears.shader, "uTearColor");
        shTears.locs.side       = GetShaderLocation(shTears.shader, "uSide");
    }

    void CheckHotReload() {
        long oldEye = shEye.lastModTime;
        long oldBrow = shBrow.lastModTime;
        long oldTear = shTears.lastModTime;

        shEye.ReloadIfChanged();
        if (shEye.lastModTime > oldEye) RefreshLocationsEye();

        shBrow.ReloadIfChanged();
        if (shBrow.lastModTime > oldBrow) RefreshLocationsBrow();

        shTears.ReloadIfChanged();
        if (shTears.lastModTime > oldTear) RefreshLocationsTear();
    }
};