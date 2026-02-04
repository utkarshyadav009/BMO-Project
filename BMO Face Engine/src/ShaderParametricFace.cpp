// ShaderParametricFace.cpp
// UNIFIED ENGINE: Handles Logic, Physics, and Rendering for Eyes and Mouths
// MERGED FROM: ShaderParametricEyes.cpp and ShaderParametricMouth.cpp

#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>

// Check for utility.h availability, otherwise provide fallback for GlobalScaler
#if __has_include("utility.h")
    #include "utility.h"
#else
    // Fallback if utility.h is missing from compile context
    struct MockScaler {
        float scale = 1.0f;
        void Update() { 
            // Basic screen scale logic if needed, or static 1.0
            scale = (float)GetScreenHeight() / 1000.0f; 
        }
        float S(float v) { return v * scale; }
    };
    static MockScaler GlobalScaler; 
#endif

// --------------------------------------------------------
// SHARED UTILITIES
// --------------------------------------------------------
namespace FaceMath {
    static inline Vector2 V2Add(Vector2 a, Vector2 b) { return {a.x + b.x, a.y + b.y}; }
    static inline Vector2 V2Sub(Vector2 a, Vector2 b) { return {a.x - b.x, a.y - b.y}; }
    static inline Vector2 V2Scale(Vector2 a, float s) { return {a.x * s, a.y * s}; }
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

    // Shader Hot-Reload Wrapper
    struct ShaderAsset {
        Shader shader;
        std::string filePath;
        long lastModTime;
        
        void Load(const char* filename, const char* dir) {
            if (FileExists(filename)) filePath = filename;
            else if (FileExists((std::string(dir) + filename).c_str())) filePath = std::string(dir) + filename;
            else filePath = filename; 

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
                }
            }
        }
        
        void Unload() { UnloadShader(shader); }
    };

    // Physics Spring
    struct Spring {
        float val, vel, target;
        float stiffness, damping;

        void Reset(float initial) { val = initial; target = initial; vel = 0.0f; }
        void Update(float dt) {
            float f = stiffness * (target - val);
            vel = (vel + f * dt) * damping;
            val += vel * dt;
        }
    };
}

// --------------------------------------------------------
// PART A: EYE ENGINE
// --------------------------------------------------------
namespace Eyes {
    namespace Config {
        const char* SHADER_EYE   = "eyes_es.fs";
        const char* SHADER_BROW  = "brow_es.fs";
        const char* SHADER_TEAR  = "tears_es.fs";
        const char* SHADER_PIXELIZER = "pixelizer_es.fs";
        const char* SHADER_DIR   = "src/";

        const float BLINK_MIN_DELAY = 2.0f;
        const float BLINK_MAX_DELAY = 5.0f;
        const float BLINK_HOLD_DURATION = 0.10f;
        const float BLINK_CLOSE_THRESHOLD = 0.2f;
        const float BLINK_OPEN_THRESHOLD = 0.9f;
        const float REF_EYE_SIZE = 100.0f;
        const float EYE_SCALE_FACTOR = 2.7f;
        const float EYE_HEIGHT_OFFSET = 150.0f;
    }

    struct EyeParams {
        float eyeShapeID = 0.0f; float bend = 0.0f; float eyeThickness = 4.0f; float eyeSide = 0.0f;
        float scaleX = 1.0f; float scaleY = 1.0f; float angle = 0.0f; float spacing = 612.0f; float squareness = 0.0f;
        float stressLevel = 0.0f; float gloomLevel = 0.0f; int distortMode = 0; float spiralSpeed = -1.2f;
        float lookX = 0.0f; float lookY = 62.50f;
        bool showBrow = false; bool useLowerBrow = false;
        float eyebrowThickness = 4.0f; float eyebrowLength = 1.0f; float eyebrowSpacing = 0.0f;    
        float eyebrowX = 0.0f; float eyebrowY = 0.0f; float browScale = 1.0f; float browSide = 1.0f; 
        float browAngle = 0.0f; float browBend = 0.0f; float browBendOffset = 0.85f;
        bool showTears = false; bool showBlush = false; float tearsLevel = 0.0f; int blushMode = 0;
        float blushScale = 1.0f; float blushX = 0.0f; float blushY = 0.0f; float blushSpacing = 0.0f;
        float pixelation = 1.0f;
    };

    struct ParametricEyes {
        FaceMath::ShaderAsset shEye, shBrow, shTears, shPixel;
        RenderTexture2D canvas;
        
        struct Locs { 
            int resolution, time, color; 
            int shape, bend, thickness, side, spiral, distort, stress, gloom, squareness; // Eye
            int browLen, browY, browAngle, browBendOffset; // Brow
            int scale, tearLevel, blushMode, showBlush, blushCol, tearCol; // Tears
            int locPixelSize, locRenderSize; // Pixel
        };
        Locs lEye, lBrow, lTear, lPix;

        const Color COL_STAR   = { 255, 184, 0, 255 };
        const Color COL_HEART  = { 220, 1, 1, 255 };
        const Color COL_BLUSH  = { 255, 105, 180, 150 };
        const Color COL_TEAR   = { 100, 180, 255, 220 };

        FaceMath::Spring sScaleX = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
        FaceMath::Spring sScaleY = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
        FaceMath::Spring sLookX  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};
        FaceMath::Spring sLookY  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};

        float blinkTimer = 0.0f;
        float nextBlinkTime = 3.0f;
        int blinkPhase = 0; 
        bool usePhysics = true;
        float hotReloadTimer = 0.0f;
        float currentGlobalScale = 1.0f;
        bool debugBoxes = false;

        void RefreshLocations() {
            // Eye
            lEye.resolution = GetShaderLocation(shEye.shader, "uResolution");
            lEye.time = GetShaderLocation(shEye.shader, "uTime");
            lEye.color = GetShaderLocation(shEye.shader, "uColor");
            lEye.shape = GetShaderLocation(shEye.shader, "uShapeID");
            lEye.bend = GetShaderLocation(shEye.shader, "uBend");
            lEye.thickness = GetShaderLocation(shEye.shader, "uThickness");
            lEye.side = GetShaderLocation(shEye.shader, "uEyeSide");
            lEye.spiral = GetShaderLocation(shEye.shader, "uSpiralSpeed");
            lEye.distort = GetShaderLocation(shEye.shader, "uDistortMode");
            lEye.stress = GetShaderLocation(shEye.shader, "uStressLevel");
            lEye.gloom = GetShaderLocation(shEye.shader, "uGloomLevel");
            lEye.squareness = GetShaderLocation(shEye.shader, "uSquareness");
            // Brow
            lBrow.resolution = GetShaderLocation(shBrow.shader, "uResolution");
            lBrow.color = GetShaderLocation(shBrow.shader, "uColor");
            lBrow.bend = GetShaderLocation(shBrow.shader, "uBend");
            lBrow.thickness = GetShaderLocation(shBrow.shader, "uThickness");
            lBrow.browLen = GetShaderLocation(shBrow.shader, "uEyeBrowLength");
            lBrow.side = GetShaderLocation(shBrow.shader, "uBrowSide");
            lBrow.browAngle = GetShaderLocation(shBrow.shader, "uAngle");
            lBrow.browBendOffset = GetShaderLocation(shBrow.shader, "uBendOffset");
            // Tears
            lTear.resolution = GetShaderLocation(shTears.shader, "uResolution");
            lTear.scale = GetShaderLocation(shTears.shader, "uScale");
            lTear.time = GetShaderLocation(shTears.shader, "uTime");
            lTear.tearLevel = GetShaderLocation(shTears.shader, "uTearsLevel");
            lTear.blushMode = GetShaderLocation(shTears.shader, "uBlushMode");
            lTear.showBlush = GetShaderLocation(shTears.shader, "uShowBlush");
            lTear.blushCol = GetShaderLocation(shTears.shader, "uBlushColor");
            lTear.tearCol = GetShaderLocation(shTears.shader, "uTearColor");
            lTear.side = GetShaderLocation(shTears.shader, "uSide");
            // Pixel
            lPix.locPixelSize = GetShaderLocation(shPixel.shader, "pixelSize");
            lPix.locRenderSize = GetShaderLocation(shPixel.shader, "renderSize");
        }

        void Init() {
            shEye.Load(Config::SHADER_EYE, Config::SHADER_DIR);
            shBrow.Load(Config::SHADER_BROW, Config::SHADER_DIR);
            shTears.Load(Config::SHADER_TEAR, Config::SHADER_DIR);
            shPixel.Load(Config::SHADER_PIXELIZER, Config::SHADER_DIR);
            RefreshLocations();
            canvas = LoadRenderTexture(GetScreenHeight(), GetScreenHeight());
        }

        void Unload() {
            shEye.Unload(); shBrow.Unload(); shTears.Unload(); shPixel.Unload();
            UnloadRenderTexture(canvas);
        }

        void Update(float dt, EyeParams& params) {
            currentGlobalScale = GlobalScaler.scale;
            hotReloadTimer += dt;
            if (hotReloadTimer > 1.0f) {
                shEye.ReloadIfChanged(); shBrow.ReloadIfChanged(); 
                shTears.ReloadIfChanged(); shPixel.ReloadIfChanged();
                RefreshLocations();
                hotReloadTimer = 0.0f;
            }

            if (!usePhysics) {
                sScaleX.val = params.scaleX; sScaleY.val = params.scaleY;
                sLookX.val = params.lookX; sLookY.val = params.lookY;
                blinkPhase = 0;
                return;
            }

            // Blink Logic
            bool canBlink = (params.eyeShapeID < 0.5f || params.eyeShapeID > 3.5f);
            if (canBlink) {
                blinkTimer += dt;
                if (blinkTimer > nextBlinkTime && blinkPhase == 0) {
                    blinkPhase = 1; blinkTimer = 0.0f;
                    nextBlinkTime = (float)GetRandomValue((int)(Config::BLINK_MIN_DELAY*10), (int)(Config::BLINK_MAX_DELAY*10))/10.0f;
                }
            } else blinkPhase = 0;

            if (blinkPhase == 1) { // Closing
                sScaleY.target = 0.0f;
                if (sScaleY.val < Config::BLINK_CLOSE_THRESHOLD) { blinkPhase = 2; blinkTimer = 0.0f; }
            } else if (blinkPhase == 2) { // Closed
                sScaleY.val = 1.0f; sScaleY.target = 1.0f; 
                if (blinkTimer > Config::BLINK_HOLD_DURATION) { blinkPhase = 3; sScaleY.val = 0.0f; }
            } else if (blinkPhase == 3) { // Opening
                sScaleY.target = params.scaleY;
                if (sScaleY.val > params.scaleY * Config::BLINK_OPEN_THRESHOLD) blinkPhase = 0;
            } else { // Open / Breathing
                float breath = (params.scaleY > 0.8f) ? sinf((float)GetTime() * 2.0f) * 0.02f : 0.0f;
                sScaleY.target = params.scaleY + breath;
                sScaleX.target = params.scaleX - breath;
            }
            sLookX.target = params.lookX; sLookY.target = params.lookY;
            sScaleX.Update(dt); sScaleY.Update(dt); sLookX.Update(dt); sLookY.Update(dt);
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
        
        void DrawDebug(Rectangle r, Color c) {
             if(debugBoxes) { DrawRectangleLinesEx(r, 2.0f, c); DrawRectangleRec(r, Fade(c, 0.2f)); }
        }

        void DrawLayer(Shader s, Locs& locs, Rectangle r, Color c, int uRes, int uTime, int uCol, std::function<void()> setUniforms) {
            rlSetTexture(0);
            BeginShaderMode(s);
            float res[2] = { r.width, r.height }; float time = (float)GetTime();
            float col[4] = { c.r/255.f, c.g/255.f, c.b/255.f, c.a/255.f };
            if (uRes != -1) SetShaderValue(s, uRes, res, SHADER_UNIFORM_VEC2);
            if (uTime != -1) SetShaderValue(s, uTime, &time, SHADER_UNIFORM_FLOAT);
            if (uCol != -1) SetShaderValue(s, uCol, col, SHADER_UNIFORM_VEC4);
            if (setUniforms) setUniforms();
            DrawQuad(r, c);
            EndShaderMode();
        }

        void DrawStack(Rectangle rect, EyeParams p, Color c) {
            Vector2 origRes = { rect.width, rect.height };
            Rectangle eyeRect = rect;
            eyeRect.x += GlobalScaler.S(sLookX.val); eyeRect.y += GlobalScaler.S(sLookY.val);
            float oldW = eyeRect.width, oldH = eyeRect.height;
            eyeRect.width *= sScaleX.val; eyeRect.height *= sScaleY.val;
            eyeRect.x += (oldW - eyeRect.width)*0.5f; eyeRect.y += (oldH - eyeRect.height)*0.5f;
            if (p.eyeShapeID >= 12.0f) eyeRect.height *= 2.0f;

            auto RenderEye = [&]() {
                DrawLayer(shEye.shader, lEye, eyeRect, c, lEye.resolution, lEye.time, lEye.color, [&](){
                    float scaledThick = GlobalScaler.S(p.eyeThickness);
                    float finalShape = (blinkPhase == 2) ? 1.0f : p.eyeShapeID;
                    SetShaderValue(shEye.shader, lEye.shape, &finalShape, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shEye.shader, lEye.bend, &p.bend, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shEye.shader, lEye.thickness, &scaledThick, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shEye.shader, lEye.side, &p.eyeSide, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shEye.shader, lEye.spiral, &p.spiralSpeed, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shEye.shader, lEye.distort, &p.distortMode, SHADER_UNIFORM_INT);
                    SetShaderValue(shEye.shader, lEye.stress, &p.stressLevel, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shEye.shader, lEye.gloom, &p.gloomLevel, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shEye.shader, lEye.squareness, &p.squareness, SHADER_UNIFORM_FLOAT);
                });
                DrawDebug(eyeRect, RED);
            };

            auto RenderTearsBlush = [&]() {
                if (p.showTears) {
                    float tw = eyeRect.width + GlobalScaler.S(130.f);
                    Rectangle tr = { eyeRect.x + eyeRect.width*0.5f - tw*0.5f, eyeRect.y + eyeRect.height*0.5f - 5.f, tw, (float)GetScreenHeight() - (eyeRect.y) };
                    EyeParams tp = p; tp.showBlush = false;
                    DrawLayer(shTears.shader, lTear, tr, WHITE, lTear.resolution, lTear.time, -1, [&](){
                        float pink[4] = { COL_BLUSH.r/255.f, COL_BLUSH.g/255.f, COL_BLUSH.b/255.f, COL_BLUSH.a/255.f };
                        float blue[4] = { COL_TEAR.r/255.f, COL_TEAR.g/255.f, COL_TEAR.b/255.f, COL_TEAR.a/255.f };
                        int bt = 0;
                        SetShaderValue(shTears.shader, lTear.scale, &currentGlobalScale, SHADER_UNIFORM_FLOAT);
                        SetShaderValue(shTears.shader, lTear.tearLevel, &tp.tearsLevel, SHADER_UNIFORM_FLOAT);
                        SetShaderValue(shTears.shader, lTear.showBlush, &bt, SHADER_UNIFORM_INT);
                        SetShaderValue(shTears.shader, lTear.blushCol, pink, SHADER_UNIFORM_VEC4);
                        SetShaderValue(shTears.shader, lTear.tearCol, blue, SHADER_UNIFORM_VEC4);
                        SetShaderValue(shTears.shader, lTear.side, &p.eyeSide, SHADER_UNIFORM_FLOAT);
                    });
                    DrawDebug(tr, BLUE);
                }
                if (p.showBlush) {
                    float bs = eyeRect.width * 0.6f * p.blushScale;
                    float cx = eyeRect.x + eyeRect.width*0.5f + GlobalScaler.S(150.f * -p.eyeSide + p.blushX * 20.f + p.blushSpacing * 10.f * -p.eyeSide);
                    float cy = eyeRect.y + eyeRect.height*0.5f + GlobalScaler.S(200.f + p.blushY * 20.f);
                    Rectangle br = { cx - bs*0.5f, cy - bs*0.5f, bs, bs };
                    DrawLayer(shTears.shader, lTear, br, WHITE, lTear.resolution, lTear.time, -1, [&](){
                         float pink[4] = { COL_BLUSH.r/255.f, COL_BLUSH.g/255.f, COL_BLUSH.b/255.f, COL_BLUSH.a/255.f };
                         int bt = 1;
                         SetShaderValue(shTears.shader, lTear.scale, &currentGlobalScale, SHADER_UNIFORM_FLOAT);
                         SetShaderValue(shTears.shader, lTear.blushMode, &p.blushMode, SHADER_UNIFORM_INT);
                         SetShaderValue(shTears.shader, lTear.showBlush, &bt, SHADER_UNIFORM_INT);
                         SetShaderValue(shTears.shader, lTear.blushCol, pink, SHADER_UNIFORM_VEC4);
                    });
                    DrawDebug(br, MAGENTA);
                }
            };

            auto RenderBrows = [&]() {
                if (!p.showBrow) return;
                // Main Brow
                Rectangle br = rect;
                br.width = rect.width * 2.0f * p.browScale * p.eyebrowLength;
                br.height = rect.height * 0.5f * p.browScale;
                br.x = rect.x + GlobalScaler.S(sLookX.val) + (rect.width - br.width)*0.5f + GlobalScaler.S(p.eyebrowX * 20.f + p.eyebrowSpacing*20.f*p.browSide);
                br.y = rect.y + GlobalScaler.S(sLookY.val) - (rect.height*0.5f) + (rect.height - br.height)*0.5f + GlobalScaler.S(p.eyebrowY * 20.f);
                
                DrawLayer(shBrow.shader, lBrow, br, BLACK, lBrow.resolution, -1, lBrow.color, [&](){
                    float st = GlobalScaler.S(p.eyebrowThickness);
                    SetShaderValue(shBrow.shader, lBrow.bend, &p.browBend, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shBrow.shader, lBrow.thickness, &st, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shBrow.shader, lBrow.browLen, &p.eyebrowLength, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shBrow.shader, lBrow.side, &p.browSide, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shBrow.shader, lBrow.browAngle, &p.browAngle, SHADER_UNIFORM_FLOAT);
                    SetShaderValue(shBrow.shader, lBrow.browBendOffset, &p.browBendOffset, SHADER_UNIFORM_FLOAT);
                });
                DrawDebug(br, YELLOW);

                if (p.useLowerBrow) {
                    EyeParams lp = p; lp.eyebrowY = 0.f; lp.bend = -p.bend; lp.eyebrowLength = 1.37f; lp.eyebrowThickness = 7.15f;
                    Rectangle lbr = eyeRect;
                    lbr.width = eyeRect.width * 2.5f * p.browScale * lp.eyebrowLength;
                    lbr.height = eyeRect.height * (1.0f + fabsf(p.bend));
                    lbr.x = eyeRect.x + (eyeRect.width - lbr.width)*0.5f + GlobalScaler.S(48.6235f);
                    lbr.y = eyeRect.y + eyeRect.height*0.5f + eyeRect.height*0.5f - (lbr.height*0.53f) - eyeRect.height*0.3f;
                    
                    DrawLayer(shBrow.shader, lBrow, lbr, BLACK, lBrow.resolution, -1, lBrow.color, [&](){
                        float st = GlobalScaler.S(lp.eyebrowThickness);
                        SetShaderValue(shBrow.shader, lBrow.bend, &lp.browBend, SHADER_UNIFORM_FLOAT);
                        SetShaderValue(shBrow.shader, lBrow.thickness, &st, SHADER_UNIFORM_FLOAT);
                        SetShaderValue(shBrow.shader, lBrow.browLen, &lp.eyebrowLength, SHADER_UNIFORM_FLOAT);
                        SetShaderValue(shBrow.shader, lBrow.side, &lp.browSide, SHADER_UNIFORM_FLOAT);
                        SetShaderValue(shBrow.shader, lBrow.browAngle, &lp.browAngle, SHADER_UNIFORM_FLOAT);
                        float off = 0.85f; SetShaderValue(shBrow.shader, lBrow.browBendOffset, &off, SHADER_UNIFORM_FLOAT);
                    });
                    DrawDebug(lbr, ORANGE);
                }
            };

            if (p.tearsLevel < 0.4f) { RenderEye(); RenderTearsBlush(); RenderBrows(); }
            else { RenderTearsBlush(); RenderEye(); RenderBrows(); }
        }

        void Draw(Vector2 centerPos, EyeParams p, Color c) {
            if (canvas.texture.width != GetScreenWidth() || canvas.texture.height != GetScreenHeight()) {
                UnloadRenderTexture(canvas); canvas = LoadRenderTexture(GetScreenWidth(), GetScreenHeight());
            }

            BeginTextureMode(canvas);
            ClearBackground(BLANK);
            float ss = GlobalScaler.S(Config::REF_EYE_SIZE);
            float sw = ss * Config::EYE_SCALE_FACTOR;
            float sh = ss * Config::EYE_SCALE_FACTOR;
            float shOffset = GlobalScaler.S(Config::EYE_HEIGHT_OFFSET);
            float sp = GlobalScaler.S(p.spacing);

            rlPushMatrix();
            if (fabsf(p.angle) > 0.01f) {
                rlTranslatef(centerPos.x, centerPos.y, 0); rlRotatef(p.angle, 0, 0, 1); rlTranslatef(-centerPos.x, -centerPos.y, 0);
            }
            
            bool isStar = (p.eyeShapeID > 3.5 && p.eyeShapeID < 4.5) || (p.eyeShapeID > 7.5 && p.eyeShapeID < 8.5);
            bool isHeart = (p.eyeShapeID > 4.5 && p.eyeShapeID < 5.5);
            if (isStar) c = COL_STAR; else if (isHeart) c = COL_HEART;

            EyeParams lp = p; lp.browSide = -1.0f; lp.eyeSide = 1.0f;
            DrawStack({centerPos.x - sp*0.5f - sw*0.5f, centerPos.y - shOffset - sh*0.5f, sw, sh}, lp, c);
            
            EyeParams rp = p; rp.browSide = 1.0f; rp.eyeSide = -1.0f;
            DrawStack({centerPos.x + sp*0.5f - sw*0.5f, centerPos.y - shOffset - sh*0.5f, sw, sh}, rp, c);
            rlPopMatrix();
            EndTextureMode();

            BeginShaderMode(shPixel.shader);
            float ps = p.pixelation; float rs[2] = { (float)canvas.texture.width, (float)canvas.texture.height };
            SetShaderValue(shPixel.shader, lPix.locPixelSize, &ps, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shPixel.shader, lPix.locRenderSize, rs, SHADER_UNIFORM_VEC2);
            DrawTexturePro(canvas.texture, {0,0,rs[0],-rs[1]}, {0,0,(float)GetScreenWidth(),(float)GetScreenHeight()}, {0,0}, 0.0f, WHITE);
            EndShaderMode();
        }
    };
}

// --------------------------------------------------------
// PART B: MOUTH ENGINE
// --------------------------------------------------------
namespace Mouth {
    struct MouthParams {
        float open = 0.05f; float width = 0.5f; float curve = 0.0f; 
        float squeezeTop = 0.0f; float squeezeBottom = 0.0f; 
        float teethY = 0.0f; float tongueUp = 0.0f; float tongueX = 0.0f;
        float tongueWidth = 0.65f; float asymmetry = 0.0f; float squareness = 0.0f;
        float teethWidth = 0.50f; float teethGap = 45.0f; float scale = 1.0f; float outlineThickness = 1.5f;
        float sigma = 0.45f; float power = 6.0f; float maxLiftValue = 0.55f;
    };

    static MouthParams MakeDefaultParams() {
        MouthParams p{};
        p.open = 0.05f; p.width = 0.5f; p.curve = 0.2f; p.scale = 1.0f;
        p.teethWidth = 0.5f; p.teethGap = 45.0f; p.tongueWidth = 0.65f;
        p.teethY = -1.0f; p.squeezeTop = 0.0f; p.squeezeBottom = 0.0f;
        p.tongueUp = 0.0f; p.tongueX = 0.0f; p.asymmetry = 0.0f;
        p.squareness = 0.0f; p.sigma = 0.45f; p.power = 6.0f; p.maxLiftValue = 0.55f;
        p.outlineThickness = 1.5f;
        return p;
    }

    // Helper: Normalize poly
    static void ResamplePoly(const std::vector<Vector2>& input, float* outputFlat, int maxPts, int& outCount, float scale, Vector2 offset, bool closedLoop) {
        if (input.empty()) { outCount = 0; return; }
        if (input.size() <= (size_t)maxPts) {
            outCount = (int)input.size();
            for(int i=0; i<outCount; i++) {
                outputFlat[i*2] = (input[i].x - offset.x) * scale;
                outputFlat[i*2+1] = (input[i].y - offset.y) * scale;
            }
            return;
        }
        outCount = maxPts;
        float totalLen = 0;
        for (size_t i = 0; i < input.size(); i++) {
            if (!closedLoop && i == input.size() - 1) break;
            totalLen += FaceMath::V2Dist(input[i], input[(i + 1) % input.size()]);
        }
        float step = totalLen / (float)(closedLoop ? maxPts : maxPts - 1);
        outputFlat[0] = (input[0].x - offset.x) * scale;
        outputFlat[1] = (input[0].y - offset.y) * scale;
        int ptsEmitted = 1; float currentDist = 0.0f; float nextSample = step;
        for (size_t i = 0; i < input.size(); i++) {
            if (ptsEmitted >= maxPts) break;
            if (!closedLoop && i == input.size() - 1) break;
            Vector2 p1 = input[i]; Vector2 p2 = input[(i + 1) % input.size()];
            float segLen = FaceMath::V2Dist(p1, p2);
            if (segLen < 1e-6f) { currentDist += segLen; continue; }
            while (nextSample <= currentDist + segLen) {
                float t = (nextSample - currentDist) / segLen;
                outputFlat[ptsEmitted*2] = (p1.x + (p2.x - p1.x) * t - offset.x) * scale;
                outputFlat[ptsEmitted*2+1] = (p1.y + (p2.y - p1.y) * t - offset.y) * scale;
                ptsEmitted++; nextSample += step;
                if (ptsEmitted >= maxPts) break;
            }
            currentDist += segLen;
        }
    }

    struct ParametricMouth {
        MouthParams current, target, velocity;
        std::vector<Vector2> controlPoints; 
        std::vector<Vector2> smoothContour, topTeethPoly, botTeethPoly, tonguePoly;
        Shader sdfShader;
        bool shaderLoaded = false;
        int locMouthPts, locMouthCnt, locTopPts, locTopCnt, locBotPts, locBotCnt, locTonguePts, locTongueCnt;
        int locRes, locPadding, locScale, locOutlineThickness, locColBg, locColLine, locColTeeth, locColTongue;
        Vector2 centerPos; bool usePhysics = true;
        const float GEO_SIZE = 1024.0f; const float SS = 4.0f; 
        Color colBg={57,99,55,255}, colLine={20,35,20,255}, colTeeth={245,245,245,255}, colTongue={162,178,106,255}; 

        void Init(Vector2 pos) {
            velocity = {}; centerPos = pos;
            controlPoints.resize(16); smoothContour.reserve(512); 
            topTeethPoly.reserve(64); botTeethPoly.reserve(64); tonguePoly.reserve(64);
            const char* filename = "mouth_es.fs";
            std::string path = std::string(GetApplicationDirectory()) + filename;
            if (!FileExists(path.c_str())) path = filename; 
            if (FileExists(path.c_str())) {
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
                     locScale    = GetShaderLocation(sdfShader, "uScale");
                     locOutlineThickness = GetShaderLocation(sdfShader, "uOutlineThickness");
                     locColBg    = GetShaderLocation(sdfShader, "uColBg");
                     locColLine  = GetShaderLocation(sdfShader, "uColLine");
                     locColTeeth = GetShaderLocation(sdfShader, "uColTeeth");
                     locColTongue= GetShaderLocation(sdfShader, "uColTongue");
                 }
            } else { std::cout << "[Mouth] SHADER MISSING: " << path << std::endl; }
            target = MakeDefaultParams(); current = target;
        }

        void Unload() { if(shaderLoaded) UnloadShader(sdfShader); }
        void resetPosition(Vector2 pos) { centerPos = pos; }

        void UpdatePhysics(float dt) {
            target.open = Clamp(target.open, 0.0f, 1.2f);
            target.scale = Clamp(target.scale, 0.5f, 4.0f);
            if (!usePhysics) { current = target; return; }
            if (dt > 0.05f) dt = 0.05f; 
            const float STIFFNESS = 180.0f, DAMPING = 14.0f;    
            auto Upd = [&](float& c, float& v, float t) {
                float f = STIFFNESS * (t - c); float d = DAMPING * v;
                v += (f - d) * dt; c += v * dt;
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
            float cx = GEO_SIZE * 0.5f; float cy = GEO_SIZE * 0.5f;
            float baseRadius = 40.0f * SS; 
            float w = baseRadius * (0.5f + current.width);
            float h = (current.open < 0.08f) ? 0.0f : (baseRadius * (0.2f + current.open * 1.5f));

            for (int i = 0; i < 16; i++) {
                float t = (float)i / 16.0f; float angle = t * PI * 2.0f + PI; 
                float x = cosf(angle) * w; float rawSin = sinf(angle);
                bool isTop = (i <= 8);
                float flatness = 0.0f;
                if (current.asymmetry > 0.0f && isTop) flatness = current.asymmetry; 
                if (current.asymmetry < 0.0f && !isTop) flatness = -current.asymmetry;

                float effectiveH = h; float wave = std::abs(rawSin);
                if (flatness > 0.01f) {
                    effectiveH *= (1.0f - flatness * 0.6f); 
                    wave = std::pow(wave, 1.0f - (flatness * 0.5f)); 
                }
                float sqPower = 1.0f - (current.squareness * 0.8f);
                float shapedSin = std::pow(wave, sqPower);
                float sign = (rawSin >= 0.0f) ? 1.0f : -1.0f;
                float y = shapedSin * effectiveH * sign;
                
                float bendFactor = 15.0f * SS; float normalizedX = x / w;
                float rawBend = (normalizedX * normalizedX) * bendFactor * current.curve;
                float bendMult = (flatness > 0.0f) ? 1.0f - flatness : 1.0f;
                y -= rawBend * bendMult; 

                float activeSqueeze = isTop ? current.squeezeTop : current.squeezeBottom;
                float tX = Clamp(std::abs(x) / (w + 1e-6f), 0.0f, 1.0f);
                float u = tX / current.sigma;
                float influence = expf(-powf(u, current.power));  
                float lift = activeSqueeze * influence * (h * current.maxLiftValue);
                float archSign = (rawSin >= 0.0f) ? 1.0f : -1.0f;
                y -= lift * archSign;
                controlPoints[i] = { cx + x, cy + y };
            }

            smoothContour.clear();
            for (int i = 0; i < 16; i++) {
                Vector2 p0 = controlPoints[(i-1+16)%16]; Vector2 p1 = controlPoints[i];
                Vector2 p2 = controlPoints[(i+1)%16]; Vector2 p3 = controlPoints[(i+2)%16];
                for (int k = 0; k < 16; k++) smoothContour.push_back(FaceMath::CatmullRom(p0, p1, p2, p3, (float)k/16.0f));
            }
            if (smoothContour.size() < 2) return;
            // Clean duplicates
            std::vector<Vector2> clean; clean.push_back(smoothContour[0]);
            for (size_t i=1; i<smoothContour.size(); i++){
                if (FaceMath::V2Dist(smoothContour[i], clean.back()) > 0.5f) clean.push_back(smoothContour[i]);
            }
            if (clean.size() > 2 && FaceMath::V2Dist(clean.front(), clean.back()) <= 0.5f) clean.pop_back();
            smoothContour = clean;

            // Rotate lowest Y to start (approximate fix for teeth rendering order consistency)
            if(!smoothContour.empty()) {
                int best = 0;
                for (int i = 1; i < (int)smoothContour.size(); i++) {
                    if (smoothContour[i].x < smoothContour[best].x || (smoothContour[i].x == smoothContour[best].x && smoothContour[i].y < smoothContour[best].y)) best = i;
                }
                std::rotate(smoothContour.begin(), smoothContour.begin() + best, smoothContour.end());
                if (smoothContour.size()>4) {
                    int n = (int)smoothContour.size();
                    if (smoothContour[n/4].y > smoothContour[(3*n)/4].y) {
                        std::reverse(smoothContour.begin(), smoothContour.end());
                        best=0; for (int i=1; i<(int)smoothContour.size(); i++) if(smoothContour[i].x < smoothContour[best].x) best=i;
                        std::rotate(smoothContour.begin(), smoothContour.begin() + best, smoothContour.end());
                    }
                }
            }

            topTeethPoly.clear(); botTeethPoly.clear(); tonguePoly.clear();
            if (current.open > 0.10f && !smoothContour.empty()) {
                float minY = 1e10, maxY = -1e10, minX = 1e10, maxX = -1e10;
                for (const auto& p : smoothContour) {
                    if(p.y < minY) minY = p.y; if(p.y > maxY) maxY = p.y;
                    if(p.x < minX) minX = p.x; if(p.x > maxX) maxX = p.x;
                }
                
                float shiftX = current.tongueX * (w * 0.4f);
                float tongueCX = cx + shiftX;
                float tongueW = (maxX - minX) * 0.5f * current.tongueWidth;
                float tongueTip = Lerp(maxY, minY, current.tongueUp);
                for (int i = 0; i <= 32; i++) {
                    float t = (float)i / 32.0f; float tx = -cosf(t * PI); float ty = sinf(t * PI);  
                    tonguePoly.push_back({tongueCX + (tx * tongueW), Lerp(maxY, tongueTip, ty)});
                }
                tonguePoly.push_back({tongueCX + tongueW, maxY + 10});
                tonguePoly.push_back({tongueCX - tongueW, maxY + 10});
                
                float teethW = (maxX - minX) * current.teethWidth * 0.5f;
                float gap = current.teethGap * SS;
                float midY = cy + (current.teethY * 20.0f * SS);
                float topTeethEdge = midY - gap/2; float botTeethEdge = midY + gap/2;

                if(gap != 400 && topTeethEdge > minY + 5.0f) {
                    float tW_curr = (current.teethWidth >= 0.95f) ? (maxX - minX)*0.6f : teethW;
                    float tx1 = (current.teethWidth >= 0.95f) ? minX : cx - teethW;
                    float tx2 = (current.teethWidth >= 0.95f) ? maxX : cx + teethW;
                    topTeethPoly = { {tx1, minY}, {tx2, minY}, {tx2, topTeethEdge}, {tx1, topTeethEdge} };
                }
                if(gap != 400 && botTeethEdge < maxY - 5.0f) {
                     float tW_curr = (current.teethWidth >= 0.95f) ? (maxX - minX)*0.6f : teethW;
                     float tx1 = (current.teethWidth >= 0.95f) ? minX : cx - teethW;
                     float tx2 = (current.teethWidth >= 0.95f) ? maxX : cx + teethW;
                     botTeethPoly = { {tx1, botTeethEdge}, {tx2, botTeethEdge}, {tx2, maxY}, {tx1, maxY} };
                }
            }
        }

        void Draw() {
            if (!shaderLoaded || smoothContour.empty()) return;
            const float paddingPx = 8.0f; 
            float unitScale = (256.0f / GEO_SIZE) * current.scale;
            if (unitScale < 0.0001f) unitScale = 0.0001f;
            
            // Calc bounds
            float minX = 1e10f, maxX = -1e10f, minY = 1e10f, maxY = -1e10f;
            for(auto& p : smoothContour) { if(p.x < minX) minX = p.x; if(p.x > maxX) maxX = p.x; if(p.y < minY) minY = p.y; if(p.y > maxY) maxY = p.y; }
            Rectangle boundsPhys = {minX, minY, maxX - minX, maxY - minY};

            float paddingPhys = paddingPx / unitScale;
            boundsPhys.x -= paddingPhys; boundsPhys.y -= paddingPhys;
            boundsPhys.width += paddingPhys * 2.0f; boundsPhys.height += paddingPhys * 2.0f;

            float physRefLeft = centerPos.x - (GEO_SIZE * 0.5f * unitScale);
            float physRefTop  = centerPos.y - (GEO_SIZE * 0.5f * unitScale);
            Rectangle screenRect = { physRefLeft + (boundsPhys.x * unitScale), physRefTop  + (boundsPhys.y * unitScale), boundsPhys.width * unitScale, boundsPhys.height * unitScale };

            const int MAX_PTS = 64;
            float fMouth[MAX_PTS*2], fTop[MAX_PTS*2], fBot[MAX_PTS*2], fTng[MAX_PTS*2];
            int cM=0, cT=0, cB=0, cTg=0;
            Vector2 offset = {boundsPhys.x, boundsPhys.y};
            
            ResamplePoly(smoothContour, fMouth, MAX_PTS, cM, unitScale, offset, true);
            ResamplePoly(topTeethPoly, fTop, MAX_PTS, cT, unitScale, offset, true);
            ResamplePoly(botTeethPoly, fBot, MAX_PTS, cB, unitScale, offset, true);
            ResamplePoly(tonguePoly, fTng, MAX_PTS, cTg, unitScale, offset, true);

            auto V4 = [](Color c){ return Vector4{c.r/255.f, c.g/255.f, c.b/255.f, c.a/255.f}; };

            BeginShaderMode(sdfShader);
                SetShaderValueV(sdfShader, locMouthPts, fMouth, SHADER_UNIFORM_VEC2, MAX_PTS);
                SetShaderValue(sdfShader, locMouthCnt, &cM, SHADER_UNIFORM_INT);
                SetShaderValueV(sdfShader, locTopPts, fTop, SHADER_UNIFORM_VEC2, MAX_PTS);
                SetShaderValue(sdfShader, locTopCnt, &cT, SHADER_UNIFORM_INT);
                SetShaderValueV(sdfShader, locBotPts, fBot, SHADER_UNIFORM_VEC2, MAX_PTS);
                SetShaderValue(sdfShader, locBotCnt, &cB, SHADER_UNIFORM_INT);
                SetShaderValueV(sdfShader, locTonguePts, fTng, SHADER_UNIFORM_VEC2, MAX_PTS);
                SetShaderValue(sdfShader, locTongueCnt, &cTg, SHADER_UNIFORM_INT);
                
                float res[2] = { screenRect.width, screenRect.height };
                SetShaderValue(sdfShader, locRes, res, SHADER_UNIFORM_VEC2);
                SetShaderValue(sdfShader, locPadding, &paddingPx, SHADER_UNIFORM_FLOAT);
                SetShaderValue(sdfShader, locScale, &current.scale, SHADER_UNIFORM_FLOAT);
                SetShaderValue(sdfShader, locOutlineThickness, &current.outlineThickness, SHADER_UNIFORM_FLOAT);
                
                Vector4 cb=V4(colBg), cl=V4(colLine), ct=V4(colTeeth), ctg=V4(colTongue);
                SetShaderValue(sdfShader, locColBg, &cb, SHADER_UNIFORM_VEC4);
                SetShaderValue(sdfShader, locColLine, &cl, SHADER_UNIFORM_VEC4);
                SetShaderValue(sdfShader, locColTeeth, &ct, SHADER_UNIFORM_VEC4);
                SetShaderValue(sdfShader, locColTongue, &ctg, SHADER_UNIFORM_VEC4);

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
}