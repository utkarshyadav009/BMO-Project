#include "raylib.h"
#include "rlgl.h"
#include <cmath>
#include <string>
#include <iostream>

struct Spring {
    float val, vel, target;
    float stiffness, damping;
    void Update(float dt) {
        float f = stiffness * (target - val);
        vel = (vel + f * dt) * damping;
        val += vel * dt;
    }
};

struct EyeParams {
    float shapeID = 0.0f;    
    float bend = 0.0f;       
    float thickness = 4.0f; 
    float pupilSize = 0.0f;  
    float targetLookX = 0.0f;
    float targetLookY = 0.0f;
};

struct ParametricEyes {
    Shader shader;
    
    // PHYSICS (Tuned for speed)
    // [FIX] Increased Stiffness 200 -> 600 for Snappy Blinks
    Spring sScaleX = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f}; 
    Spring sScaleY = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
    Spring sLookX  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f}; 
    Spring sLookY  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};
    
    float blinkTimer = 0.0f;
    float nextBlinkTime = 3.0f;
    bool isBlinking = false;
    int blinkPhase = 0; 

    int locRes, locColor, locShape, locBend, locThick, locPupil;

    void Init() {
        const char* filename = "eyes_es.fs";
        std::string path = std::string(GetApplicationDirectory()) + filename;
        if (!FileExists(path.c_str())) path = filename;

        shader = LoadShader(0, path.c_str());
        locRes   = GetShaderLocation(shader, "uResolution");
        locColor = GetShaderLocation(shader, "uColor");
        locShape = GetShaderLocation(shader, "uShapeID");
        locBend  = GetShaderLocation(shader, "uBend");
        locThick = GetShaderLocation(shader, "uThickness");
        locPupil = GetShaderLocation(shader, "uPupilSize");
    }

    void Update(float dt, EyeParams& params) {
        // AUTO BLINK
        if (params.shapeID < 0.5f) {
            blinkTimer += dt;
            if (blinkTimer > nextBlinkTime && blinkPhase == 0) {
                blinkPhase = 1;
                blinkTimer = 0.0f;
                nextBlinkTime = (float)GetRandomValue(20, 50) / 10.0f;
            }
        } else {
            blinkPhase = 0; 
        }

        // BLINK ANIMATION
        if (blinkPhase == 1) { // Closing
            sScaleY.target = 0.0f; 
            if (sScaleY.val < 0.2f) { blinkPhase = 2; blinkTimer = 0.0f; }
        } 
        else if (blinkPhase == 2) { // Closed (HOLD)
            sScaleY.val = 1.0f; sScaleY.target = 1.0f;
            // [FIX] Shorter hold time (0.15s -> 0.10s)
            if (blinkTimer > 0.10f) { blinkPhase = 3; sScaleY.val = 0.0f; }
        }
        else if (blinkPhase == 3) { // Opening
            sScaleY.target = 1.0f; 
            if (sScaleY.val > 0.9f) blinkPhase = 0;
        }

        sLookX.target = params.targetLookX;
        sLookY.target = params.targetLookY;

        if (blinkPhase == 0) {
             float breath = sinf((float)GetTime() * 2.0f) * 0.02f;
             sScaleY.target = 1.0f + breath;
             sScaleX.target = 1.0f - breath;
        }

        sScaleX.Update(dt); sScaleY.Update(dt);
        sLookX.Update(dt);  sLookY.Update(dt);
    }

    void DrawEye(Rectangle rect, EyeParams p, Color c) {
        rlSetTexture(0);
        BeginShaderMode(shader);
            float res[2] = { rect.width, rect.height };
            SetShaderValue(shader, locRes, res, SHADER_UNIFORM_VEC2);
            float col[4] = { c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f };
            SetShaderValue(shader, locColor, col, SHADER_UNIFORM_VEC4);

            // [FIX] Blink uses Shape 2 (Arc) with +Bend (n shape)
            float finalShape = (blinkPhase == 2) ? 2.0f : p.shapeID;
            float finalBend  = (blinkPhase == 2) ? 0.6f : p.bend; 
            
            SetShaderValue(shader, locShape, &finalShape, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locBend,  &finalBend, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locThick, &p.thickness, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locPupil, &p.pupilSize, SHADER_UNIFORM_FLOAT);

            Rectangle drawRect = rect;
            drawRect.x += sLookX.val;
            drawRect.y += sLookY.val;
            
            float oldW = drawRect.width;
            float oldH = drawRect.height;
            drawRect.width  *= sScaleX.val;
            drawRect.height *= sScaleY.val;
            drawRect.x += (oldW - drawRect.width) * 0.5f;
            drawRect.y += (oldH - drawRect.height) * 0.5f;

            rlBegin(RL_QUADS);
                rlColor4ub(c.r, c.g, c.b, c.a);
                rlTexCoord2f(0.0f, 0.0f); rlVertex2f(drawRect.x, drawRect.y);
                rlTexCoord2f(0.0f, 1.0f); rlVertex2f(drawRect.x, drawRect.y + drawRect.height);
                rlTexCoord2f(1.0f, 1.0f); rlVertex2f(drawRect.x + drawRect.width, drawRect.y + drawRect.height);
                rlTexCoord2f(1.0f, 0.0f); rlVertex2f(drawRect.x + drawRect.width, drawRect.y);
            rlEnd();
        EndShaderMode();
    }
};