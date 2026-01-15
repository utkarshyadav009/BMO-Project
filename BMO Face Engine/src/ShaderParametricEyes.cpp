#include "raylib.h"
#include "rlgl.h"
#include <cmath>

// Physics Helper
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
    // Visual Targets (Where we WANT to be)
    float shapeID = 0.0f;    
    float bend = 0.0f;       
    float thickness = 4.0f; 
    float pupilSize = 0.0f;  
    
    // Physics Targets (Look direction)
    float targetLookX = 0.0f;
    float targetLookY = 0.0f;
};

struct ParametricEyes {
    Shader shader;
    
    // PHYSICS STATE
    Spring sScaleX = {1.0f, 0.0f, 1.0f, 200.0f, 0.5f}; // Fast snap
    Spring sScaleY = {1.0f, 0.0f, 1.0f, 200.0f, 0.5f};
    Spring sLookX  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f}; // Smooth follow
    Spring sLookY  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};
    
    // BLINK STATE
    float blinkTimer = 0.0f;
    float nextBlinkTime = 3.0f;
    bool isBlinking = false;
    int blinkPhase = 0; // 0=Idle, 1=Closing, 2=Closed, 3=Opening

    // SHADER LOCS
    int locRes, locColor;
    int locShape, locBend, locThick, locPupil;

    void Init() {
        shader = LoadShader(0, "eyes_es.fs");
        locRes   = GetShaderLocation(shader, "uResolution");
        locColor = GetShaderLocation(shader, "uColor");
        locShape = GetShaderLocation(shader, "uShapeID");
        locBend  = GetShaderLocation(shader, "uBend");
        locThick = GetShaderLocation(shader, "uThickness");
        locPupil = GetShaderLocation(shader, "uPupilSize");
    }

    void Update(float dt, EyeParams& params) {
        // 1. AUTO BLINK LOGIC
        // Only blink if we are in "Dot" mode (ID 0)
        if (params.shapeID < 0.5f) {
            blinkTimer += dt;
            
            // Trigger Blink
            if (blinkTimer > nextBlinkTime && blinkPhase == 0) {
                blinkPhase = 1;
                blinkTimer = 0.0f;
                nextBlinkTime = (float)GetRandomValue(20, 50) / 10.0f; // Random 2-5s
            }
        }

        // 2. BLINK ANIMATION STATE MACHINE
        float currentShape = params.shapeID;
        
        if (blinkPhase == 1) { // CLOSING
            sScaleY.target = 0.0f; // Squash down
            if (sScaleY.val < 0.2f) {
                blinkPhase = 2; // Fully closed
                blinkTimer = 0.0f;
            }
        } 
        else if (blinkPhase == 2) { // CLOSED (Hold line)
            currentShape = 1.0f; // Force Shape 1 (Line)
            sScaleY.val = 1.0f;  // Reset scale for the line shape
            sScaleY.target = 1.0f;
            
            if (blinkTimer > 0.15f) { // Keep closed for 0.15s
                blinkPhase = 3; 
                sScaleY.val = 0.0f; // Squash ready to open
            }
        }
        else if (blinkPhase == 3) { // OPENING
            sScaleY.target = 1.0f; // Pop up
            if (sScaleY.val > 0.9f) {
                blinkPhase = 0; // Done
            }
        }

        // 3. APPLY PHYSICS
        sLookX.target = params.targetLookX;
        sLookY.target = params.targetLookY;

        // Add some "life" (breathing)
        if (blinkPhase == 0) {
             float breath = sinf((float)GetTime() * 2.0f) * 0.02f;
             sScaleY.target = 1.0f + breath;
             sScaleX.target = 1.0f - breath;
        }

        sScaleX.Update(dt);
        sScaleY.Update(dt);
        sLookX.Update(dt);
        sLookY.Update(dt);

        // Override visual shape if blinking
        if (blinkPhase == 2) params.shapeID = 1.0f; 
    }

    void DrawEye(Rectangle rect, EyeParams p, Color c) {
        BeginShaderMode(shader);
            float res[2] = { rect.width, rect.height };
            SetShaderValue(shader, locRes, res, SHADER_UNIFORM_VEC2);
            
            float col[4] = { c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f };
            SetShaderValue(shader, locColor, col, SHADER_UNIFORM_VEC4);

            // Use internal physics values, not just raw inputs
            float finalShape = (blinkPhase == 2) ? 1.0f : p.shapeID;
            
            SetShaderValue(shader, locShape, &finalShape, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locBend,  &p.bend, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locThick, &p.thickness, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locPupil, &p.pupilSize, SHADER_UNIFORM_FLOAT);

            // Calculate Transforms with Physics
            Rectangle drawRect = rect;
            
            // Apply Spring Offsets
            drawRect.x += sLookX.val;
            drawRect.y += sLookY.val;
            
            // Apply Spring Scale
            // Center scaling: Move x/y to compensate for width/height change
            float oldW = drawRect.width;
            float oldH = drawRect.height;
            drawRect.width  *= sScaleX.val;
            drawRect.height *= sScaleY.val;
            drawRect.x += (oldW - drawRect.width) * 0.5f;
            drawRect.y += (oldH - drawRect.height) * 0.5f;

            // DRAW QUAD
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