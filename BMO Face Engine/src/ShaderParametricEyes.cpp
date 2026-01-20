// ShaderParametricEyes.cpp
// REFINED: Added Direct Control for Editor & Scale Support

#include "raylib.h"
#include "rlgl.h"
#include <cmath>
#include <string>
#include <vector>

struct Spring {
    float val, vel, target;
    float stiffness, damping;
    
    void Reset(float initial) {
        val = initial; target = initial; vel = 0;
    }
    
    void Update(float dt) {
        float f = stiffness * (target - val);
        vel = (vel + f * dt) * damping;
        val += vel * dt;
    }
};

struct EyeParams {
    float shapeID = 0.0f;     // 0=Dot, 1=Line, 2=Arc, 3=Cross, 4=Star, 5=Heart, 6=Spiral, 7=Chevron
    float bend = 0.0f;        
    float thickness = 4.0f;   
    float pupilSize = 0.0f;   // Used for Pupil OR Highlight size
    
    // NEW PARAMS
    float eyebrowType = 0.0f; // 0=None, 1=Angry, 2=Sad, 3=Neutral
    float eyebrowY = 0.0f;    // Offset Y
    float tears = 0.0f;       // 0.0 to 1.0
    
    float lookX = 0.0f;
    float lookY = 0.0f;
    float scaleX = 1.0f;      
    float scaleY = 1.0f;      
    float spacing = 200.0f;   
    float spiralSpeed = 1.2f;
};

struct ParametricEyes {
    Shader shader;
    
    // PHYSICS SPRINGS
    Spring sScaleX = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f}; 
    Spring sScaleY = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
    Spring sLookX  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f}; 
    Spring sLookY  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};
    
    float blinkTimer = 0.0f;
    float nextBlinkTime = 3.0f;
    bool isBlinking = false;
    int blinkPhase = 0; 

    // Editor Mode: Bypasses physics for instant feedback
    bool usePhysics = true; 

    // Shader Locations
    int locRes, locColor, locShape, locBend, locThick, locPupil;
    int locEyebrow, locEyebrowY, locTears;
    int locDeltaTime; int locSpiralSpeed;

    void Init() {
        // Try multiple paths to find the shader
        const char* filename = "eyes_es.fs";
        std::string path = std::string(GetApplicationDirectory()) + filename;
        if (!FileExists(path.c_str())) path = filename;
        if (!FileExists(path.c_str())) path = "src/eyes_es.fs"; // Check src folder

        shader = LoadShader(0, path.c_str());
        locRes   = GetShaderLocation(shader, "uResolution");
        locColor = GetShaderLocation(shader, "uColor");
        locShape = GetShaderLocation(shader, "uShapeID");
        locBend  = GetShaderLocation(shader, "uBend");
        locThick = GetShaderLocation(shader, "uThickness");
        locPupil = GetShaderLocation(shader, "uPupilSize");

        locEyebrow  = GetShaderLocation(shader, "uEyebrow");
        locEyebrowY = GetShaderLocation(shader, "uEyebrowY");
        locTears    = GetShaderLocation(shader, "uTears");

        locDeltaTime = GetShaderLocation(shader, "uTime");
        locSpiralSpeed = GetShaderLocation(shader, "uSpiralSpeed");
    }
    
    void Unload() {
        UnloadShader(shader);
    }

    void Update(float dt, EyeParams& params) {
        if (!usePhysics) {
            // DIRECT MAPPING (For Editor)
            sScaleX.val = params.scaleX;
            sScaleY.val = params.scaleY;
            sLookX.val = params.lookX;
            sLookY.val = params.lookY;
            blinkPhase = 0; // Disable blink in editor unless requested
            return;
        }

        // --- AUTO BLINK LOGIC ---
        // Only blink if shape is Dot (0) or Heart (5) or Star (4)
        // Don't blink if already in "Line" or "Cross" mode
        bool canBlink = (params.shapeID < 0.5f || params.shapeID > 3.5f);
        
        if (canBlink) {
            blinkTimer += dt;
            if (blinkTimer > nextBlinkTime && blinkPhase == 0) {
                blinkPhase = 1;
                blinkTimer = 0.0f;
                nextBlinkTime = (float)GetRandomValue(20, 50) / 10.0f;
            }
        } else {
            blinkPhase = 0; 
        }

        // Blink Animation State Machine
        if (blinkPhase == 1) { // Closing
            sScaleY.target = 0.0f; 
            if (sScaleY.val < 0.2f) { blinkPhase = 2; blinkTimer = 0.0f; }
        } 
        else if (blinkPhase == 2) { // Closed (HOLD)
            sScaleY.val = 1.0f; sScaleY.target = 1.0f; // Reset scale but change shape to line
            if (blinkTimer > 0.10f) { blinkPhase = 3; sScaleY.val = 0.0f; }
        }
        else if (blinkPhase == 3) { // Opening
            sScaleY.target = params.scaleY; 
            if (sScaleY.val > params.scaleY * 0.9f) blinkPhase = 0;
        }

        // Apply Targets
        if (blinkPhase == 0) {
             // Idle "breathing" only if scale is near 1.0
             float breath = (params.scaleY > 0.8f) ? sinf((float)GetTime() * 2.0f) * 0.02f : 0.0f;
             sScaleY.target = params.scaleY + breath;
             sScaleX.target = params.scaleX - breath;
        }

        sLookX.target = params.lookX;
        sLookY.target = params.lookY;

        sScaleX.Update(dt); sScaleY.Update(dt);
        sLookX.Update(dt);  sLookY.Update(dt);
    }

    // Helper to draw a single eye
    void DrawSingleEye(Rectangle rect, EyeParams p, Color c) {
        rlSetTexture(0);
        BeginShaderMode(shader);
            // Apply Physics transforms
            Rectangle drawRect = rect;
            drawRect.x += sLookX.val;
            drawRect.y += sLookY.val;
            
            // Apply Scaling from center
            float oldW = drawRect.width;
            float oldH = drawRect.height;
            drawRect.width  *= sScaleX.val;
            drawRect.height *= sScaleY.val;
            drawRect.x += (oldW - drawRect.width) * 0.5f;
            drawRect.y += (oldH - drawRect.height) * 0.5f;


            float res[2] = { rect.width, rect.height };
            SetShaderValue(shader, locRes, res, SHADER_UNIFORM_VEC2);
            float col[4] = { c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f };
            SetShaderValue(shader, locColor, col, SHADER_UNIFORM_VEC4);

            // If Blinking (Phase 2), force Shape to Line/Arc
            float finalShape = (blinkPhase == 2) ? 1.0f : p.shapeID; // 1.0 is Line
            
            // --- CORE PARAMS ---
            SetShaderValue(shader, locShape, &finalShape, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locBend,  &p.bend, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locThick, &p.thickness, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locPupil, &p.pupilSize, SHADER_UNIFORM_FLOAT);

            // --- [FIX] SEND NEW PARAMS TO GPU ---
            SetShaderValue(shader, locEyebrow,  &p.eyebrowType, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locEyebrowY, &p.eyebrowY, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locTears,    &p.tears, SHADER_UNIFORM_FLOAT);

            //Time parameter to GPU 
             float totalTime = (float)GetTime(); 
            SetShaderValue(shader, locDeltaTime,    &totalTime, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shader, locSpiralSpeed,    &p.spiralSpeed, SHADER_UNIFORM_FLOAT);


            

            rlBegin(RL_QUADS);
                rlColor4ub(c.r, c.g, c.b, c.a);
                rlTexCoord2f(0.0f, 0.0f); rlVertex2f(drawRect.x, drawRect.y);
                rlTexCoord2f(0.0f, 1.0f); rlVertex2f(drawRect.x, drawRect.y + drawRect.height);
                rlTexCoord2f(1.0f, 1.0f); rlVertex2f(drawRect.x + drawRect.width, drawRect.y + drawRect.height);
                rlTexCoord2f(1.0f, 0.0f); rlVertex2f(drawRect.x + drawRect.width, drawRect.y);
            rlEnd();
        EndShaderMode();
    }
    
    // Main Draw Call - Draws BOTH eyes based on center position
    void Draw(Vector2 centerPos, EyeParams p, Color c) {
        float eyeSize = 100.0f; // Base size of the eye box
        
        Rectangle leftRect = { 
            centerPos.x - (p.spacing/2) - (eyeSize/2), 
            centerPos.y - (eyeSize/2), 
            eyeSize, eyeSize 
        };

        Rectangle rightRect = { 
            centerPos.x + (p.spacing/2) - (eyeSize/2), 
            centerPos.y - (eyeSize/2), 
            eyeSize, eyeSize 
        };

        DrawSingleEye(leftRect, p, c);
        DrawSingleEye(rightRect, p, c);
    }
};