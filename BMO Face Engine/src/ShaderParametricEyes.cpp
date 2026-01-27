// ShaderParametricEyes.cpp
// REFINED: Multi-Shader Architecture (Eye, Brow, Tears)
// Preserves Physics, Scaling, and Color Overrides

#include "raylib.h"
#include "rlgl.h"
#include <cmath>
#include <string>
#include <vector>

// --------------------------------------------------------
// PHYSICS STRUCTS
// --------------------------------------------------------
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

// --------------------------------------------------------
// EYE PARAMETERS
// --------------------------------------------------------
struct EyeParams {
    // --- MAIN EYE PARAMS ---
    float eyeShapeID = 0.0f;     // 0-8=Std, 9=Kawaii, 10=Shocked
    float bend = 0.0f;        
    float thickness = 4.0f;   
    float pupilSize = 0.0f;   
    
    // --- NEW SURFACE FX (For Angry/Shocked Sprites) ---
    float stressLevel = 0.0f;    // 0-1: Angry lines
    float gloomLevel = 0.0f;     // 0-1: Shocked vertical lines
    int distortMode = 0;         // 1 = Squash/Stretch distortion

    // --- ELEMENT TOGGLES ---
    bool showBrow = false;
    bool showTears = false;
    bool showBlush = false;

    // --- ELEMENT DETAILS ---
    float eyebrowType = 0.0f; 
    float eyebrowY = 0.0f;
    float eyebrowLength = 1.0f;  // New: Control brow width
    
    float tearsLevel = 0.0f;
    int tearMode = 0;            // 0=Drip, 1=Wail

    // --- EXTRAS ---
    float spiralSpeed = -1.2f;

    // --- LAYOUT & PHYSICS TARGETS ---
    float lookX = 0.0f;
    float lookY = 62.50f;
    float scaleX = 1.0f;      
    float scaleY = 1.0f;      
    float spacing = 612.0f;   
};

// --------------------------------------------------------
// MAIN CLASS
// --------------------------------------------------------
struct ParametricEyes {
    // SHADERS (Split Architecture)
    Shader shEye;
    Shader shBrow;
    Shader shTears;

    // COLORS
    Color starEyesColour = { 255, 184, 0, 255 };
    Color heartEyesColour = { 220, 1, 1, 255};
    Color blushColour = { 255, 105, 180, 150 }; // Soft Pink
    Color tearColour = { 100, 180, 255, 220 };  // Soft Blue

    // PHYSICS SPRINGS
    Spring sScaleX = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f}; 
    Spring sScaleY = {1.0f, 0.0f, 1.0f, 600.0f, 0.5f};
    Spring sLookX  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f}; 
    Spring sLookY  = {0.0f, 0.0f, 0.0f, 120.0f, 0.6f};
    
    // BLINK STATE
    float blinkTimer = 0.0f;
    float nextBlinkTime = 3.0f;
    bool isBlinking = false;
    int blinkPhase = 0; 
    bool usePhysics = true; 

    // SHADER LOCATIONS (Cached for performance)
    // Eye Shader
    int locResEye, locTimeEye, locColorEye;
    int locShape, locBend, locThick, locPupil, locSpiral, locDistort, locStress, locGloom;
    
    // Brow Shader
    int locResBrow, locColorBrow;
    int locBrowType, locBrowBend, locBrowThick, locBrowLen;

    // Tears Shader
    int locResTear, locTimeTear;
    int locTearLevel, locTearMode, locShowBlush, locBlushCol, locTearCol;


    // ----------------------------------------------------
    // INIT
    // ----------------------------------------------------
    void Init() {
        // Load the 3 separate shaders
        // Assuming they are in the same directory as the executable or src/
        shEye   = LoadShaderOrFallback("eyes_es.fs");
        shBrow  = LoadShaderOrFallback("brow_es.fs");
        shTears = LoadShaderOrFallback("tears_es.fs");

        // --- GET LOCATIONS (EYE) ---
        locResEye   = GetShaderLocation(shEye, "uResolution");
        locTimeEye  = GetShaderLocation(shEye, "uTime");
        locColorEye = GetShaderLocation(shEye, "uColor");
        locShape    = GetShaderLocation(shEye, "uShapeID");
        locBend     = GetShaderLocation(shEye, "uBend");
        locThick    = GetShaderLocation(shEye, "uThickness");
        locPupil    = GetShaderLocation(shEye, "uPupilSize");
        locSpiral   = GetShaderLocation(shEye, "uSpiralSpeed");
        locDistort  = GetShaderLocation(shEye, "uDistortMode");
        locStress   = GetShaderLocation(shEye, "uStressLevel");
        locGloom    = GetShaderLocation(shEye, "uGloomLevel");

        // --- GET LOCATIONS (BROW) ---
        locResBrow   = GetShaderLocation(shBrow, "uResolution");
        locColorBrow = GetShaderLocation(shBrow, "uColor");
        locBrowType  = GetShaderLocation(shBrow, "uEyebrowType");
        locBrowBend  = GetShaderLocation(shBrow, "uBend");
        locBrowThick = GetShaderLocation(shBrow, "uThickness");
        locBrowLen   = GetShaderLocation(shBrow, "uEyeBrowLength");

        // --- GET LOCATIONS (TEARS) ---
        locResTear   = GetShaderLocation(shTears, "uResolution");
        locTimeTear  = GetShaderLocation(shTears, "uTime");
        locTearLevel = GetShaderLocation(shTears, "uTearsLevel");
        locTearMode  = GetShaderLocation(shTears, "uTearMode");
        locShowBlush = GetShaderLocation(shTears, "uShowBlush");
        locBlushCol  = GetShaderLocation(shTears, "uBlushColor");
        locTearCol   = GetShaderLocation(shTears, "uTearColor");
    }

    // Helper to find shader path (kept from your code)
    Shader LoadShaderOrFallback(const char* filename) {
        std::string path = std::string(GetApplicationDirectory()) + filename;
        if (!FileExists(path.c_str())) path = filename;
        if (!FileExists(path.c_str())) path = std::string("src/") + filename;
        return LoadShader(0, path.c_str());
    }
    
    void Unload() {
        UnloadShader(shEye);
        UnloadShader(shBrow);
        UnloadShader(shTears);
    }

    // ----------------------------------------------------
    // UPDATE (Physics & Blink)
    // ----------------------------------------------------
    void Update(float dt, EyeParams& params) {
        if (!usePhysics) {
            // DIRECT MAPPING (For Editor)
            sScaleX.val = params.scaleX;
            sScaleY.val = params.scaleY;
            sLookX.val = params.lookX;
            sLookY.val = params.lookY;
            blinkPhase = 0; 
            return;
        }

        // --- AUTO BLINK LOGIC ---
        // Only blink if shape is Dot (0), Star (4), Heart (5), etc.
        bool canBlink = (params.eyeShapeID < 0.5f || params.eyeShapeID > 3.5f);
        
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
            sScaleY.val = 1.0f; sScaleY.target = 1.0f; 
            if (blinkTimer > 0.10f) { blinkPhase = 3; sScaleY.val = 0.0f; }
        }
        else if (blinkPhase == 3) { // Opening
            sScaleY.target = params.scaleY; 
            if (sScaleY.val > params.scaleY * 0.9f) blinkPhase = 0;
        }

        // Apply Targets
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

    // ----------------------------------------------------
    // DRAW
    // ----------------------------------------------------

    // Helper to draw a specific layer into a rect
    // layerType: 0=Eye, 1=Brow, 2=Tears
    void DrawLayer(Rectangle rect, EyeParams p, Color color, int layerType) {
        Shader* targetShader = nullptr;
        float time = (float)GetTime();
        float res[2] = { rect.width, rect.height };
        float colVec[4] = { color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f };

        // Select Shader
        if (layerType == 0) targetShader = &shEye;
        else if (layerType == 1) targetShader = &shBrow;
        else if (layerType == 2) targetShader = &shTears;
        
        if (!targetShader) return;

        rlSetTexture(0);
        BeginShaderMode(*targetShader);

        // --- SEND UNIFORMS ---
        if (layerType == 0) { // EYE
            SetShaderValue(shEye, locResEye, res, SHADER_UNIFORM_VEC2);
            SetShaderValue(shEye, locTimeEye, &time, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locColorEye, colVec, SHADER_UNIFORM_VEC4);
            
            // Handle Blink Override (Phase 2 -> Shape 1.0 Line)
            float finalShape = (blinkPhase == 2) ? 1.0f : p.eyeShapeID;
            
            SetShaderValue(shEye, locShape, &finalShape, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locBend, &p.bend, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locThick, &p.thickness, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locPupil, &p.pupilSize, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locSpiral, &p.spiralSpeed, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locDistort, &p.distortMode, SHADER_UNIFORM_INT);
            SetShaderValue(shEye, locStress, &p.stressLevel, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locGloom, &p.gloomLevel, SHADER_UNIFORM_FLOAT);
        }
        else if (layerType == 1) { // BROW
            SetShaderValue(shBrow, locResBrow, res, SHADER_UNIFORM_VEC2);
            SetShaderValue(shBrow, locColorBrow, colVec, SHADER_UNIFORM_VEC4);
            
            SetShaderValue(shBrow, locBrowType, &p.eyebrowType, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow, locBrowBend, &p.bend, SHADER_UNIFORM_FLOAT); // Re-use bend
            SetShaderValue(shBrow, locBrowThick, &p.thickness, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow, locBrowLen, &p.eyebrowLength, SHADER_UNIFORM_FLOAT);
        }
        else if (layerType == 2) { // TEARS
            SetShaderValue(shTears, locResTear, res, SHADER_UNIFORM_VEC2);
            SetShaderValue(shTears, locTimeTear, &time, SHADER_UNIFORM_FLOAT);
            
            SetShaderValue(shTears, locTearLevel, &p.tearsLevel, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shTears, locTearMode, &p.tearMode, SHADER_UNIFORM_INT);
            
            int blushToggle = p.showBlush ? 1 : 0;
            SetShaderValue(shTears, locShowBlush, &blushToggle, SHADER_UNIFORM_INT);
            
            // Upload specific accessory colors
            float pink[4] = { blushColour.r/255.0f, blushColour.g/255.0f, blushColour.b/255.0f, blushColour.a/255.0f };
            float blue[4] = { tearColour.r/255.0f, tearColour.g/255.0f, tearColour.b/255.0f, tearColour.a/255.0f };
            SetShaderValue(shTears, locBlushCol, pink, SHADER_UNIFORM_VEC4);
            SetShaderValue(shTears, locTearCol, blue, SHADER_UNIFORM_VEC4);
        }

        // --- DRAW QUAD ---
        rlBegin(RL_QUADS);
            rlColor4ub(color.r, color.g, color.b, color.a);
            rlTexCoord2f(0.0f, 0.0f); rlVertex2f(rect.x, rect.y);
            rlTexCoord2f(0.0f, 1.0f); rlVertex2f(rect.x, rect.y + rect.height);
            rlTexCoord2f(1.0f, 1.0f); rlVertex2f(rect.x + rect.width, rect.y + rect.height);
            rlTexCoord2f(1.0f, 0.0f); rlVertex2f(rect.x + rect.width, rect.y);
        rlEnd();
        EndShaderMode();
    }

    // Draws all 3 layers for a single eye stack
    void DrawSingleEyeStack(Rectangle rect, EyeParams p, Color c) {
        // 1. CALCULATE PHYSICS TRANSFORM
        Rectangle drawRect = rect;
        drawRect.x += sLookX.val;
        drawRect.y += sLookY.val;
        
        float oldW = drawRect.width;
        float oldH = drawRect.height;
        drawRect.width  *= sScaleX.val;
        drawRect.height *= sScaleY.val;
        drawRect.x += (oldW - drawRect.width) * 0.5f;
        drawRect.y += (oldH - drawRect.height) * 0.5f;

        // 2. LAYER 1: MAIN EYE
        DrawLayer(drawRect, p, c, 0);

        // 3. LAYER 2: EYEBROW (If enabled)
        if (p.showBrow) {
            DrawLayer(drawRect, p, BLACK, 1);
        }

        // 4. LAYER 3: TEARS / BLUSH (If enabled)
        // (Note: Color passed here is ignored by shader in favor of internal uniform colors)
        if (p.showTears || p.showBlush) {
            DrawLayer(drawRect, p, WHITE, 2);
        }
    }
       
    // Main Draw Call - Draws BOTH eyes
    void Draw(Vector2 centerPos, EyeParams p, Color c) {
        float eyeSize = 100.0f; 
        float eyeHeight  = 150.0f;  

        // [FIX] TALLER RECTANGLES (Kept from your original code)
        float drawWidth  = eyeSize;
        float drawHeight = eyeSize * 2.5f;

        Rectangle leftRect = { 
            centerPos.x - (p.spacing/2) - (drawWidth/2), 
            centerPos.y - eyeHeight - (drawHeight/2), 
            drawWidth, drawHeight 
        };

        Rectangle rightRect = { 
            centerPos.x + (p.spacing/2) - (drawWidth/2), 
            centerPos.y - eyeHeight - (drawHeight/2), 
            drawWidth, drawHeight 
        };

        // --- COLOR OVERRIDES ---
        // Star Eyes -> Gold
        if((p.eyeShapeID > 3.5 && p.eyeShapeID < 4.5) || p.eyeShapeID > 7.5)
            c = starEyesColour;
        
        // Heart Eyes -> Red
        if(p.eyeShapeID > 4.5 && p.eyeShapeID < 5.5)
            c = heartEyesColour;    

        // Draw Stack
        DrawSingleEyeStack(leftRect, p, c);
        DrawSingleEyeStack(rightRect, p, c);
    }
};