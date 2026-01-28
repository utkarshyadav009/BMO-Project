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
    float eyeThickness = 4.0f;   
    float pupilSize = 0.0f;   
    float eyeSide = 0.0f;
    
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
    float eyebrowThickness = 4.0f;
    float eyebrowY = 0.0f;
    float eyebrowX = 0.0f;
    float eyebrowLength = 1.0f;
    float browScale = 1.0f;      
    float browSide = 1.0f;        // New: 1.0 for left, -1.0 for right (mirrors the brow)
    float browAngle = 0.0f;       // New: Angle to rotate the brow
    float browBend = 0.0f;        // Re-use bend for brow arching
    float browBendOffset = 0.85f;  // New: Offset for brow bend
    bool useLowerBrow = false;   

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
    int locResEye, locTimeEye, locColorEye, locEyeSide;
    int locShape, locBend, locEyeThick, locPupil, locSpiral, locDistort, locStress, locGloom;
    
    // Brow Shader
    int locResBrow, locColorBrow;
    int locBrowType, locBrowBend, locBrowThick, locBrowLen, locBrowY, locBrowSide, locBrowAngle, locBrowBendOffset;

    // Tears Shader
    int locResTear, locTimeTear;
    int locTearLevel, locTearMode, locShowBlush, locBlushCol, locTearCol;


    // HOT RELOAD DATA
    long eyeFragTime = 0;
    long tearFragTime = 0;
    long browFragTime = 0;
    std::string eyeFragPath = "eyes_es.fs"; // Store the path
    std::string tearFragPath = "tears_es.fs";
    std::string browFragPath = "brow_es.fs";

    float hotReloadTimer = 0.0f;

    // ----------------------------------------------------
    // INIT
    // ----------------------------------------------------
    void Init() {
        // Load the 3 separate shaders
        // Assuming they are in the same directory as the executable or src/
        //shEye   = LoadShaderOrFallback("eyes_es.fs");
        //shBrow  = LoadShaderOrFallback("brow_es.fs");
        //shTears = LoadShaderOrFallback("tears_es.fs");

        //Loading Eye Shader (Hot Reload Enabled)
        if (FileExists("eyes_es.fs")) eyeFragPath = "eyes_es.fs";
        else if (FileExists("src/eyes_es.fs")) eyeFragPath = "src/eyes_es.fs";
        else eyeFragPath = "eyes_es.fs"; // Default fallback
        shEye = LoadShader(0, eyeFragPath.c_str());
        eyeFragTime = GetFileModTime(eyeFragPath.c_str());
        GetEyeLocations();

        //Loading Brow Shader (Hot Reload Enabled)
        if (FileExists("brow_es.fs")) browFragPath = "brow_es.fs";
        else if (FileExists("src/brow_es.fs")) browFragPath = "src/brow_es.fs";
        else browFragPath = "brow_es.fs"; // Default fallback
        shBrow = LoadShader(0, browFragPath.c_str());
        browFragTime = GetFileModTime(browFragPath.c_str());
        GetBrowLocations();

        //Loading Tears Shader (Hot Reload Enabled)
        if (FileExists("tears_es.fs")) tearFragPath = "tears_es.fs";
        else if (FileExists("src/tears_es.fs")) tearFragPath = "src/tears_es.fs";
        else tearFragPath = "tears_es.fs"; // Default fallback
        shTears = LoadShader(0, tearFragPath.c_str());
        tearFragTime = GetFileModTime(tearFragPath.c_str());
        GetTearLocations();

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

        hotReloadTimer += dt;
        if (hotReloadTimer > 1.0f) {
            CheckHotReload();
            hotReloadTimer = 0.0f;
        }

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
    void DrawLayer(Rectangle rect, Vector2 sourceRes,EyeParams p, Color color, int layerType) {
        Shader* targetShader = nullptr;
        float time = (float)GetTime();
        float res[2] = { sourceRes.x, sourceRes.y };
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
            SetShaderValue(shEye, locEyeThick, &p.eyeThickness, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locPupil, &p.pupilSize, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locEyeSide, &p.eyeSide, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locSpiral, &p.spiralSpeed, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locDistort, &p.distortMode, SHADER_UNIFORM_INT);
            SetShaderValue(shEye, locStress, &p.stressLevel, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shEye, locGloom, &p.gloomLevel, SHADER_UNIFORM_FLOAT);
        }
        else if (layerType == 1) { // BROW
            SetShaderValue(shBrow, locResBrow, res, SHADER_UNIFORM_VEC2);
            SetShaderValue(shBrow, locColorBrow, colVec, SHADER_UNIFORM_VEC4);
            
            SetShaderValue(shBrow, locBrowType, &p.eyebrowType, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow, locBrowBend, &p.browBend, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow, locBrowThick, &p.eyebrowThickness, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow, locBrowLen, &p.eyebrowLength, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow, locBrowSide, &p.browSide, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow, locBrowAngle, &p.browAngle, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shBrow, locBrowBendOffset, &p.browBendOffset, SHADER_UNIFORM_FLOAT);
        }
        else if (layerType == 2) { // TEARS
            SetShaderValue(shTears, locResTear, res, SHADER_UNIFORM_VEC2);
            SetShaderValue(shTears, locTimeTear, &time, SHADER_UNIFORM_FLOAT);
            
            SetShaderValue(shTears, locTearLevel, &p.tearsLevel, SHADER_UNIFORM_FLOAT);
            SetShaderValue(shTears, locTearMode, &p.tearMode, SHADER_UNIFORM_INT);
            SetShaderValue(shTears, locEyeSide, &p.eyeSide, SHADER_UNIFORM_FLOAT);
            
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

        Vector2 originalRes = { rect.width, rect.height }; // Save this!

        // 1. CALCULATE PHYSICS TRANSFORM
        Rectangle eyeRect = rect;
        eyeRect.x += sLookX.val;
        eyeRect.y += sLookY.val;
        
        float oldW = eyeRect.width;
        float oldH = eyeRect.height;
        eyeRect.width  *= sScaleX.val;
        eyeRect.height *= sScaleY.val;
        eyeRect.x += (oldW - eyeRect.width) * 0.5f;
        eyeRect.y += (oldH - eyeRect.height) * 0.5f;
        

        // 2. LAYER 1: MAIN EYE
        DrawLayer(eyeRect, originalRes, p, c, 0);
        //DrawRectangleLinesEx(eyeRect, 2.0f, { 255, 0, 0, 150 }); 
        //DrawRectangleRec(eyeRect, { 255, 0, 0, 40 }); // Faint fill

        // 3. LAYER 2: EYEBROW (If enabled)
        //Offsetting the brow ret so that it doesn't share or use the same
        // resolution as the eye (makes it less squashed)
        if (p.showBrow) {

            Rectangle browRect = rect;

            browRect.width  = rect.width * 2.0f * p.browScale * p.eyebrowLength; 
            browRect.height = rect.height * 0.5f * p.browScale;

            browRect.x = rect.x + sLookX.val; 
            browRect.y = rect.y + sLookY.val - (rect.height * 0.5f); // Position above eye

            browRect.x += (rect.width - browRect.width) * 0.5f;
            browRect.y += (rect.height - browRect.height) * 0.5f;
            browRect.x += p.eyebrowX*20.f* p.browSide; // Apply X offset
            browRect.y += p.eyebrowY*20.f; // Apply Y offset
            Vector2 browRes = { browRect.width, browRect.height };

            DrawLayer(browRect, browRes, p, BLACK, 1);
            if (p.useLowerBrow) 
            {
                EyeParams lowerP = p;
                lowerP.eyebrowY = 0.0f;
                lowerP.bend = -p.bend;
                lowerP.eyebrowLength = 1.37f;
                lowerP.eyebrowThickness = 7.15f;
            
                // 1. Calculate the exact bottom pixel of the eye
                Vector2 eyeCenter = { 
                    eyeRect.x + eyeRect.width * 0.5f,
                    eyeRect.y + eyeRect.height * 0.5f 
                };

                // Use 0.5f to get the radius (distance from center to bottom)
                float ryEye = eyeRect.height * 0.5f; 
                float eyeBottomY = eyeCenter.y + ryEye;
            
                Rectangle lowerRect = eyeRect;
                lowerRect.width  = eyeRect.width  * 2.5f * p.browScale * lowerP.eyebrowLength;
                lowerRect.height = eyeRect.height * (1.0f + fabsf(p.bend));

                // Center X
                lowerRect.x = eyeRect.x + (eyeRect.width - lowerRect.width) * 0.5f;
            
                // --- THE CRITICAL FIX ---
                // Currently you are doing: lowerRect.y = eyeBottomY;
                // You need to Move it UP by half its height so the 'Line' sits on the edge
                lowerRect.y = eyeBottomY - (lowerRect.height * 0.53f);

                // 2. Optional Nudge
                // If the line feels too thick and sits "outside" the eye, 
                // subtract a tiny bit more to push it UP into the eye.
                lowerRect.y -= eyeRect.height * 0.3f;

                float myFixedOffset = 48.6235f;
                lowerRect.x += myFixedOffset;
                std::cout<<"Lower Brow X Offset Applied: "<< myFixedOffset <<std::endl; 

                
                //lowerRect.x += p.eyebrowX*20.f; // Apply X offset
                //std::cout<<"Lower Brow X: "<< lowerRect.x <<std::endl;

                Vector2 lowerRes = { lowerRect.width, lowerRect.height };
                DrawLayer(lowerRect, lowerRes, lowerP, BLACK, 1);

                // Debug visuals
                //DrawRectangleLinesEx(lowerRect, 2.0f, { 0, 255, 0, 150 });
            }
            

            // DrawRectangleLinesEx(browRect, 2.0f, { 255, 255, 0, 150 });
            // DrawRectangleRec(browRect, { 255, 255, 0, 40 }); // Faint fill
        }

        // 4. LAYER 3: TEARS / BLUSH (If enabled)
        // (Note: Color passed here is ignored by shader in favor of internal uniform colors)
        if (p.showTears || p.showBlush) {

            // Use the same width as the incoming rect (drawWidth from caller)
            float tearWidth = rect.width+130.0f;

            // Calculate how much space is left from the eye to the bottom of the screen
            // + a buffer (e.g. 100px) to ensure tears fall completely off-screen
            float startY = rect.y + (rect.height * 0.5f);
            float tearHeight = (float)GetScreenHeight() - startY + 100.0f;          

            // 2. DEFINE RECT (Anchored at Eye Center)
            Rectangle tearRect = {
                rect.x + (rect.width * 0.5f) - (tearWidth * 0.5f), // Center X
                startY,                                            // Start Y (Pupil Center)
                tearWidth,
                tearHeight
            };

            // Small wrapper that accepts a single Rectangle and forwards to DrawLayer
            auto DrawTearRect = [&](Rectangle r) {
            Vector2 rRes = { r.width, r.height };
            DrawLayer(r, rRes, p, WHITE, 2);
            };

            // Pass only the tearRect into the wrapper
            DrawTearRect(tearRect);

            //DrawRectangleLinesEx(tearRect, 2.0f, { 255, 200, 0, 150 }); 
            //DrawRectangleRec(tearRect, { 255, 200, 0, 40 }); // Faint fill
        }
    }
       
    // Main Draw Call - Draws BOTH eyes
    void Draw(Vector2 centerPos, EyeParams p, Color c) {
        float eyeSize = 100.0f; 
        float eyeHeight  = 150.0f;  

        // [FIX] TALLER RECTANGLES (Kept from your original code)
        float drawWidth  = eyeSize*2.7f;
        float drawHeight = eyeSize*2.7f;

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
        if((p.eyeShapeID > 3.5 && p.eyeShapeID < 4.5) || (p.eyeShapeID > 7.5 && p.eyeShapeID < 8.5))
            c = starEyesColour;
        
        // Heart Eyes -> Red
        if(p.eyeShapeID > 4.5 && p.eyeShapeID < 5.5)
            c = heartEyesColour;    

        // Draw Stack
        EyeParams leftParams = p;
        leftParams.browSide = -1.0f;  // Left brow: no flip
        leftParams.eyeSide = 1.0f; // Left eye: flipped
        DrawSingleEyeStack(leftRect, leftParams, c);
        
        EyeParams rightParams = p;
        rightParams.browSide = 1.0f;  // Right brow: flipped
        rightParams.eyeSide = -1.0f;   // Right eye: normal
        DrawSingleEyeStack(rightRect, rightParams, c);
    }

    // MOVED: Extract this from Init() so we can reuse it
    void GetEyeLocations() {
        locResEye   = GetShaderLocation(shEye, "uResolution");
        locTimeEye  = GetShaderLocation(shEye, "uTime");
        locColorEye = GetShaderLocation(shEye, "uColor");
        locShape    = GetShaderLocation(shEye, "uShapeID");
        locBend     = GetShaderLocation(shEye, "uBend");
        locEyeThick = GetShaderLocation(shEye, "uThickness");
        locEyeSide  = GetShaderLocation(shEye, "uEyeSide");
        locPupil    = GetShaderLocation(shEye, "uPupilSize");
        locSpiral   = GetShaderLocation(shEye, "uSpiralSpeed");
        locDistort  = GetShaderLocation(shEye, "uDistortMode");
        locStress   = GetShaderLocation(shEye, "uStressLevel");
        locGloom    = GetShaderLocation(shEye, "uGloomLevel");
    }

    void GetBrowLocations() {
        locResBrow   = GetShaderLocation(shBrow, "uResolution");
        locColorBrow = GetShaderLocation(shBrow, "uColor");
        locBrowType  = GetShaderLocation(shBrow, "uEyebrowType");
        locBrowBend  = GetShaderLocation(shBrow, "uBend");
        locBrowThick = GetShaderLocation(shBrow, "uThickness");
        locBrowLen   = GetShaderLocation(shBrow, "uEyeBrowLength");
        locBrowSide  = GetShaderLocation(shBrow, "uBrowSide");
        locBrowAngle = GetShaderLocation(shBrow, "uAngle");
        locBrowBendOffset = GetShaderLocation(shBrow, "uBendOffset");
    }

    void GetTearLocations() {
        locResTear   = GetShaderLocation(shTears, "uResolution");
        locTimeTear  = GetShaderLocation(shTears, "uTime");
        locTearLevel = GetShaderLocation(shTears, "uTearsLevel");
        locTearMode  = GetShaderLocation(shTears, "uTearMode");
        locShowBlush = GetShaderLocation(shTears, "uShowBlush");
        locBlushCol  = GetShaderLocation(shTears, "uBlushColor");
        locTearCol   = GetShaderLocation(shTears, "uTearColor");
        locEyeSide   = GetShaderLocation(shTears, "uSide");
    }

    void CheckHotReload() {
        long eyeModTimeNow  = GetFileModTime(eyeFragPath.c_str());
        long tearModTimeNow = GetFileModTime(tearFragPath.c_str());
        long browModTimeNow = GetFileModTime(browFragPath.c_str());

        std::cout << "[HOT RELOAD] Checking shaders... Eye: " << eyeModTimeNow << " (last " << eyeFragTime << ")"
                  << " | Tears: " << tearModTimeNow << " (last " << tearFragTime << ")"
                  << " | Brow: " << browModTimeNow << " (last " << browFragTime << ")" << std::endl;

        std::ofstream logFile("hotreload_debug.log", std::ios::app);
        logFile << "[HOT RELOAD] Check called. Eye: " << eyeModTimeNow << " (last " << eyeFragTime << "), "
                << "Tears: " << tearModTimeNow << " (last " << tearFragTime << "), "
                << "Brow: " << browModTimeNow << " (last " << browFragTime << ")\n";

        bool anyChanged = (eyeModTimeNow > eyeFragTime) || (tearModTimeNow > tearFragTime) || (browModTimeNow > browFragTime);
        if (!anyChanged) {
            std::cout << "[HOT RELOAD] No changes detected." << std::endl;
            logFile << "[HOT RELOAD] No changes detected.\n";
            logFile.close();
            return;
        }

        std::cout << "[HOT RELOAD] Change detected. Attempting per-shader reload..." << std::endl;
        logFile << "[HOT RELOAD] Change detected. Attempting per-shader reload...\n";

        // --- Eye Shader ---
        if (eyeModTimeNow > eyeFragTime) {
            std::cout << "[HOT RELOAD] Reloading Eye shader from: " << eyeFragPath << std::endl;
            logFile << "[HOT RELOAD] Reloading Eye shader from: " << eyeFragPath << "\n";

            Shader newEyeShader = LoadShader(0, eyeFragPath.c_str());
            if (newEyeShader.id != rlGetShaderIdDefault()) {
                UnloadShader(shEye);
                shEye = newEyeShader;
                GetEyeLocations();
                eyeFragTime = eyeModTimeNow;

                std::cout << "[HOT RELOAD] Eye shader reloaded successfully." << std::endl;
                logFile << "[HOT RELOAD] Eye shader reloaded successfully.\n";
                TraceLog(LOG_INFO, "HOT RELOAD: Eye shader updated.");
            } else {
                std::cout << "[HOT RELOAD] Eye shader reload FAILED (compile error)." << std::endl;
                logFile << "[HOT RELOAD] Eye shader reload FAILED (compile error).\n";
                TraceLog(LOG_WARNING, "HOT RELOAD: Eye shader compile failed.");
            }
        }

        // --- Tears Shader ---
        if (tearModTimeNow > tearFragTime) {
            std::cout << "[HOT RELOAD] Reloading Tears shader from: " << tearFragPath << std::endl;
            logFile << "[HOT RELOAD] Reloading Tears shader from: " << tearFragPath << "\n";

            Shader newTearShader = LoadShader(0, tearFragPath.c_str());
            if (newTearShader.id != rlGetShaderIdDefault()) {
                UnloadShader(shTears);
                shTears = newTearShader;
                GetTearLocations();
                tearFragTime = tearModTimeNow;

                std::cout << "[HOT RELOAD] Tears shader reloaded successfully." << std::endl;
                logFile << "[HOT RELOAD] Tears shader reloaded successfully.\n";
                TraceLog(LOG_INFO, "HOT RELOAD: Tears shader updated.");
            } else {
                std::cout << "[HOT RELOAD] Tears shader reload FAILED (compile error)." << std::endl;
                logFile << "[HOT RELOAD] Tears shader reload FAILED (compile error).\n";
                TraceLog(LOG_WARNING, "HOT RELOAD: Tears shader compile failed.");
            }
        }

        // --- Brow Shader ---
        if (browModTimeNow > browFragTime) {
            std::cout << "[HOT RELOAD] Reloading Brow shader from: " << browFragPath << std::endl;
            logFile << "[HOT RELOAD] Reloading Brow shader from: " << browFragPath << "\n";

            Shader newBrowShader = LoadShader(0, browFragPath.c_str());
            if (newBrowShader.id != rlGetShaderIdDefault()) {
                UnloadShader(shBrow);
                shBrow = newBrowShader;
                GetBrowLocations();
                browFragTime = browModTimeNow;

                std::cout << "[HOT RELOAD] Brow shader reloaded successfully." << std::endl;
                logFile << "[HOT RELOAD] Brow shader reloaded successfully.\n";
                TraceLog(LOG_INFO, "HOT RELOAD: Brow shader updated.");
            } else {
                std::cout << "[HOT RELOAD] Brow shader reload FAILED (compile error)." << std::endl;
                logFile << "[HOT RELOAD] Brow shader reload FAILED (compile error).\n";
                TraceLog(LOG_WARNING, "HOT RELOAD: Brow shader compile failed.");
            }
        }

        logFile << "[HOT RELOAD] Reload pass complete.\n";
        std::cout << "[HOT RELOAD] Reload pass complete." << std::endl;
        logFile.close();
    }
};