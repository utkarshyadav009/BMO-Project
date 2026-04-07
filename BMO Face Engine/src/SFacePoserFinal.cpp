// FinalFacePoser.cpp
// UNIFIED DRIVER: Combined Eye and Mouth Editor
// USAGE: Compile with raylib, raygui, and json.hpp

#include "raylib.h"

// 1. SETUP RAYGUI
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "json.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <filesystem>

// Utility header (Assumed to exist per prompt, provides GlobalScaler)
#include "utility.h"

// ---------------------------------------------------------
// ENGINE INCLUDE
// ---------------------------------------------------------
// The authoritative unified shader backend
#include "ShaderParametricFace.cpp" 
#include "FaceData.h"
#include "AffectiveEngine.h"

using json = nlohmann::json;

// ---------------------------------------------------------
// UI CONSTANTS & HELPERS
// ---------------------------------------------------------
namespace UI {
    // Helper to shorten the syntax
    float S(float v) { return GlobalScaler.S(v); }

    float START_X() { return S(20.0f); }
    float START_Y() { return S(60.0f); }
    float PANEL_WIDTH() { return S(340.0f); }
    float LABEL_WIDTH() { return S(90.0f); }
    float VAL_WIDTH() { return S(40.0f); }
    float ROW_HEIGHT() { return S(25.0f); }
    
    float GetSliderWidth() {
        return PANEL_WIDTH() - LABEL_WIDTH() - VAL_WIDTH() - S(30.0f);
    }

    void Slider(const char* text, float* var, float min, float max, float& yPos) {
        GuiLabel({START_X() + S(10.0f), yPos, LABEL_WIDTH(), S(20.0f)}, text);
        GuiSliderBar({START_X() + S(10.0f) + LABEL_WIDTH(), yPos, GetSliderWidth(), S(20.0f)}, NULL, NULL, var, min, max);
        GuiLabel({START_X() + S(15.0f) + LABEL_WIDTH() + GetSliderWidth(), yPos, VAL_WIDTH(), S(20.0f)}, TextFormat("%.2f", *var));
        yPos += ROW_HEIGHT();
    }

    bool Checkbox(const char* text, bool* var, float xOffset, float yPos) {
        return GuiCheckBox({START_X() + S(xOffset), yPos, S(20.0f), S(20.0f)}, text, var);
    }
}

// ---------------------------------------------------------
// 1. UNIFIED ATLAS LOADER
// ---------------------------------------------------------
struct ReferenceAtlas {
    Texture2D texture;
    std::unordered_map<std::string, Rectangle> frames;
    std::vector<std::string> faceNames;


    void Load(const char* img, const char* data) {
        texture = LoadTexture(img);
        if (texture.id == 0) {
            std::cerr << "[Atlas] Error loading texture: " << img << std::endl;
            return;
        }

        std::ifstream f(data);
        if (!f.good()) {
            std::cerr << "[Atlas] Error loading JSON: " << data << std::endl;
            return;
        }

        try {
            json j = json::parse(f);
            if (j.contains("textures")) {
                for (auto& t : j["textures"]) {
                    for (auto& fr : t["frames"]) {
                        std::string name = fr["filename"];
                        std::string lowerName = name;
                        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                        
                        frames[name] = {
                            (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                            (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                        };

               
                        faceNames.push_back(name);
                    }
                }
            }
            std::sort(faceNames.begin(), faceNames.end());
            
            std::cout << "[Atlas] Loaded " << faceNames.size() << " Faces." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Atlas] JSON Parse Error: " << e.what() << std::endl;
        }
    }

    void Draw(const std::string& name, Vector2 pos, float scale, float alpha) {
        if (name.empty() || frames.find(name) == frames.end()) return;
        Rectangle src = frames[name];
        Rectangle dest = { pos.x, pos.y, src.width * scale, src.height * scale };
        Vector2 origin = { dest.width/2, dest.height/2 };
        DrawTexturePro(texture, src, dest, origin, 0.0f, Fade(WHITE, alpha));
    }
};

// ---------------------------------------------------------
// EDITOR STATE
// ---------------------------------------------------------
struct EditorState {
    FaceState current;
    
    // Atlas Selection Indices
    int faceRefIdx = 0;
    
    // Editor Settings
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false;
    bool enableGUI = true;
    bool showFace = true;
    bool debugBoxes = false;
    int tabIndex = 0; 
    
    // Dropdown UI
    int dropdownActive = 0;
    bool dropdownEditMode = false;
    
    void CycleFace(ReferenceAtlas& atlas, int dir ){
		if(atlas.faceNames.empty()) return;
		faceRefIdx += dir;
		if(faceRefIdx < 0) faceRefIdx = (int)atlas.faceNames.size() - 1;
		if(faceRefIdx >= (int)atlas.faceNames.size()) faceRefIdx = 0;
	}

    AffectiveState moodPhysics;
    bool useAI = false; //Toggle between manual sliders and AI brain 

    EditorState() {
        // Set default "Happy/Neutral" state
        AppraisalVector start = { 0.8f, 0.4f, 0.8f, 0.0f, 0.0f };
        moodPhysics.Reset(start);
    }
};


// ---------------------------------------------------------
// UI SUB-PANELS
// ---------------------------------------------------------
void DrawEyeControls(float& y, EyeParams& p) {
    using namespace UI;
    GuiGroupBox({START_X(), y, PANEL_WIDTH(), S(350.0f)}, "EYE SHAPE"); y += S(20.0f);

    int shapeInt = (int)p.eyeShapeID;
    Slider("Shape ID", &p.eyeShapeID, 0.0f, 12.0f, y);
    const char* shapeNames[] = { "Dot", "Line", "Arc", "Cross", "Star", "Heart", "Spiral", "Chevron", "Shuriken", "Kawaii", "Shocked", "Teary", "Colon Eyes" };
    if(shapeInt >= 0 && shapeInt <= 12) GuiLabel({START_X() + S(10.0f) + LABEL_WIDTH(), y - S(25.0f), GetSliderWidth(), S(20.0f)}, shapeNames[shapeInt]);

    if(p.eyeShapeID > 5.5f && p.eyeShapeID < 6.5f) Slider("Spiral Spd", &p.spiralSpeed, -10.0f, 10.0f, y);

    Slider("Bend", &p.bend, -2.0f, 2.0f, y);
    Slider("Thickness", &p.eyeThickness, 1.0f, 30.0f, y); y += S(5.0f);
    Slider("Scale X", &p.scaleX, 0.1f, 30.0f, y);
    Slider("Scale Y", &p.scaleY, 0.1f, 30.0f, y);
    Slider("Spacing", &p.spacing, 0.0f, 1000.0f, y);
    Slider("Look X", &p.lookX, -500.0f, 500.0f, y);
    Slider("Look Y", &p.lookY, -500.0f, 500.0f, y);
    Slider("Angle", &p.angle, -180.0f, 180.0f, y);
    Slider("Squareness", &p.squareness, 0.0f, 1.0f, y);
    Slider("Pixelation", &p.pixelation,1.0f, 15.0f, y); y += S(20.0f);

    GuiGroupBox({START_X(), y, PANEL_WIDTH(), S(480.0f)}, "EYE FX"); y += S(5.0f);
    Checkbox("Brows", &p.showBrow, 10.0f, y);
    Checkbox("Tears", &p.showTears, 90.0f, y);
    Checkbox("Blush", &p.showBlush, 170.0f, y); y += S(30.0f);

    if (p.showBrow) {
        GuiLabel({START_X() + S(10.0f), y, S(200.0f), S(20.0f)}, "- BROW SETTINGS -"); y+= S(20.0f);
        Slider("Thick", &p.eyebrowThickness, 1, 20, y); Slider("Len", &p.eyebrowLength, 0.5f, 2.0f, y);
        Slider("Spacing", &p.eyebrowSpacing, -100.0f, 100.0f, y); Slider("Pos X", &p.eyebrowX, -10, 10, y);
        Slider("Pos Y", &p.eyebrowY, -10, 10, y); Slider("Scale", &p.browScale, 0.5f, 2.0f, y);
        Slider("Angle", &p.browAngle, -45.0f, 45.0f, y); Slider("Bend", &p.browBend, -2.0f, 2.0f, y);
        Slider("Bend Off", &p.browBendOffset, 0.0f, 0.99f, y); Checkbox("Use Lower Brow", &p.useLowerBrow, 10.0f, y); y += S(35.0f);
    }
    if (p.showTears) { GuiLabel({START_X() + S(10.0f), y, S(200.0f), S(20.0f)}, "- TEAR SETTINGS -"); y+= S(20.0f); Slider("Level", &p.tearsLevel, 0, 1, y); }
    if(p.showBlush) {
        GuiLabel({START_X() + S(10.0f), y, S(200.0f), S(20.0f)}, "- BLUSH SETTINGS -"); y+= S(20.0f);
        Slider("Scale", &p.blushScale, 0.1f, 3.0f, y); Slider("Pos X", &p.blushX, -10.0f, 10.0f, y);
        Slider("Pos Y", &p.blushY, -10.0f, 10.0f, y); Slider("Space", &p.blushSpacing, -100.0f, 100.0f, y);
        GuiLabel({START_X() + S(10.0f), y, LABEL_WIDTH(), S(20.0f)}, "Blush Mode");
        if(GuiButton({START_X() + S(10.0f) + LABEL_WIDTH(), y, S(80.0f), S(20.0f)}, p.blushMode == 0 ? "Pink" : (p.blushMode == 1 ? "Green" : "Yellow"))) p.blushMode = (p.blushMode + 1) % 3;
        y += S(35.0f);
    }
    GuiLabel({START_X() + S(10.0f), y, PANEL_WIDTH(), S(20.0f)}, "--- SURFACE FX ---"); y += S(20.0f);
    Slider("Stress", &p.stressLevel, 0.0f, 1.0f, y); Slider("Gloom", &p.gloomLevel, 0.0f, 1.0f, y);
}

void DrawMouthControls(float& y, MouthParams& p) {
    using namespace UI;
    GuiGroupBox({START_X(), y, PANEL_WIDTH(), S(480.0f)}, "MOUTH SETTINGS"); y += S(20.0f);
    Slider("Scale", &p.scale, 0.5f, 10.0f, y);
    Slider("Look X", &p.lookX, -250.0f, 250.0f, y);
    Slider("Look Y", &p.lookY, -250.0f, 250.0f, y);
    Slider("Mouth Angle", &p.mouthAngle, -180.0f, 180.0f, y);
    Slider("Outline", &p.outlineThickness, 1.f, 30.0f, y); y += S(10.0f);
    Slider("Open", &p.open, 0.0f, 1.2f, y); 
    Slider("Width", &p.width, 0.1f, 1.5f, y);
    Slider("Curve", &p.curve, -5.0f, 5.0f, y); y += S(10.0f);
    Slider("Sqze Top", &p.squeezeTop, -1.0f, 1.0f, y); Slider("Sqze Bot", &p.squeezeBottom, -1.0, 1.0f, y); y += S(10.0f);
    Slider("Sqze Sigma", &p.sigma, 0.0f, 1.0f, y); Slider("Sqze Pow", &p.power, 0.0f, 10.0f, y);
    Slider("Sqze Lift", &p.maxLiftValue, 0.0f, 1.0f, y); y += S(10.0f);
    Slider("Teeth Y", &p.teethY, -1.0f, 1.0f, y); Slider("Teeth W", &p.teethWidth, 0.1f, 1.0f, y);
    Slider("Teeth Gap",&p.teethGap, 0.0f, 100.0f, y); y += S(10.0f);
    Slider("Tongue Up", &p.tongueUp, 0.0f, 1.0f, y); Slider("Tongue W", &p.tongueWidth, 0.3f, 1.0f, y);
    Slider("Tongue X", &p.tongueX, -1.0f, 1.0f, y); y += S(10.0f);
    Slider("Asymmetry", &p.asymmetry, -1.0f, 1.0f, y); Slider("Squareness", &p.squareness, 0.0f, 1.0f, y); y += S(10.0f);
    Slider("Stress Lns", &p.stressLines, 0.0f, 1.0f, y); y += S(10.0f);
    Checkbox("Show Inner Mouth", &p.showInnerMouth, 10.0f, y); y += S(20.0f);
    Checkbox("3 Shape", &p.isThreeShape, 10.0f, y); y += S(20.0f);
    Checkbox("D Shape", &p.isDShape, 10.0f, y); y += S(20.0f);
    Checkbox("- Shape", &p.isSlashShape, 10.0f, y);
}
// ---------------------------------------------------------
// COGNITIVE LAYER: KEYWORD EXTRACTION
// ---------------------------------------------------------
void AnalyzeText(char* text, EditorState& state) {
    std::string s = text;
    // Convert to lowercase for easier matching
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    // 1. SURPRISE (Reflex)
    if (s.find("!") != std::string::npos || s.find("wow") != std::string::npos || s.find("what") != std::string::npos) {
        state.moodPhysics.target.novelty = 1.0f; // Spike!
    }

    // 2. HAPPINESS (High Valence)
    if (s.find("happy") != std::string::npos || s.find("love") != std::string::npos || s.find("good") != std::string::npos || s.find("hi") != std::string::npos) {
        state.moodPhysics.target.valence = 0.8f;
        state.moodPhysics.target.arousal = 0.5f;
        state.moodPhysics.target.control = 0.5f;
        state.moodPhysics.target.obstruct = 0.0f;
    }

    // 3. ANGER (High Arousal, Neg Valence, HIGH CONTROL)
    else if (s.find("hate") != std::string::npos || s.find("bad") != std::string::npos || s.find("stop") != std::string::npos || s.find("angry") != std::string::npos) {
        state.moodPhysics.target.valence = -0.9f;
        state.moodPhysics.target.arousal = 0.9f;
        state.moodPhysics.target.control = 1.0f; 
        state.moodPhysics.target.obstruct = 1.0f;
    }

    // 4. SADNESS (Low Arousal, Neg Valence, LOW CONTROL)
    else if (s.find("sad") != std::string::npos || s.find("sorry") != std::string::npos || s.find("miss") != std::string::npos) {
        state.moodPhysics.target.valence = -0.8f;
        state.moodPhysics.target.arousal = 0.2f;
        state.moodPhysics.target.control = 0.1f; // <--- Helplessness
        state.moodPhysics.target.obstruct = 0.5f;
    }
    
    // 5. FEAR (High Arousal, Neg Valence, LOW CONTROL)
    else if (s.find("scared") != std::string::npos || s.find("help") != std::string::npos || s.find("no") != std::string::npos) {
        state.moodPhysics.target.valence = -0.9f;
        state.moodPhysics.target.arousal = 0.9f;
        state.moodPhysics.target.control = 0.0f; // <--- The difference between Rage and Fear
    }
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 950, "BMO Face Poser: Final");
    SetTargetFPS(60);

    // Style Setup
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt(BLACK));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(BLACK));
    GuiSetStyle(SLIDER, BASE_COLOR_NORMAL, ColorToInt({ 60, 60, 60, 255 }));
    GuiSetStyle(SLIDER, BASE_COLOR_FOCUSED, ColorToInt({ 120, 180, 255, 255 }));
    GuiSetStyle(DROPDOWNBOX, BASE_COLOR_NORMAL, ColorToInt({ 40, 40, 40, 255 }));
    GuiSetStyle(DROPDOWNBOX, TEXT_COLOR_NORMAL, ColorToInt(WHITE));

    // Initialize Engine
    FaceSystem engine;
    engine.Init();
    
    // Load Assets
    ReferenceAtlas atlas;
    atlas.Load("assets/BMO_SpriteSheet_Texture.png", "assets/BMO_SpriteSheet_Data.json");

    FaceDatabase db;
    // Attempt to load unified database, fallbacks handled by struct defaults
    db.Load("face_database.txt");

    EditorState state;

    AffectiveEngine brain;
    brain.LoadFromDB(db);
    brain.InitLogger(); 
    state.showReference = false;
    state.useAI = true;
    state.enableGUI = false;
    state.usePhysics = true;
    char textInput[256] = "Type here..."; 
    bool textEditMode = false;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        // Assume GlobalScaler is available from utility.h as per environment
        GlobalScaler.Update();

        // Calculate positions
        Vector2 center = { (float)GetScreenWidth() * 0.5f, (float)GetScreenHeight() * 0.5f };
        float mouthOffset = 100.0f * GlobalScaler.scale;
        // Mouth is offset slightly lower than eyes
        Vector2 mouthPos = { center.x, center.y + mouthOffset };

        // -------------------------------------------------
        // LOGIC
        // -------------------------------------------------
        engine.usePhysics = state.usePhysics;
        engine.debugBoxes = state.debugBoxes;
        
        // Push state to engine
        engine.Update(dt, state.current.eyes, state.current.mouth);
        state.moodPhysics.Update(dt);
        // -------------------------------------------------
        // DRAWING
        // -------------------------------------------------
        BeginDrawing();
        ClearBackground({201, 228, 195, 255}); // BMO Green

        // 1. Reference Layer (Background)
        if (state.showReference) {
            if (!atlas.faceNames.empty()) 
                atlas.Draw(atlas.faceNames[state.faceRefIdx], center, 1.0f, state.refOpacity);
        }

        // 2. Procedural Face Layer (Midground)
        // Note: Engine draws internally to a texture then presents it
        if(state.showFace) engine.Draw(center, mouthPos, state.current.eyes, state.current.mouth, BLACK);

        // 3. UI Layer (Foreground)
        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();
        
        // Sprite Navigation Shortcuts
        if (IsKeyPressed(KEY_RIGHT)) {
            if (state.tabIndex == 0) state.CycleFace(atlas, 1);
        }
        if (IsKeyPressed(KEY_LEFT)) {
            if (state.tabIndex == 0) state.CycleFace(atlas, -1);
        }
        float screenW = (float)GetScreenWidth();
        // auto DrawGhostSlider = [&](const char* label, float* target, float current, float y) {
        //     GuiLabel({screenW - 260, y, 80, 20}, label);

        //     // 1. Draw the "Ghost" (Current Physical State) as a thin colored line BEHIND the slider
        //     // Map -1..1 to 0..140 (pixel width)
        //     float normalized = (current + 1.0f) * 0.5f; 
        //     if (std::string(label) == "Arousal" || std::string(label) == "Control" || std::string(label) == "Novelty" || std::string(label) == "Obstruct") {
        //         normalized = current; // 0..1 range
        //     }

        //     DrawRectangle(screenW - 180, y + 5, (int)(140 * normalized), 10, Fade(BLUE, 0.5f));
        
        //     // 2. Draw the Actual Slider (The User Target) on top
        //     GuiSliderBar({screenW - 180, y, 140, 20}, NULL, NULL, target, 
        //                  (std::string(label) == "Valence") ? -1.0f : 0.0f, 1.0f);
        // };

        // 2. LOGIC: AI vs Manual
        if (state.useAI) {
            // Decay Novelty by 50% every second
           state.moodPhysics.current.novelty = Lerp(state.moodPhysics.current.novelty, 0.0f, dt * 0.5f);
            // A. Solve the Manifold
            //FaceState targetState = brain.SolveDual(state.currentMood);
            FaceState targetState = brain.Solve(state.moodPhysics.current);

            //FaceState targetState = brain.SolveDual(state.moodPhysics.current);
            // B. Apply Physics (Smooths the transition)
            // We assign targetState to state.current, but physics engine will interpolate it
            // If you want instant snap, just copy. If you want physics, set as target.
            // For now, let's just copy to visualize the raw manifold output:
            state.current = targetState;
        }

        GuiGroupBox({screenW - 270, 300, 250, 220}, "COGNITIVE CORE");
        
        GuiCheckBox({screenW - 260, 320, 20, 20}, "ACTIVATE AI BRAIN", &state.useAI);
        if (state.useAI) {
            float y = 350;
            
            
            // --- GHOST SLIDERS (Visualizes the Physics Lag) ---
            auto DrawGhostSlider = [&](const char* label, float* target, float current, float yPos) {
                GuiLabel({screenW - 260, yPos, 80, 20}, label);
                
                // Draw BLUE BAR (Where the face IS) behind the slider
                float normalized = (current + 1.0f) * 0.5f; 
                if (std::string(label) != "Valence") normalized = current;
                DrawRectangle(screenW - 180, yPos + 5, (int)(140 * normalized), 10, Fade(BLUE, 0.4f));

                // Draw SLIDER (Where you WANT it to be)
                GuiSliderBar({screenW - 180, yPos, 140, 20}, NULL, NULL, target, 
                             (std::string(label) == "Valence") ? -1.0f : 0.0f, 1.0f);
            };

            DrawGhostSlider("Valence", &state.moodPhysics.target.valence, state.moodPhysics.current.valence, y); y+=25;
            DrawGhostSlider("Arousal", &state.moodPhysics.target.arousal, state.moodPhysics.current.arousal, y); y+=25;
            DrawGhostSlider("Control", &state.moodPhysics.target.control, state.moodPhysics.current.control, y); y+=25;
            
            // Surprise Decay Visualizer
            GuiLabel({screenW - 260, y, 80, 20}, "Novelty");
            float novPhys = state.moodPhysics.current.novelty;
            DrawRectangle((int)screenW - 180.0f, y + 5.0f, (int)(140 * novPhys), 10, Fade(RED, 0.7f)); 
            // Also allow manual trigger
            if (GuiButton({screenW - 60, y, 20, 20}, "!")) state.moodPhysics.target.novelty = 1.0f;
            y += 25;

            DrawGhostSlider("Obstruct", &state.moodPhysics.target.obstruct, state.moodPhysics.current.obstruct, y);
            y+=25;
            // --- TEXT INPUT BOX ---
            if (GuiTextBox({screenW - 260, y, 240, 30}, textInput, 256, textEditMode)) {
                textEditMode = !textEditMode;
            }
            // If user presses ENTER, analyze the text
            if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER)) {
                AnalyzeText(textInput, state);
                textEditMode = false; // Unfocus
            }
        }

        if (state.enableGUI) {
            // FIX: Added parentheses to START_Y()
            float y = UI::START_Y();

            // Tab Bar
            // FIX: Added parentheses to START_X() and scaled the box sizes
            GuiToggleGroup({UI::START_X(), GlobalScaler.S(20.0f), GlobalScaler.S(120.0f), GlobalScaler.S(30.0f)}, "EYES;MOUTH", &state.tabIndex);

            // -- LEFT COLUMN: REFERENCE --
            // FIX: Added parentheses to START_X() and PANEL_WIDTH()
            GuiGroupBox({UI::START_X(), y, UI::PANEL_WIDTH(), GlobalScaler.S(110.0f)}, "SPRITE REFERENCE");
            if (GuiButton({UI::START_X() + GlobalScaler.S(10.0f), y + GlobalScaler.S(30.0f), GlobalScaler.S(40.0f), GlobalScaler.S(30.0f)}, "<")) {
                if (state.tabIndex == 0) state.CycleFace(atlas, -1);
            }
            if (GuiButton({UI::START_X() + GlobalScaler.S(270.0f), y + GlobalScaler.S(30.0f), GlobalScaler.S(40.0f), GlobalScaler.S(30.0f)}, ">")) {
                if (state.tabIndex == 0) state.CycleFace(atlas, 1);
            }

            std::string refLabel = atlas.faceNames.empty() ? "NONE" : atlas.faceNames[state.faceRefIdx];
            GuiLabel({UI::START_X() + GlobalScaler.S(60.0f), y + GlobalScaler.S(30.0f), GlobalScaler.S(200.0f), GlobalScaler.S(30.0f)}, refLabel.c_str());
            
            if (GuiButton({UI::START_X() + GlobalScaler.S(10.0f), y + GlobalScaler.S(70.0f), UI::PANEL_WIDTH() - GlobalScaler.S(20.0f), GlobalScaler.S(30.0f)}, "SAVE PRESET (Enter)")) {
                db.Save("face_database.txt", refLabel, state.current);
            }
            y += GlobalScaler.S(120.0f);

            // -- LEFT COLUMN: PARAMETER CONTROLS --
            if (state.tabIndex == 0) {
                DrawEyeControls(y, state.current.eyes);
            } else {
                DrawMouthControls(y, state.current.mouth);
            }

            // -- RIGHT COLUMN: DATABASE & GLOBAL --
            float screenW = (float)GetScreenWidth();
            float screenH = (float)GetScreenHeight();
            float panelW = 250.0f;
            float vx = screenW - panelW - 16;
            float vy = 16;
            
            if (GuiButton({ vx + 10, vy + 65, 160, 30 }, "LOAD SELECTED")) {
                if (!db.entries.empty() && state.dropdownActive < (int)db.entries.size()) {
                    state.current = db.entries[state.dropdownActive].state;
                }
            }

            if (GuiButton({ vx + panelW - 60, vy + 65, 50, 30 }, "RLD")) db.Load("face_database.txt");

            // Viewport Settings
            float vyView = vy + 120 + 12.0f;
            GuiGroupBox({ vx, vyView, panelW, 130 }, "VIEWPORT");
            GuiCheckBox({ vx + 10, vyView + 25, 20, 20 }, "Show Ref", &state.showReference);
            GuiSliderBar({ vx + 80, vyView + 55, 100, 20 }, "Opac", nullptr, &state.refOpacity, 0.0f, 1.0f);
            GuiCheckBox({ vx + 10, vyView + 80, 20, 20 }, "Test Physics", &state.usePhysics);
            GuiCheckBox({ vx + 10, vyView + 105, 20, 20 }, "Show Face", &state.showFace);

            // Bottom Right Toggles
            GuiCheckBox({screenW - 180, screenH - 40, 20, 20}, "Debug Boxes", &state.debugBoxes);
            GuiCheckBox({screenW - 180, screenH - 70, 20, 20}, "Enable GUI", &state.enableGUI);
            if (GuiButton({screenW - 180, screenH - 120, 50, 20}, "Reset")) state.current.reset();


            GuiGroupBox({ vx, vy, panelW, 120 }, "DATABASE LOAD");
            if (GuiDropdownBox({ vx + 10, vy + 30, panelW - 20, 25 }, db.dropdownStr.c_str(), &state.dropdownActive, state.dropdownEditMode)) {
                state.dropdownEditMode = !state.dropdownEditMode;
            }

            
        }
        else {
             // Minimal UI to re-enable
             GuiCheckBox({(float)GetScreenWidth() - 180, (float)GetScreenHeight() - 70, 20, 20}, "Enable GUI", &state.enableGUI);
        }

        // Global Save Shortcut
        // if (IsKeyPressed(KEY_ENTER)) {
        //      std::string name = "custom";
        //      if (state.tabIndex == 0 && !atlas.faceNames.empty()) name = atlas.faceNames[state.faceRefIdx];
        //      db.Save("face_database.txt", name, state.current);
        // }

        EndDrawing();
    }

    engine.Unload();
    CloseWindow();
    return 0;
}