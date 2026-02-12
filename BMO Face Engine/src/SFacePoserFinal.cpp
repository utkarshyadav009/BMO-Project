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
    const float START_X = 20.0f;
    const float START_Y = 60.0f; // Shifted down for Tab Bar
    const float PANEL_WIDTH = 340.0f;
    const float LABEL_WIDTH = 90.0f;
    const float VAL_WIDTH = 40.0f;
    const float ROW_HEIGHT = 25.0f;
    
    float GetSliderWidth() {
        return PANEL_WIDTH - LABEL_WIDTH - VAL_WIDTH - 30.0f;
    }

    void Slider(const char* text, float* var, float min, float max, float& yPos) {
        GuiLabel({START_X + 10, yPos, LABEL_WIDTH, 20}, text);
        GuiSliderBar({START_X + 10 + LABEL_WIDTH, yPos, GetSliderWidth(), 20}, NULL, NULL, var, min, max);
        GuiLabel({START_X + 10 + LABEL_WIDTH + GetSliderWidth() + 5, yPos, VAL_WIDTH, 20}, TextFormat("%.2f", *var));
        yPos += ROW_HEIGHT;
    }

    bool Checkbox(const char* text, bool* var, float xOffset, float yPos) {
        return GuiCheckBox({START_X + xOffset, yPos, 20, 20}, text, var);
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
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 350}, "EYE SHAPE"); y += 20.0f;

    int shapeInt = (int)p.eyeShapeID;
    UI::Slider("Shape ID", &p.eyeShapeID, 0.0f, 12.0f, y);
    const char* shapeNames[] = { "Dot", "Line", "Arc", "Cross", "Star", "Heart", "Spiral", "Chevron", "Shuriken", "Kawaii", "Shocked", "Teary", "Colon Eyes" };
    if(shapeInt >= 0 && shapeInt <= 12) GuiLabel({UI::START_X + 10 + UI::LABEL_WIDTH, y - 25, UI::GetSliderWidth(), 20}, shapeNames[shapeInt]);

    if(p.eyeShapeID > 5.5f && p.eyeShapeID < 6.5f) UI::Slider("Spiral Spd", &p.spiralSpeed, -10.0f, 10.0f, y);

    UI::Slider("Bend", &p.bend, -2.0f, 2.0f, y);
    UI::Slider("Thickness", &p.eyeThickness, 1.0f, 30.0f, y);
    y += 5;
    UI::Slider("Scale X", &p.scaleX, 0.1f, 30.0f, y);
    UI::Slider("Scale Y", &p.scaleY, 0.1f, 30.0f, y);
    UI::Slider("Spacing", &p.spacing, 0.0f, 1000.0f, y);
    UI::Slider("Look X", &p.lookX, -500.0f, 500.0f, y);
    UI::Slider("Look Y", &p.lookY, -500.0f, 500.0f, y);
    UI::Slider("Angle", &p.angle, -180.0f, 180.0f, y);
    UI::Slider("Squareness", &p.squareness, 0.0f, 1.0f, y);
    UI::Slider("Pixelation", &p.pixelation,1.0f, 15.0f, y);
    y += 20.0f;

    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 480}, "EYE FX"); y += 20.0f;
    UI::Checkbox("Brows", &p.showBrow, 10, y);
    UI::Checkbox("Tears", &p.showTears, 90, y);
    UI::Checkbox("Blush", &p.showBlush, 170, y); y += 30.0f;

    if (p.showBrow) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- BROW SETTINGS -"); y+= 20;
        UI::Slider("Thick", &p.eyebrowThickness, 1, 20, y); UI::Slider("Len", &p.eyebrowLength, 0.5f, 2.0f, y);
        UI::Slider("Spacing", &p.eyebrowSpacing, -100.0f, 100.0f, y); UI::Slider("Pos X", &p.eyebrowX, -10, 10, y);
        UI::Slider("Pos Y", &p.eyebrowY, -10, 10, y); UI::Slider("Scale", &p.browScale, 0.5f, 2.0f, y);
        UI::Slider("Angle", &p.browAngle, -45.0f, 45.0f, y); UI::Slider("Bend", &p.browBend, -2.0f, 2.0f, y);
        UI::Slider("Bend Off", &p.browBendOffset, 0.0f, 0.99f, y); UI::Checkbox("Use Lower Brow", &p.useLowerBrow, 10, y); y += 35;
    }
    if (p.showTears) { GuiLabel({UI::START_X+10, y, 200, 20}, "- TEAR SETTINGS -"); y+= 20; UI::Slider("Level", &p.tearsLevel, 0, 1, y); }
    if(p.showBlush) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- BLUSH SETTINGS -"); y+= 20;
        UI::Slider("Scale", &p.blushScale, 0.1f, 3.0f, y); UI::Slider("Pos X", &p.blushX, -10.0f, 10.0f, y);
        UI::Slider("Pos Y", &p.blushY, -10.0f, 10.0f, y); UI::Slider("Space", &p.blushSpacing, -100.0f, 100.0f, y);
        GuiLabel({UI::START_X+10, y, UI::LABEL_WIDTH, 20}, "Blush Mode");
        if(GuiButton({UI::START_X+10+UI::LABEL_WIDTH,  y, 80, 20}, p.blushMode == 0 ? "Pink" : (p.blushMode == 1 ? "Green" : "Yellow"))) p.blushMode = (p.blushMode + 1) % 3;
        y += 35;
    }
    GuiLabel({UI::START_X+10, y, UI::PANEL_WIDTH, 20}, "--- SURFACE FX ---"); y += 20;
    UI::Slider("Stress", &p.stressLevel, 0.0f, 1.0f, y); UI::Slider("Gloom", &p.gloomLevel, 0.0f, 1.0f, y);
}

void DrawMouthControls(float& y, MouthParams& p) {
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 480}, "MOUTH SETTINGS"); y += 20.0f;
    UI::Slider("Scale", &p.scale, 0.5f, 10.0f, y);
    UI::Slider("Look X", &p.lookX, -250.0f, 250.0f, y);
    UI::Slider("Look Y", &p.lookY, -250.0f, 250.0f, y);
    UI::Slider("Mouth Angle", &p.mouthAngle, -180.0f, 180.0f, y);
    UI::Slider("Outline", &p.outlineThickness, 1.f, 30.0f, y); y += 10;
    UI::Slider("Open", &p.open, 0.0f, 1.2f, y); 
    UI::Slider("Width", &p.width, 0.1f, 1.5f, y);
    UI::Slider("Curve", &p.curve, -5.0f, 5.0f, y); y += 10;
    UI::Slider("Sqze Top", &p.squeezeTop, -1.0f, 1.0f, y); UI::Slider("Sqze Bot", &p.squeezeBottom, -1.0, 1.0f, y); y += 10;
    UI::Slider("Sqze Sigma", &p.sigma, 0.0f, 1.0f, y); UI::Slider("Sqze Pow", &p.power, 0.0f, 10.0f, y);
    UI::Slider("Sqze Lift", &p.maxLiftValue, 0.0f, 1.0f, y); y += 10;
    UI::Slider("Teeth Y", &p.teethY, -1.0f, 1.0f, y); UI::Slider("Teeth W", &p.teethWidth, 0.1f, 1.0f, y);
    UI::Slider("Teeth Gap",&p.teethGap, 0.0f, 100.0f, y); y += 10;
    UI::Slider("Tongue Up", &p.tongueUp, 0.0f, 1.0f, y); UI::Slider("Tongue W", &p.tongueWidth, 0.3f, 1.0f, y);
    UI::Slider("Tongue X", &p.tongueX, -1.0f, 1.0f, y); y += 10;
    UI::Slider("Asymmetry", &p.asymmetry, -1.0f, 1.0f, y); UI::Slider("Squareness", &p.squareness, 0.0f, 1.0f, y);y+=10;
    UI::Slider("Stress Lns", &p.stressLines, 0.0f, 1.0f, y);y+=10;
    UI::Checkbox("Show Inner Mouth", &p.showInnerMouth, 10, y); y+=20;
    UI::Checkbox("3 Shape", &p.isThreeShape, 10, y);y+=20;
    UI::Checkbox("D Shape", &p.isDShape, 10, y);y+=20;
    UI::Checkbox("- Shape", &p.isSlashShape, 10, y);

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

        float screenW = (float)GetScreenWidth();
        GuiGroupBox({screenW - 270, 300, 250, 200}, "COGNITIVE CORE");
        
        GuiCheckBox({screenW - 260, 320, 20, 20}, "ACTIVATE AI BRAIN", &state.useAI);
        if (state.useAI) {
            float y = 350;
            // The 5 Knobs of the Soul
            GuiLabel({screenW - 260, y, 80, 20}, "Valence");
            GuiSliderBar({screenW - 180, y, 140, 20}, NULL, NULL, &state.moodPhysics.target.valence, -1.0f, 1.0f); y+=25;
            
            GuiLabel({screenW - 260, y, 80, 20}, "Arousal");
            GuiSliderBar({screenW - 180, y, 140, 20}, NULL, NULL, &state.moodPhysics.target.arousal, 0.0f, 1.0f); y+=25;
            
            GuiLabel({screenW - 260, y, 80, 20}, "Control");
            GuiSliderBar({screenW - 180, y, 140, 20}, NULL, NULL, &state.moodPhysics.target.control, 0.0f, 1.0f); y+=25;
            
            if (GuiButton({screenW - 180, y, 140, 20}, "TRIGGER SURPRISE")) {
                state.moodPhysics.target.novelty = 1.0f; // Spike to max
            }
            y += 25;           
            GuiLabel({screenW - 260, y, 80, 20}, "Obstruct");
            GuiSliderBar({screenW - 180, y, 140, 20}, NULL, NULL, &state.moodPhysics.target.obstruct, 0.0f, 1.0f);
        }

        if (state.enableGUI) {
            float y = UI::START_Y;

            // Tab Bar
            GuiToggleGroup({UI::START_X, 20, 120, 30}, "EYES;MOUTH", &state.tabIndex);

            // -- LEFT COLUMN: REFERENCE --
            GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 110}, "SPRITE REFERENCE");
            if (GuiButton({UI::START_X + 10, y + 30, 40, 30}, "<")) {
                if (state.tabIndex == 0) state.CycleFace(atlas, -1);
            }
            if (GuiButton({UI::START_X + 270, y + 30, 40, 30}, ">")) {
                if (state.tabIndex == 0) state.CycleFace(atlas, 1);
            }

            std::string refLabel = atlas.faceNames.empty() ? "NONE" : atlas.faceNames[state.faceRefIdx];
			GuiLabel({UI::START_X + 60, y + 30, 200, 30}, refLabel.c_str());
            
            GuiLabel({UI::START_X + 60, y + 30, 200, 30}, refLabel.c_str());

            if (GuiButton({UI::START_X + 10, y + 70, UI::PANEL_WIDTH - 20, 30}, "SAVE PRESET (Enter)")) {
                db.Save("face_database.txt", refLabel, state.current);
            }
            y += 120.0f;

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
            GuiGroupBox({ vx, vyView, panelW, 110 }, "VIEWPORT");
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
        if (IsKeyPressed(KEY_ENTER)) {
             std::string name = "custom";
             if (state.tabIndex == 0 && !atlas.faceNames.empty()) name = atlas.faceNames[state.faceRefIdx];
             db.Save("face_database.txt", name, state.current);
        }

        EndDrawing();
    }

    engine.Unload();
    CloseWindow();
    return 0;
}