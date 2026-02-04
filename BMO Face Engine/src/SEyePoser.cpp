// SEyePoser.cpp
// DRIVER FILE: Links the GUI to the Shader-Based Eye Engine
// USAGE: Compile this to create the Eye Editor tool.

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
#include "utility.h"

// Include the Refactored Shader Engine
#include "ShaderParametricEyes.cpp" 

using json = nlohmann::json;

// ---------------------------------------------------------
// UI CONSTANTS
// ---------------------------------------------------------
namespace UI {
    const float START_X = 20.0f;
    const float START_Y = 20.0f;
    const float PANEL_WIDTH = 340.0f;
    const float LABEL_WIDTH = 90.0f;
    const float VAL_WIDTH = 40.0f;
    const float ROW_HEIGHT = 25.0f;
    
    // Helper to calculate slider width dynamically
    float GetSliderWidth() {
        return PANEL_WIDTH - LABEL_WIDTH - VAL_WIDTH - 30.0f;
    }
}

// ---------------------------------------------------------
// GUI HELPERS
// ---------------------------------------------------------
namespace GuiHelper {
    // Standard Slider with Label and Value Display
    void Slider(const char* text, float* var, float min, float max, float& yPos) {
        GuiLabel({UI::START_X + 10, yPos, UI::LABEL_WIDTH, 20}, text);
        GuiSliderBar({UI::START_X + 10 + UI::LABEL_WIDTH, yPos, UI::GetSliderWidth(), 20}, NULL, NULL, var, min, max);
        GuiLabel({UI::START_X + 10 + UI::LABEL_WIDTH + UI::GetSliderWidth() + 5, yPos, UI::VAL_WIDTH, 20}, TextFormat("%.2f", *var));
        yPos += UI::ROW_HEIGHT;
    }

    // Checkbox wrapper
    bool Checkbox(const char* text, bool* var, float xOffset, float yPos) {
        return GuiCheckBox({UI::START_X + xOffset, yPos, 20, 20}, text, var);
    }
}

// ---------------------------------------------------------
// 1. ATLAS LOADER
// ---------------------------------------------------------
struct ReferenceAtlas {
    Texture2D texture;
    std::unordered_map<std::string, Rectangle> frames;
    std::vector<std::string> eyeNames; 

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
                        
                        if (lowerName.find("eye") != std::string::npos) {
                            frames[name] = {
                                (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                                (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                            };
                            eyeNames.push_back(name);
                        }
                    }
                }
            }
            std::sort(eyeNames.begin(), eyeNames.end());
            std::cout << "[Atlas] Loaded " << eyeNames.size() << " eye sprites." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Atlas] JSON Parse Error: " << e.what() << std::endl;
        }
    }

    void Draw(std::string name, Vector2 pos, float scale, float alpha) {
        if (frames.find(name) == frames.end()) return;
        Rectangle src = frames[name];
        Rectangle dest = { pos.x, pos.y, src.width * scale, src.height * scale };
        Vector2 origin = { dest.width/2, dest.height/2 };
        DrawTexturePro(texture, src, dest, origin, 0.0f, Fade(WHITE, alpha));
    }
};

// ---------------------------------------------------------
// 2. DATABASE MANAGER
// ---------------------------------------------------------
struct EyeDatabase {
    struct Entry {
        std::string name;
        EyeParams params;
    };
    
    std::vector<Entry> entries;
    std::string dropdownStr;

    void Load(const char* filename) {
        entries.clear();
        dropdownStr = "";
        
        std::ifstream in(filename);
        if (!in.is_open()) return;

        std::string line;
        while (std::getline(in, line)) {
            // Very basic Lua-style table parser
            size_t namePos = line.find("eyes[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                if (endPos != std::string::npos) {
                    Entry e;
                    e.params = EyeParams(); // Reset to defaults

                    e.name = line.substr(namePos + 6, endPos - (namePos + 6));
                    size_t bracePos = line.find("{");
                    
                    if (bracePos != std::string::npos) {
                        std::string valStr = line.substr(bracePos + 1);
                        
                        // Clean syntax: remove }, comma, f suffix
                        std::replace(valStr.begin(), valStr.end(), '}', ' '); 
                        std::replace(valStr.begin(), valStr.end(), ',', ' ');
                        std::replace(valStr.begin(), valStr.end(), 'f', ' ');

                        std::stringstream ss(valStr);
                        
                        // Temporary ints for boolean fields
                        int tShowBrow, tUseLowerBrow, tShowTears, tShowBlush, tDistortMode, tBlushMode;

                        // --- LOAD SEQUENCE (Matches Struct Order) ---
                        ss >> e.params.eyeShapeID 
                           >> e.params.bend 
                           >> e.params.eyeThickness 
                           >> e.params.eyeSide 
                           >> e.params.scaleX 
                           >> e.params.scaleY
                           >> e.params.angle 
                           >> e.params.spacing
                           >> e.params.squareness

                           // Surface FX
                           >> e.params.stressLevel 
                           >> e.params.gloomLevel 
                           >> tDistortMode
                           >> e.params.spiralSpeed

                           // Look Targets
                           >> e.params.lookX 
                           >> e.params.lookY

                           // Eyebrow
                           >> tShowBrow 
                           >> tUseLowerBrow 
                           >> e.params.eyebrowThickness 
                           >> e.params.eyebrowLength 
                           >> e.params.eyebrowSpacing
                           >> e.params.eyebrowX 
                           >> e.params.eyebrowY 
                           >> e.params.browScale 
                           >> e.params.browSide 
                           >> e.params.browAngle 
                           >> e.params.browBend 
                           >> e.params.browBendOffset

                           // Tears & Blush
                           >> tShowTears 
                           >> tShowBlush 
                           >> e.params.tearsLevel 
                           >> tBlushMode
                           >> e.params.blushScale 
                           >> e.params.blushX 
                           >> e.params.blushY 
                           >> e.params.blushSpacing

                           //Pixelation
                           >> e.params.pixelation;

                        // Cast ints back to types
                        e.params.distortMode = tDistortMode;
                        e.params.showBrow = (bool)tShowBrow;
                        e.params.useLowerBrow = (bool)tUseLowerBrow;
                        e.params.showTears = (bool)tShowTears;
                        e.params.showBlush = (bool)tShowBlush;
                        e.params.blushMode = tBlushMode;

                        // Safety Defaults
                        if (e.params.scaleX == 0) e.params.scaleX = 1.0f;
                        if (e.params.scaleY == 0) e.params.scaleY = 1.0f;
                        if (e.params.spacing == 0) e.params.spacing = 200.0f;

                        entries.push_back(e);
                    }
                }
            }
        }

        // Rebuild Dropdown
        for (size_t i = 0; i < entries.size(); i++) {
            dropdownStr += entries[i].name;
            if (i < entries.size() - 1) dropdownStr += ";";
        }
        std::cout << "[Database] Loaded " << entries.size() << " presets." << std::endl;
    }

    void Save(const char* filename, std::string name, EyeParams p) {
        std::ifstream infile(filename);
        std::vector<std::string> lines;
        std::string line;
        bool found = false;

        std::stringstream newLine;
        newLine << "eyes[\"" << name << "\"] = { "
                // Main Eye
                << p.eyeShapeID << "f, " << p.bend << "f, " << p.eyeThickness << "f, " << p.eyeSide << "f, "
                << p.scaleX << "f, " << p.scaleY << "f, " << p.angle << "f, " << p.spacing << "f, " << p.squareness << "f, "

                // Surface FX
                << p.stressLevel << "f, " << p.gloomLevel << "f, " << p.distortMode << ", " << p.spiralSpeed << "f, "

                // Look Targets
                << p.lookX << "f, " << p.lookY << "f, "

                // Eyebrow
                << (int)p.showBrow << ", " << (int)p.useLowerBrow << ", "
                << p.eyebrowThickness << "f, " << p.eyebrowLength << "f, "
                << p.eyebrowSpacing << "f, " << p.eyebrowX << "f, " << p.eyebrowY << "f, "
                << p.browScale << "f, " << p.browSide << "f, " 
                << p.browAngle << "f, " << p.browBend << "f, " << p.browBendOffset << "f, "

                // Tears & Blush
                << (int)p.showTears << ", " << (int)p.showBlush << ", "
                << p.tearsLevel << "f, " << p.blushMode << ", "
                << p.blushScale << "f, " << p.blushX << "f, " << p.blushY << "f, " << p.blushSpacing << "f, "

                //pixelation
                << p.pixelation << "f };"; 

        if (infile.is_open()) {
            while (std::getline(infile, line)) {
                if (line.find("eyes[\"" + name + "\"]") != std::string::npos) {
                    lines.push_back(newLine.str());
                    found = true;
                } else {
                    lines.push_back(line);
                }
            }
            infile.close();
        }

        if (!found) lines.push_back(newLine.str());

        std::ofstream outfile(filename);
        for (const auto& l : lines) outfile << l << "\n";

        std::cout << "[Database] Saved Preset: " << name << std::endl;
        Load(filename);
    }
};

// ---------------------------------------------------------
// EDITOR STATE WRAPPER
// ---------------------------------------------------------
struct EditorState {
    EyeParams currentParams;
    int currentIdx = 0;
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false;
    bool enableGUI = true;
    bool debugBoxes = false;
    
    // Dropdown UI state
    int dropdownActive = 0;
    bool dropdownEditMode = false;
    
    // Helper to cycle reference sprites
    void CycleReference(ReferenceAtlas& atlas, int dir) {
        if (atlas.eyeNames.empty()) return;
        currentIdx += dir;
        if (currentIdx < 0) currentIdx = (int)atlas.eyeNames.size() - 1;
        if (currentIdx >= (int)atlas.eyeNames.size()) currentIdx = 0;
    }
};

// ---------------------------------------------------------
// GUI PANELS
// ---------------------------------------------------------
void DrawReferencePanel(float& y, EditorState& state, ReferenceAtlas& atlas, EyeDatabase& db) {
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 110}, "SPRITE REFERENCE");
    
    if (GuiButton({UI::START_X + 10, y + 30, 40, 30}, "<")) state.CycleReference(atlas, -1);
    if (GuiButton({UI::START_X + 270, y + 30, 40, 30}, ">")) state.CycleReference(atlas, 1);
    
    std::string lbl = atlas.eyeNames.empty() ? "NONE" : atlas.eyeNames[state.currentIdx];
    GuiLabel({UI::START_X + 60, y + 30, 200, 30}, lbl.c_str());

    if (GuiButton({UI::START_X + 10, y + 70, UI::PANEL_WIDTH - 20, 30}, "SAVE PRESET (Enter)")) {
        std::string name = atlas.eyeNames.empty() ? "custom" : atlas.eyeNames[state.currentIdx];
        db.Save("eyes_database.txt", name, state.currentParams);
    }
    y += 120.0f;
}

void DrawMainSettings(float& y, EyeParams& p) {
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 350}, "MAIN EYE SETTINGS");
    y += 20.0f;

    // Shape Selector
    int shapeInt = (int)p.eyeShapeID;
    GuiHelper::Slider("Shape ID", &p.eyeShapeID, 0.0f, 12.0f, y);
    
    const char* shapeNames[] = { "Dot", "Line", "Arc", "Cross", "Star", "Heart", "Spiral", "Chevron", "Shuriken", "Kawaii", "Shocked", "Teary", "Colon Eyes" };
    if(shapeInt >= 0 && shapeInt <= 12) {
        GuiLabel({UI::START_X + 10 + UI::LABEL_WIDTH, y - 25, UI::GetSliderWidth(), 20}, shapeNames[shapeInt]);
    }

    if(p.eyeShapeID > 5.5f && p.eyeShapeID < 6.5f) {
        GuiHelper::Slider("Spiral Spd", &p.spiralSpeed, -10.0f, 10.0f, y);
    }

    GuiHelper::Slider("Bend", &p.bend, -2.0f, 2.0f, y);
    GuiHelper::Slider("Thickness", &p.eyeThickness, 1.0f, 30.0f, y);
    
    y += 5; // Spacer
    GuiHelper::Slider("Scale X", &p.scaleX, 0.1f, 10.0f, y);
    GuiHelper::Slider("Scale Y", &p.scaleY, 0.1f, 10.0f, y);
    GuiHelper::Slider("Spacing", &p.spacing, 0.0f, 1000.0f, y);
    GuiHelper::Slider("Look X", &p.lookX, -300.0f, 300.0f, y);
    GuiHelper::Slider("Look Y", &p.lookY, -300.0f, 300.0f, y);
    GuiHelper::Slider("Angle", &p.angle, -180.0f, 180.0f, y);
    GuiHelper::Slider("Squareness", &p.squareness, 0.0f, 1.0f, y);
    GuiHelper::Slider("Pixelation", &p.pixelation,1.0f, 8.0f, y);
    
    y += 20.0f; // Padding bottom
}

void DrawElementsPanel(float& y, EyeParams& p) {
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 480}, "ELEMENTS & FX");
    y += 20.0f;

    // Toggles
    GuiHelper::Checkbox("Brows", &p.showBrow, 10, y);
    GuiHelper::Checkbox("Tears", &p.showTears, 90, y);
    GuiHelper::Checkbox("Blush", &p.showBlush, 170, y);
    y += 30.0f;

    if (p.showBrow) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- BROW SETTINGS -"); y+= 20;
        GuiHelper::Slider("Thick", &p.eyebrowThickness, 1, 20, y);
        GuiHelper::Slider("Len", &p.eyebrowLength, 0.5f, 2.0f, y);
        GuiHelper::Slider("Spacing", &p.eyebrowSpacing, -100.0f, 100.0f, y);
        GuiHelper::Slider("Pos X", &p.eyebrowX, -10, 10, y);
        GuiHelper::Slider("Pos Y", &p.eyebrowY, -10, 10, y);
        GuiHelper::Slider("Scale", &p.browScale, 0.5f, 2.0f, y);
        GuiHelper::Slider("Angle", &p.browAngle, -45.0f, 45.0f, y);
        GuiHelper::Slider("Bend", &p.browBend, -2.0f, 2.0f, y);
        GuiHelper::Slider("Bend Off", &p.browBendOffset, 0.0f, 0.99f, y);
        GuiHelper::Checkbox("Use Lower Brow", &p.useLowerBrow, 10, y); y += 25;
        y += 10;
    }

    if (p.showTears) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- TEAR SETTINGS -"); y+= 20;
        GuiHelper::Slider("Level", &p.tearsLevel, 0, 1, y);
        y += 10;
    }

    if(p.showBlush) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- BLUSH SETTINGS -"); y+= 20;
        GuiHelper::Slider("Scale", &p.blushScale, 0.5f, 3.0f, y);
        GuiHelper::Slider("Pos X", &p.blushX, -10.0f, 10.0f, y);
        GuiHelper::Slider("Pos Y", &p.blushY, -10.0f, 10.0f, y);
        GuiHelper::Slider("Space", &p.blushSpacing, -100.0f, 100.0f, y);

        GuiLabel({UI::START_X+10, y, UI::LABEL_WIDTH, 20}, "Blush Mode");
        //Adding button to toggle between 3 modes 
        if(GuiButton({UI::START_X+10+UI::LABEL_WIDTH,  y, 80, 20}, p.blushMode == 0 ? "Pink" : (p.blushMode == 1 ? "Green" : "Yellow"))) {
            p.blushMode = (p.blushMode + 1) % 3;
        }
        std::cout << "Blush Mode: " << p.blushMode << std::endl;
        y += 25;
        y += 10;
    }

    GuiLabel({UI::START_X+10, y, UI::PANEL_WIDTH, 20}, "--- SURFACE FX ---"); y += 20;
    GuiHelper::Slider("Stress", &p.stressLevel, 0.0f, 1.0f, y);
    GuiHelper::Slider("Gloom", &p.gloomLevel, 0.0f, 1.0f, y);
    
    bool distMode = (p.distortMode == 1);
    if (GuiHelper::Checkbox("Squash/Stretch Distortion", &distMode, 10, y)) {
        p.distortMode = distMode ? 1 : 0;
    }
}

void DrawDatabasePanel(float screenW, EditorState& state, EyeDatabase& db) {
    const float panelW = 250.0f;
    const float marginR = 16.0f;
    const float marginT = 16.0f;
    float vx = screenW - panelW - marginR;
    float vy = marginT;

    // DB Load Box
    GuiGroupBox({ vx, vy, panelW, 120 }, "DATABASE LOAD");
    
    if (GuiDropdownBox({ vx + 10, vy + 30, panelW - 20, 25 }, db.dropdownStr.c_str(), &state.dropdownActive, state.dropdownEditMode))
        state.dropdownEditMode = !state.dropdownEditMode;

    if (GuiButton({ vx + 10, vy + 65, 160, 30 }, "LOAD SELECTED")) {
        if (!db.entries.empty() && state.dropdownActive < (int)db.entries.size()) {
            state.currentParams = db.entries[state.dropdownActive].params;
        }
    }
    if (GuiButton({ vx + panelW - 60, vy + 65, 50, 30 }, "RELOAD")) db.Load("eyes_database.txt");

    // Viewport Box
    float vyView = vy + 120 + 12.0f;
    GuiGroupBox({ vx, vyView, panelW, 110 }, "VIEWPORT");
    GuiCheckBox({ vx + 10, vyView + 25, 20, 20 }, "Show Ref", &state.showReference);
    GuiSliderBar({ vx + 80, vyView + 55, 100, 20 }, "Opac", nullptr, &state.refOpacity, 0.0f, 1.0f);
    GuiCheckBox({ vx + 10, vyView + 80, 20, 20 }, "Test Physics", &state.usePhysics);
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    std::cout << "BasePath: " << GetApplicationDirectory() << std::endl;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 800, "BMO Face Engine");
    SetTargetFPS(60);

    // Style
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt(BLACK));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
    GuiSetStyle(SLIDER, BASE_COLOR_NORMAL, ColorToInt({ 60, 60, 60, 255 }));
    GuiSetStyle(SLIDER, BASE_COLOR_FOCUSED, ColorToInt({ 120, 180, 255, 255 }));
    
    // Components
    ParametricEyes rig;
    rig.Init();
    
    EyeDatabase db;
    db.Load("eyes_database.txt");

    ReferenceAtlas atlas;
    atlas.Load("assets/BMO_Animation_LipSyncSprite.png", "assets/BMO_Animation_Lipsync.json");

    EditorState state;
    // Set default defaults
    state.currentParams.scaleX = 1.0f;
    state.currentParams.scaleY = 1.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        GlobalScaler.Update();
        
        Vector2 centerScreen = { (float)GetScreenWidth() * 0.5f, (float)GetScreenHeight() * 0.5f };

        // Logic
        rig.usePhysics = state.usePhysics;
        rig.debugBoxes = state.debugBoxes;
        rig.Update(dt, state.currentParams);

        // Drawing
        BeginDrawing();
        ClearBackground({201, 228, 195, 255}); // BMO Face Color

        // 1. Reference
        if (state.showReference && !atlas.eyeNames.empty()) {
            atlas.Draw(atlas.eyeNames[state.currentIdx], centerScreen, 1.0f, state.refOpacity);
        }

        // 2. Parametric Eyes
        EyeParams renderP = state.currentParams;
        if (state.usePhysics) {
            renderP.scaleX = rig.sScaleX.val;
            renderP.scaleY = rig.sScaleY.val;
            renderP.lookX  = rig.sLookX.val;
            renderP.lookY  = rig.sLookY.val;
        }
        rig.Draw(centerScreen, renderP, BLACK);

        // 3. GUI
        float screenW = (float)GetScreenWidth();
        float screenH = (float)GetScreenHeight();

        // Bottom Right Toggles
        float cbX = screenW - 180;
        float cbY = screenH - 40;
        GuiGroupBox({cbX - 10, cbY - 30, 170, 50}, "");
        GuiCheckBox({cbX-5, cbY-5, 20, 20}, "Enable GUI", &state.enableGUI);
        GuiCheckBox({cbX-5, cbY-30, 20, 20}, "Debug Boxes", &state.debugBoxes);

        // Fullscreen Toggle
        if (IsKeyPressed(KEY_F11)) {
            ToggleFullscreen();
            if (IsWindowFullscreen()) {
                 int display = GetCurrentMonitor();
                 SetWindowSize(GetMonitorWidth(display), GetMonitorHeight(display));
            } else {
                 SetWindowSize(1280, 720);
            }
        }

        // Main Editor Panels
        if(state.enableGUI) {
            float y = UI::START_Y;
            DrawReferencePanel(y, state, atlas, db);
            DrawMainSettings(y, state.currentParams);
            DrawElementsPanel(y, state.currentParams);
            DrawDatabasePanel(screenW, state, db);
        }

        // Keyboard Shortcuts
        if (IsKeyPressed(KEY_LEFT)) state.CycleReference(atlas, -1);
        if (IsKeyPressed(KEY_RIGHT)) state.CycleReference(atlas, 1);
        if (IsKeyPressed(KEY_ENTER)) db.Save("eyes_database.txt", atlas.eyeNames.empty() ? "custom" : atlas.eyeNames[state.currentIdx], state.currentParams);

        EndDrawing();
    }

    rig.Unload();
    CloseWindow();
    return 0;
}