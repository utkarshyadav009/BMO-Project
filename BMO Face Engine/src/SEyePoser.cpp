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

// Include the NEW Shader-based Eye Engine
// Ensure ShaderParametricEyes.cpp is in the same folder or src/
#include "ShaderParametricEyes.cpp" 

using json = nlohmann::json;

// ---------------------------------------------------------
// 1. ATLAS LOADER (Filters for "Eye" sprites)
// ---------------------------------------------------------
struct ReferenceAtlas {
    Texture2D texture;
    std::unordered_map<std::string, Rectangle> frames;
    std::vector<std::string> eyeNames; 

    void Load(const char* img, const char* data) {
        texture = LoadTexture(img);
        if (texture.id == 0) {
            std::cout << "[Atlas] Error loading texture: " << img << std::endl;
            return;
        }

        std::ifstream f(data);
        if (!f.good()) {
            std::cout << "[Atlas] Error loading JSON: " << data << std::endl;
            return;
        }

        json j = json::parse(f, nullptr, false);
        if (j.is_discarded()) return;

        if (j.contains("textures")) {
            for (auto& t : j["textures"]) {
                for (auto& fr : t["frames"]) {
                    std::string name = fr["filename"];
                    
                    // FILTER: Look for "eye" in filename (case insensitive)
                    std::string lowerName = name;
                    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                    
                    bool isEye = (lowerName.find("eye") != std::string::npos);
                    
                    if (isEye) {
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
// 2. DATABASE MANAGER (eyes_database.txt)
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
        if (!in.is_open()) return; // File might not exist yet

        std::string line;
        while (std::getline(in, line)) {
            // Simple Parse: eyes["NAME"] = { ... }
            size_t namePos = line.find("eyes[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                if (endPos != std::string::npos) {
                    Entry e;
                    e.name = line.substr(namePos + 6, endPos - (namePos + 6));
                    
                    // Extract numbers
                    size_t bracePos = line.find("{");
                    if (bracePos != std::string::npos) {
                        std::string valStr = line.substr(bracePos + 1);
                        std::replace(valStr.begin(), valStr.end(), '}', ' '); 
                        std::replace(valStr.begin(), valStr.end(), ',', ' ');
                        std::replace(valStr.begin(), valStr.end(), 'f', ' ');
                        
                        std::stringstream ss(valStr);
                        ss >> e.params.eyeShapeID >> e.params.bend >> e.params.thickness 
                           >> e.params.pupilSize >> e.params.lookX >> e.params.lookY 
                           >> e.params.scaleX >> e.params.scaleY >> e.params.spacing;
                        
                        // Default check if parsing failed for newer params
                        if (e.params.scaleX == 0) e.params.scaleX = 1.0f;
                        if (e.params.scaleY == 0) e.params.scaleY = 1.0f;
                        if (e.params.spacing == 0) e.params.spacing = 200.0f;
                        
                        entries.push_back(e);
                    }
                }
            }
        }
        
        // Build Dropdown String
        for (size_t i = 0; i < entries.size(); i++) {
            dropdownStr += entries[i].name;
            if (i < entries.size() - 1) dropdownStr += ";";
        }
    }

    void Save(const char* filename, std::string name, EyeParams p) {
        // Read all existing lines first
        std::ifstream infile(filename);
        std::vector<std::string> lines;
        std::string line;
        bool found = false;

        std::stringstream newLine;
        newLine << "eyes[\"" << name << "\"] = { "
                << p.eyeShapeID << "f, " << p.bend << "f, " << p.thickness << "f, "
                << p.pupilSize << "f, " << p.lookX << "f, " << p.lookY << "f, "
                << p.scaleX << "f, " << p.scaleY << "f, " << p.spacing << "f };";

        if (infile.is_open()) {
            while (std::getline(infile, line)) {
                // Check if this line is the one we are updating
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

        // Write back
        std::ofstream outfile(filename);
        for (const auto& l : lines) outfile << l << "\n";
        
        std::cout << "Saved Eye Preset: " << name << std::endl;
        
        // Reload RAM copy
        Load(filename);
    }
};

// ---------------------------------------------------------
// MAIN EDITOR
// ---------------------------------------------------------
int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 800, "BMO Eye Poser (Shader Edition)");
    SetTargetFPS(60);


    // Style Setup (Dark Theme)
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt(BLACK));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
    GuiSetStyle(SLIDER, BASE_COLOR_NORMAL, ColorToInt({ 60, 60, 60, 255 }));
    GuiSetStyle(SLIDER, BASE_COLOR_FOCUSED, ColorToInt({ 120, 180, 255, 255 }));
    
    // Rig & DB
    ParametricEyes rig;
    rig.Init();
    
    EyeDatabase db;
    db.Load("eyes_database.txt");

    // Atlas
    ReferenceAtlas atlas;
    // Assuming same atlas file, change filenames if different
    atlas.Load("assets/BMO_Animation_LipSyncSprite.png",
               "assets/BMO_Animation_Lipsync.json");

    // State
    EyeParams currentParams;
    currentParams.scaleX = 1.0f; 
    currentParams.scaleY = 1.0f;

    int currentIdx = 0;
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false; 
    
    int dropdownActive = 0;
    bool dropdownEditMode = false;
    
    Vector2 centerScreen = { 800, 360 };

    // GUI Layout Vars
    float startX = 20.0f; 
    float startY = 20.0f; 
    float w = 320.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        
        // --- LOGIC ---
        rig.usePhysics = usePhysics;
        rig.Update(dt, currentParams);

        // --- DRAWING ---
        BeginDrawing();
        ClearBackground({131, 220, 169, 255}); // BMO Body Color

        // 1. Draw Reference (Behind)
        if (showReference && !atlas.eyeNames.empty()) {
            atlas.Draw(atlas.eyeNames[currentIdx], centerScreen, 1.0f, refOpacity);
        }

        // 2. Draw Shader Eyes
        EyeParams renderP = currentParams;
        if (usePhysics) {
            renderP.scaleX = rig.sScaleX.val;
            renderP.scaleY = rig.sScaleY.val;
            renderP.lookX = rig.sLookX.val;
            renderP.lookY = rig.sLookY.val;
        }

        rig.Draw(centerScreen, renderP, BLACK); // Color overridden internally for specific IDs

        // --------------------------------------------------------
        // GUI PANELS
        // --------------------------------------------------------
        float sy = startY;

        // --- 1. SPRITE REFERENCE (Top Left) ---
        GuiGroupBox({startX, sy, w, 110}, "SPRITE REFERENCE");
        if (GuiButton({startX + 10, sy + 30, 40, 30}, "<")) {
            currentIdx--; if (currentIdx < 0) currentIdx = (int)atlas.eyeNames.size() - 1;
        }
        if (GuiButton({startX + 270, sy + 30, 40, 30}, ">")) {
            currentIdx = (int)((currentIdx + 1) % (int)atlas.eyeNames.size());
        }
        std::string lbl = atlas.eyeNames.empty() ? "NONE" : atlas.eyeNames[currentIdx];
        GuiLabel({startX + 60, sy + 30, 200, 30}, lbl.c_str());
        
        if (GuiButton({startX + 10, sy + 70, w - 20, 30}, "SAVE PRESET (Enter)")) {
            std::string name = atlas.eyeNames.empty() ? "custom" : atlas.eyeNames[currentIdx];
            db.Save("eyes_database.txt", name, currentParams);
        }
        sy += 120.0f;

        // --- 2. MAIN EYE SETTINGS (Left Column) ---
        GuiGroupBox({startX, sy, w, 320}, "MAIN EYE SETTINGS");
        sy += 20.0f;
        
        float labelW = 80; float valW = 40; float sliderW = w - labelW - valW - 30;
        
        #define GUI_SLIDE(txt, var, min, max) \
            GuiLabel({startX+10, sy, labelW, 20}, txt); \
            GuiSliderBar({startX+10+labelW, sy, sliderW, 20}, NULL, NULL, &var, min, max); \
            GuiLabel({startX+10+labelW+sliderW+5, sy, valW, 20}, TextFormat("%.2f", var)); \
            sy += 25;

        int shapeInt = (int)currentParams.eyeShapeID;
        GUI_SLIDE("Shape ID", currentParams.eyeShapeID, 0.0f, 10.0f); // Range 0-10
        
        // Contextual Label for Shape
        const char* shapeNames[] = { "Dot", "Line", "Arc", "Cross", "Star", "Heart", "Spiral", "Chevron", "Shuriken", "Kawaii", "Shocked" };
        if(shapeInt >= 0 && shapeInt <= 10) GuiLabel({startX+10+labelW, sy-25, sliderW, 20}, shapeNames[shapeInt]);

        if(currentParams.eyeShapeID > 5.5f && currentParams.eyeShapeID < 6.5f) {
            GUI_SLIDE("Spiral Spd", currentParams.spiralSpeed, -10.0f, 10.0f);
        }
        
        GUI_SLIDE("Bend", currentParams.bend, -2.0f, 2.0f);
        GUI_SLIDE("Thickness", currentParams.thickness, 1.0f, 30.0f);
        GUI_SLIDE("Pupil/Hole", currentParams.pupilSize, 0.0f, 1.0f);
        
        sy += 5; // Spacer
        GUI_SLIDE("Scale X", currentParams.scaleX, 0.1f, 3.0f);
        GUI_SLIDE("Scale Y", currentParams.scaleY, 0.0f, 3.0f);
        GUI_SLIDE("Spacing", currentParams.spacing, 0.0f, 1000.0f);
        GUI_SLIDE("Look X", currentParams.lookX, -200.0f, 200.0f);
        GUI_SLIDE("Look Y", currentParams.lookY, -200.0f, 200.0f);

        // --- 3. ELEMENTS & FX (Left Column, Below Main) ---
        sy += 25.0f;
        GuiGroupBox({startX, sy, w, 280}, "ELEMENTS & FX");
        sy += 20.0f;

        // Toggles Row
        GuiCheckBox({startX+10, sy, 20, 20}, "Brows", &currentParams.showBrow);
        GuiCheckBox({startX+90, sy, 20, 20}, "Tears", &currentParams.showTears);
        GuiCheckBox({startX+170, sy, 20, 20}, "Blush", &currentParams.showBlush);
        sy += 30.0f;

        if (currentParams.showBrow) {
            GUI_SLIDE("Brow Type", currentParams.eyebrowType, 0, 3);
            GUI_SLIDE("Brow Y", currentParams.eyebrowY, -100, 100);
            GUI_SLIDE("Brow Len", currentParams.eyebrowLength, 0.5f, 2.0f); // [NEW]
        }
        
        if (currentParams.showTears) {
            GUI_SLIDE("Tear Lvl", currentParams.tearsLevel, 0, 1);
            
            // Tear Mode Toggle
            GuiLabel({startX+10, sy, labelW, 20}, "Tear Mode");
            if (GuiButton({startX+10+labelW, sy, 80, 20}, currentParams.tearMode == 0 ? "DRIP" : "WAIL")) {
                currentParams.tearMode = !currentParams.tearMode;
            }
            sy += 25;
        }

        sy += 10; // Spacer for FX
        GuiLabel({startX+10, sy, w, 20}, "--- SURFACE FX ---");
        sy += 20;
        
        GUI_SLIDE("Stress (Angry)", currentParams.stressLevel, 0.0f, 1.0f); // [NEW]
        GUI_SLIDE("Gloom (Shock)", currentParams.gloomLevel, 0.0f, 1.0f);  // [NEW]
        
        // Distortion Toggle
        bool distMode = (currentParams.distortMode == 1);
        if (GuiCheckBox({startX+10, sy, 20, 20}, "Squash/Stretch Distortion", &distMode)) {
            currentParams.distortMode = distMode ? 1 : 0;
        }


        // --- 4. RIGHT SIDE CONTROLS (Database & Viewport) ---
        float vx = 1000; float vy = 20;

        GuiGroupBox({vx, vy, 250, 120}, "DATABASE LOAD");
        vy += 20;
        
        if (GuiDropdownBox({vx+10, vy, 230, 25}, db.dropdownStr.c_str(), &dropdownActive, dropdownEditMode)) {
            dropdownEditMode = !dropdownEditMode;
        }
        vy += 35;

        if (GuiButton({vx+10, vy, 160, 30}, "LOAD SELECTED")) {
            if (!db.entries.empty() && dropdownActive < db.entries.size()) {
                currentParams = db.entries[dropdownActive].params;
            }
        }
        if (GuiButton({vx+180, vy, 50, 30}, "RELOAD")) db.Load("eyes_database.txt");
        
        // VIEWPORT SETTINGS
        vy = 680; // Bottom Right
        GuiGroupBox({vx, vy, 250, 100}, "VIEWPORT");
        GuiCheckBox({vx+10, vy+25, 20, 20}, "Show Ref", &showReference);
        GuiSliderBar({vx+80, vy+55, 100, 20}, "Opac", NULL, &refOpacity, 0.0f, 1.0f);
        GuiCheckBox({vx+10, vy+80, 20, 20}, "Test Physics", &usePhysics);

        // KEYBOARD
        if (IsKeyPressed(KEY_LEFT)) currentIdx--;
        if (IsKeyPressed(KEY_RIGHT)) currentIdx++;
        if (currentIdx < 0) currentIdx = (int)atlas.eyeNames.size() - 1;
        if (currentIdx >= (int)atlas.eyeNames.size()) currentIdx = 0;
        
        EndDrawing();
    }

    rig.Unload();
    CloseWindow();
    return 0;
}