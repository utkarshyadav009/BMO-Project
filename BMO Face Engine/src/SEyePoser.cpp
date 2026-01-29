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
#include"utility.h"


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
        if (!in.is_open()) return;

        std::string line;
        while (std::getline(in, line)) {
            size_t namePos = line.find("eyes[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                if (endPos != std::string::npos) {
                    Entry e;
                    // Initialize defaults to be safe in case line is short
                    e.params = EyeParams(); 

                    e.name = line.substr(namePos + 6, endPos - (namePos + 6));

                    size_t bracePos = line.find("{");
                    if (bracePos != std::string::npos) {
                        std::string valStr = line.substr(bracePos + 1);

                        // Clean up syntax characters
                        std::replace(valStr.begin(), valStr.end(), '}', ' '); 
                        std::replace(valStr.begin(), valStr.end(), ',', ' ');
                        std::replace(valStr.begin(), valStr.end(), 'f', ' ');

                        std::stringstream ss(valStr);

                        // Temp ints for bool conversion
                        int tShowBrow, tShowTears, tShowBlush, tUseLowerBrow;

                        // --- READ VARIABLES (MUST MATCH SAVE ORDER) ---
                        ss  // 1. MAIN
                            >> e.params.eyeShapeID 
                            >> e.params.bend 
                            >> e.params.eyeThickness 
                            >> e.params.pupilSize 
                            >> e.params.eyeSide

                            // 2. SURFACE FX
                            >> e.params.stressLevel 
                            >> e.params.gloomLevel 
                            >> e.params.distortMode

                            // 3. TOGGLES (Read as Ints)
                            >> tShowBrow 
                            >> tShowTears 
                            >> tShowBlush

                            // 4. BROW DETAILS
                            >> e.params.eyebrowType 
                            >> e.params.eyebrowThickness 
                            >> e.params.eyebrowY 
                            >> e.params.eyebrowX 
                            >> e.params.eyebrowLength 
                            >> e.params.browScale 
                            >> e.params.browSide 
                            >> e.params.browAngle 
                            >> e.params.browBend 
                            >> e.params.browBendOffset 
                            >> tUseLowerBrow // (Int)

                            // 5. TEARS
                            >> e.params.tearsLevel 
                            >> e.params.blushMode

                            // 6. EXTRAS
                            >> e.params.spiralSpeed

                            // 7. PHYSICS/LAYOUT
                            >> e.params.lookX 
                            >> e.params.lookY 
                            >> e.params.scaleX 
                            >> e.params.scaleY 
                            >> e.params.spacing;

                        // Convert Ints back to Bools
                        e.params.showBrow     = (bool)tShowBrow;
                        e.params.showTears    = (bool)tShowTears;
                        e.params.showBlush    = (bool)tShowBlush;
                        e.params.useLowerBrow = (bool)tUseLowerBrow;

                        // Legacy/Safety Checks
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
        std::ifstream infile(filename);
        std::vector<std::string> lines;
        std::string line;
        bool found = false;

        std::stringstream newLine;
        newLine << "eyes[\"" << name << "\"] = { "
                // 1. MAIN
                << p.eyeShapeID << "f, " << p.bend << "f, " << p.eyeThickness << "f, "
                << p.pupilSize << "f, " << p.eyeSide << "f, "

                // 2. SURFACE FX
                << p.stressLevel << "f, " << p.gloomLevel << "f, " << p.distortMode << ", "

                // 3. TOGGLES (Cast bool to int for safety)
                << (int)p.showBrow << ", " << (int)p.showTears << ", " << (int)p.showBlush << ", "

                // 4. BROW DETAILS
                << p.eyebrowType << "f, " << p.eyebrowThickness << "f, " 
                << p.eyebrowY << "f, " << p.eyebrowX << "f, " << p.eyebrowLength << "f, "
                << p.browScale << "f, " << p.browSide << "f, " << p.browAngle << "f, "
                << p.browBend << "f, " << p.browBendOffset << "f, " 
                << (int)p.useLowerBrow << ", "

                // 5. TEARS
                << p.tearsLevel << "f, " << p.blushMode << ", "

                // 6. EXTRAS
                << p.spiralSpeed << "f, "

                // 7. PHYSICS/LAYOUT
                << p.lookX << "f, " << p.lookY << "f, "
                << p.scaleX << "f, " << p.scaleY << "f, " << p.spacing << "f };";

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

        std::cout << "Saved Eye Preset: " << name << std::endl;
        Load(filename);
    }
};

// ---------------------------------------------------------
// MAIN EDITOR
// ---------------------------------------------------------
int main() {

    std::cout << "BasePath: " << GetApplicationDirectory() << std::endl;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE|FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    // Get the actual physical size of the primary monitor
    int monitor = GetCurrentMonitor();
    int screenW = GetMonitorWidth(monitor);
    int screenH = GetMonitorHeight(monitor);

    // Initialize window with native resolution (borderless experience)
    InitWindow(1280, 800, "BMO Face Engine");
    
    // Immediately fullscreen it to cover taskbars etc.
    //ToggleFullscreen();
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
    
    Vector2 centerScreen = { 0, 0 };

    // GUI Layout Vars
    float startX = 20.0f; 
    float startY = 20.0f; 
    float w = 320.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        GlobalScaler.Update();

        // Always lock face to the exact center of the current window
        centerScreen.x = (float)GetScreenWidth() * 0.5f;
        centerScreen.y = (float)GetScreenHeight() * 0.5f;

        //For GUI 
        float screenWidth = (float)GetScreenWidth();
        float screenHeight = (float)GetScreenHeight();
        

        // --- LOGIC ---
        rig.usePhysics = usePhysics;
        rig.Update(dt, currentParams);

        // --- DRAWING ---
        BeginDrawing();
        //ClearBackground({131, 220, 169, 255}); // BMO Body Color
        ClearBackground({201, 228, 195, 255}); // BMO Face Color

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

        // Toggle to control whether the large GUI block should be drawn
        static bool enableGUI = true;

        // Bottom-right "Enable GUI" checkbox (positioned relative to window size)
        float cbX = screenWidth - 180;
        float cbY = screenHeight - 40;
        GuiGroupBox({cbX - 10, cbY - 30, 170, 50}, "");
        GuiCheckBox({cbX-5, cbY-5, 20, 20}, "Enable GUI", &enableGUI);
        GuiCheckBox({cbX-5, cbY-30, 20, 20}, "Enable debug", &rig.debugBoxes);



        // Fullscreen toggle 
        if (IsKeyPressed(KEY_F11)) {
            ToggleFullscreen();

            // Optional: Reset window size if leaving fullscreen
            if (!IsWindowFullscreen()) {
                SetWindowSize(1280, 720); // Go back to a manageable window
                SetWindowPosition(100, 100);
            } else {
                // If entering fullscreen, Raylib usually handles the resize,
                // but explicit sizing ensures it grabs the monitor res.
                int display = GetCurrentMonitor();
                SetWindowSize(GetMonitorWidth(display), GetMonitorHeight(display));
            }
        }

        // NOTE: Replace the later `if(true)` with `if(enableGUI)` so the main GUI block respects this checkbox.

        // --- 1. SPRITE REFERENCE (Top Left) ---
        if(enableGUI)
        {
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
            GUI_SLIDE(" Eye Thickness", currentParams.eyeThickness, 1.0f, 30.0f);
            GUI_SLIDE("Pupil/Hole", currentParams.pupilSize, 0.0f, 1.0f);

            sy += 5; // Spacer
            GUI_SLIDE("Scale X", currentParams.scaleX, 0.1f, 10.0f);
            GUI_SLIDE("Scale Y", currentParams.scaleY, 0.1f, 10.0f);
            GUI_SLIDE("Spacing", currentParams.spacing, 0.0f, 1000.0f);
            GUI_SLIDE("Look X", currentParams.lookX, -300.0f, 300.0f);
            GUI_SLIDE("Look Y", currentParams.lookY, -300.0f, 300.0f);

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
                GUI_SLIDE("Brow Type", currentParams.eyebrowType, 0, 4);
                GUI_SLIDE("Brow Thick", currentParams.eyebrowThickness, 1, 20);
                GUI_SLIDE("Brow Y", currentParams.eyebrowY, -10, 10);
                GUI_SLIDE("Brow X", currentParams.eyebrowX, -10, 10); // [NEW]
                GUI_SLIDE("Brow Len", currentParams.eyebrowLength, 0.5f, 2.0f); // [NEW]
                GUI_SLIDE("Brow Scale", currentParams.browScale, 0.5f, 2.0f);   // [NEW]
                GUI_SLIDE("Brow Angle", currentParams.browAngle, -45.0f, 45.0f); // [NEW]
                GUI_SLIDE("Brow Bend", currentParams.browBend, -2.0f, 2.0f);   // Re-use bend
                GUI_SLIDE("Brow Bend Off", currentParams.browBendOffset, 0.0f, 0.99f); // [NEW]
                GuiCheckBox({startX+10, sy, 20, 20}, "Lower Brow", &currentParams.useLowerBrow); // [NEW]
            }

            if (currentParams.showTears) {
                GUI_SLIDE("Tear Lvl", currentParams.tearsLevel, 0, 1);

                // Tear Mode Toggle
                GuiLabel({startX+10, sy, labelW, 20}, "Tear Mode");
                if (GuiButton({startX+10+labelW, sy, 80, 20}, currentParams.blushMode == 0 ? "Pink" : "Green")) {
                    currentParams.blushMode = !currentParams.blushMode;
                }
                printf("Blush Mode %i \n", currentParams.blushMode);
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
            // --- 4. RIGHT SIDE CONTROLS (Database & Viewport) ---
            // Anchor to TOP-RIGHT with margins
            const float panelW  = 250.0f;
            const float marginR = 16.0f;
            const float marginT = 16.0f;

            float vx = screenWidth - panelW - marginR;
            float vy = marginT;

            // ----- DATABASE LOAD -----
            const float dbH = 120.0f;
            GuiGroupBox({ vx, vy, panelW, dbH }, "DATABASE LOAD");

            // Dropdown
            Rectangle dd = { vx + 10, vy + 30, panelW - 20, 25 };
            if (GuiDropdownBox(dd, db.dropdownStr.c_str(), &dropdownActive, dropdownEditMode))
                dropdownEditMode = !dropdownEditMode;

            // Buttons row
            Rectangle loadBtn   = { vx + 10,           vy + 65, 160, 30 };
            Rectangle reloadBtn = { vx + panelW - 60,  vy + 65,  50, 30 };

            if (GuiButton(loadBtn, "LOAD SELECTED")) {
                if (!db.entries.empty() && dropdownActive < (int)db.entries.size()) {
                    currentParams = db.entries[dropdownActive].params;
                }
            }
            if (GuiButton(reloadBtn, "RELOAD")) db.Load("eyes_database.txt");

            // ----- VIEWPORT -----
            const float gapH  = 12.0f;
            const float viewH = 100.0f;

            float vyView = vy + dbH + gapH;
            GuiGroupBox({ vx, vyView, panelW, viewH }, "VIEWPORT");

            GuiCheckBox({ vx + 10, vyView + 25, 20, 20 }, "Show Ref", &showReference);
            GuiSliderBar({ vx + 80, vyView + 55, 100, 20 }, "Opac", nullptr, &refOpacity, 0.0f, 1.0f);
            GuiCheckBox({ vx + 10, vyView + 80, 20, 20 }, "Test Physics", &usePhysics);

        }

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