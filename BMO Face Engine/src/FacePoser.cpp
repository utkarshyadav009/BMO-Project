// FacePoser_GUI.cpp
// - A Visual Editor for the BMO Rig
// - Uses RayGui for Sliders/Buttons instead of Keyboard
// - Loads "BMO_Animation_Lipsync.json" to cycle through sprites

#include "raylib.h"

// 1. SETUP RAYGUI
#define RAYGUI_IMPLEMENTATION
#include "raygui.h" // Make sure this file is in your folder!

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

// Include your rig
#include "ParametricMouth.cpp" 

using json = nlohmann::json;

// ---------------------------------------------------------
// 1. ATLAS LOADER (Same as before)
// ---------------------------------------------------------
struct ReferenceAtlas {
    Texture2D texture;
    std::unordered_map<std::string, Rectangle> frames;
    std::vector<std::string> mouthNames; 

    void Load(const char* img, const char* data) {
        texture = LoadTexture(img);
        std::ifstream f(data);
        json j = json::parse(f);

        if (j.contains("textures")) {
            for (auto& t : j["textures"]) {
                for (auto& fr : t["frames"]) {
                    std::string name = fr["filename"];
                    
                    // Filter for Mouths only
                    bool isMouth = (name.find("_mouth") != std::string::npos) || 
                                   (name.find("mouth_phoneme") != std::string::npos);

                    if (isMouth) {
                        frames[name] = {
                            (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                            (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                        };
                        mouthNames.push_back(name);
                    }
                }
            }
        }
        std::sort(mouthNames.begin(), mouthNames.end());
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
// 2. SAVER
// ---------------------------------------------------------
void AppendToDatabase(std::string name, FacialParams p) {
    std::ofstream file("visemes_database.txt", std::ios::app);
    if (file.is_open()) {
        file << "visemes[\"" << name << "\"] = { "
        << p.open << "f, " << p.width << "f, " << p.curve << "f, "
        << p.squeezeTop << "f, " << p.squeezeBottom << "f, "
        << p.teethY << "f, "
        << p.tongueUp << "f, " << p.tongueX << "f, " << p.tongueWidth << "f, "
        << p.asymmetry << "f, " << p.squareness << "f, "
        << p.teethWidth << "f, " << p.teethGap << "f, "
        << p.scale << "f };\n";
        file.close();
        std::cout << "Saved: " << name << std::endl;
    }
}

// ---------------------------------------------------------
// MAIN EDITOR
// ---------------------------------------------------------
int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 1000, "BMO Face Poser (GUI Edition)");
    SetTargetFPS(60);

    // GUI Styling
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt(BLACK));
    GuiSetStyle(DEFAULT, BORDER_COLOR_NORMAL, ColorToInt(BLACK));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(BLACK));
    GuiSetStyle(DEFAULT, TEXT_COLOR_FOCUSED, ColorToInt(BLACK));

    GuiSetStyle(SLIDER, BASE_COLOR_NORMAL, ColorToInt({ 60, 60, 60, 255 }));
    GuiSetStyle(SLIDER, BASE_COLOR_FOCUSED, ColorToInt({ 120, 180, 255, 255 }));
    GuiSetStyle(SLIDER, BASE_COLOR_PRESSED, ColorToInt({ 120, 180, 255, 255 }));

    GuiSetStyle(BUTTON, BASE_COLOR_NORMAL, ColorToInt({ 70, 70, 70, 255 }));
    GuiSetStyle(BUTTON, TEXT_COLOR_NORMAL, ColorToInt(WHITE));

    GuiSetStyle(CHECKBOX, BORDER_COLOR_NORMAL, ColorToInt({ 200, 200, 200, 255 }));


    ParametricMouth rig;
    rig.Init({ 800, 360 }); // Move rig to the right side

    ReferenceAtlas atlas;
    atlas.Load("assets/BMO_Animation_LipSyncSprite.png",
               "assets/BMO_Animation_Lipsync.json");

    int currentIdx = 0;
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false; // Default to OFF for precise sculpting

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        // --- LEFT PANEL: CONTROLS ---
        float startX = 20.0f;
        float startY = 20.0f;
        float w = 300.0f;

        // Layout constants (prevents overlap)
        const float pad        = 10.0f;
        const float rowH       = 20.0f;
        const float rowGap     = 10.0f;
        const float spacerH    = 10.0f;
        const float btnH       = 30.0f;
        const float sectionGap = 16.0f;

        BeginDrawing();
        ClearBackground({131, 220, 169, 255}); // Dark Editor BG

        // 1. Navigation
        GuiGroupBox({startX, startY, w, 120}, "SPRITE BROWSER");
        if (GuiButton({startX + 10, startY + 30, 40, 30}, "<")) {
            currentIdx--;
            if (currentIdx < 0) currentIdx = (int)atlas.mouthNames.size() - 1;
        }
        if (GuiButton({startX + 250, startY + 30, 40, 30}, ">")) {
            currentIdx = (int)((currentIdx + 1) % (int)atlas.mouthNames.size());
        }

        std::string nameLabel = atlas.mouthNames.empty() ? "NONE" : atlas.mouthNames[currentIdx];
        GuiLabel({startX + 60, startY + 30, 180, 30}, nameLabel.c_str());

        // Save Button
        if (GuiButton({startX + 10, startY + 70, w - 20, 30}, "SAVE PRESET (Enter)")) {
            if (!atlas.mouthNames.empty()) AppendToDatabase(atlas.mouthNames[currentIdx], rig.target);
        }

        // 2. Parameters (dynamic height, no overlap)
        startY += 140.0f;

        float sy = startY + 25.0f;  // inside groupbox below title
        float innerX = startX + pad;
        float innerW = w - pad * 2.0f;

    // widths for label + value text columns
    const float labelW = 105.0f;
    const float valueW = 60.0f;
    const float colGap = 8.0f;

    #define DRAW_ROW_SLIDER(label, value, mn, mx)                                      \
        do {                                                                           \
            /* left label */                                                           \
            GuiLabel({ innerX, sy, labelW, rowH }, label);                             \
                                                                                       \
            /* slider in the middle */                                                 \
            Rectangle sld = { innerX + labelW + colGap, sy,                            \
                              innerW - labelW - valueW - colGap * 2.0f, rowH };        \
            GuiSliderBar(sld, NULL, NULL, &(value), (mn), (mx));                       \
                                                                                       \
            /* right value text */                                                     \
            GuiLabel({ sld.x + sld.width + colGap, sy, valueW, rowH },                 \
                     TextFormat("%.2f", (value)));                                     \
                                                                                       \
            sy += rowH + rowGap;                                                       \
        } while (0)

        DRAW_ROW_SLIDER("Scale",      rig.target.scale,         0.5f,  4.0f);
        sy += spacerH;
        DRAW_ROW_SLIDER("Open",       rig.target.open,          0.0f,  1.2f);
        DRAW_ROW_SLIDER("Width",      rig.target.width,         0.1f,  1.5f);
        DRAW_ROW_SLIDER("Curve",      rig.target.curve,        -1.0f,  1.0f);
        sy += spacerH;
        DRAW_ROW_SLIDER("Sqze Top",   rig.target.squeezeTop,    0.0f,  1.0f);
        DRAW_ROW_SLIDER("Sqze Bot",   rig.target.squeezeBottom, 0.0f,  1.0f);
        sy += spacerH;
        DRAW_ROW_SLIDER("Teeth Y",    rig.target.teethY,       -1.0f,  1.0f);
        DRAW_ROW_SLIDER("Teeth W",    rig.target.teethWidth,    0.1f,  0.95f);
        DRAW_ROW_SLIDER("Teeth Gap",  rig.target.teethGap,      0.0f,  100.0f);

        sy += spacerH;

        DRAW_ROW_SLIDER("Tongue Up",  rig.target.tongueUp,      0.0f,  1.0f);
        DRAW_ROW_SLIDER("Tongue W",   rig.target.tongueWidth,   0.3f,  1.0f);
        DRAW_ROW_SLIDER("Tongue X",   rig.target.tongueX,      -1.0f,  1.0f);

        sy += spacerH;

        DRAW_ROW_SLIDER("Asymmetry",  rig.target.asymmetry,    -1.0f,  1.0f);
        DRAW_ROW_SLIDER("Squareness", rig.target.squareness,    0.0f,  1.0f);


        // Reset Button
        Rectangle resetBtn = { innerX, sy, innerW, btnH };
        if (GuiButton(resetBtn, "RESET TO NEUTRAL")) {
            rig.target = { 0.05f, 0.5f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.65f, 0.0f, 0.0f, 0.5f, 45.0f, 0.0f };
        }
        sy += btnH + rowGap;

        // Draw the group box behind the contents with computed height
        float rigBoxH = (sy - startY) + pad; // include bottom padding
        GuiGroupBox({ startX, startY, w, rigBoxH }, "RIG PARAMETERS");

        // 3. View Settings (placed after rig section; no hardcoded startY += 470)
        float viewY = startY + rigBoxH + sectionGap;
        GuiGroupBox({startX, viewY, w, 120}, "VIEWPORT");

        // ---- Row 1: Show Reference ----
        Rectangle cb1 = { startX + 10, viewY + 25, 20, 20 };
        GuiCheckBox(cb1, NULL, &showReference);                 // no built-in label
        GuiLabel({ cb1.x + 28, cb1.y, 160, 20 }, "Show Reference");

        // ---- Row 2: Opacity ----
        // ---- Row 2: Opacity (label drawn separately so it never clips) ----
        GuiLabel({ startX + 10, viewY + 55, 80, 20 }, "Opacity");
        GuiSliderBar({ startX + 95, viewY + 55, w - 105, 20 }, NULL, NULL, &refOpacity, 0.0f, 1.0f);
        GuiLabel({ startX + w - 55, viewY + 55, 45, 20 }, TextFormat("%.2f", refOpacity));

        
        // ---- Row 3: Use Physics ----
        Rectangle cb2 = { startX + 10, viewY + 85, 20, 20 };
        GuiCheckBox(cb2, NULL, &usePhysics);
        GuiLabel({ cb2.x + 28, cb2.y, 160, 20 }, "Use Physics");
        rig.usePhysics = usePhysics;
        

        #undef DRAW_ROW_SLIDER

        // --- RIGHT PANEL: VIEWPORT ---
        // Draw Reference
        if (showReference && !atlas.mouthNames.empty()) {
            atlas.Draw(atlas.mouthNames[currentIdx], rig.centerPos, 1.0f, refOpacity);
        }

        // Draw Rig
        rig.UpdatePhysics(dt);
        rig.GenerateGeometry();
        rig.Draw();

        // Keyboard Shortcuts
        if (IsKeyPressed(KEY_LEFT)) currentIdx--;
        if (IsKeyPressed(KEY_RIGHT)) currentIdx++;
        if (currentIdx < 0) currentIdx = (int)atlas.mouthNames.size() - 1;
        if (currentIdx >= (int)atlas.mouthNames.size()) currentIdx = 0;

        if (IsKeyPressed(KEY_ENTER)) {
            if (!atlas.mouthNames.empty()) AppendToDatabase(atlas.mouthNames[currentIdx], rig.target);
        }

        EndDrawing();
    }

    rig.Unload();
    CloseWindow();
    return 0;
}
