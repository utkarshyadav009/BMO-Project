// SFacePoser.cpp
// DRIVER FILE: Links the GUI to the Shader-Based Mouth Engine

#include "raylib.h"

// 1. SETUP RAYGUI
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

// Include the NEW Shader-based Engine
// Ensure ParametricMouth.cpp is in the same folder (src/)
#include "ShaderParametricMouth.cpp" 

using json = nlohmann::json;

// ---------------------------------------------------------
// 1. ATLAS LOADER (Standard)
// ---------------------------------------------------------
struct ReferenceAtlas {
    Texture2D texture;
    std::unordered_map<std::string, Rectangle> frames;
    std::vector<std::string> mouthNames; 

    void Load(const char* img, const char* data) {
        texture = LoadTexture(img);
        if (texture.id == 0) return; // Safety

        std::ifstream f(data);
        if (!f.good()) return;

        json j = json::parse(f, nullptr, false);
        if (j.is_discarded()) return;

        if (j.contains("textures")) {
            for (auto& t : j["textures"]) {
                for (auto& fr : t["frames"]) {
                    std::string name = fr["filename"];
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
    // MSAA is irrelevant for the shader mouth (it handles its own AA), 
    // but useful for the GUI elements.
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 1000, "BMO Face Poser (Shader Edition)");
    SetTargetFPS(60);

    // Style Setup
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
    // Initial Position (Right side of screen)
    rig.Init({ 800, 360 }); 

    ReferenceAtlas atlas;
    atlas.Load("assets/BMO_Animation_LipSyncSprite.png",
               "assets/BMO_Animation_Lipsync.json");

    int currentIdx = 0;
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false; 

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        
        // --- LOGIC ---
        rig.usePhysics = usePhysics;
        rig.UpdatePhysics(dt);
        rig.GenerateGeometry();

        // --- DRAWING ---
        BeginDrawing();
        ClearBackground({131, 220, 169, 255}); // BMO Body Color

        // 1. Draw Reference (Behind)
        if (showReference && !atlas.mouthNames.empty()) {
            atlas.Draw(atlas.mouthNames[currentIdx], rig.centerPos, 1.0f, refOpacity);
        }

        // 2. Draw Shader Mouth (NOW VECTOR BASED!)
        rig.Draw();

        // 3. Draw GUI (Left Panel)
        float startX = 20.0f;
        float startY = 20.0f;
        float w = 300.0f;
        
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

        if (GuiButton({startX + 10, startY + 70, w - 20, 30}, "SAVE PRESET (Enter)")) {
            if (!atlas.mouthNames.empty()) AppendToDatabase(atlas.mouthNames[currentIdx], rig.target);
        }

        startY += 140.0f;
        float sy = startY + 25.0f;
        float innerX = startX + 10.0f;
        float innerW = w - 20.0f;
        const float labelW = 105.0f;
        const float valueW = 60.0f;
        const float colGap = 8.0f;
        const float rowH = 20.0f;

        #define DRAW_SLIDER(label, val, mn, mx) \
            GuiLabel({ innerX, sy, labelW, rowH }, label); \
            GuiSliderBar({ innerX + labelW + colGap, sy, innerW - labelW - valueW - colGap*2, rowH }, NULL, NULL, &(val), (mn), (mx)); \
            GuiLabel({ innerX + innerW - valueW, sy, valueW, rowH }, TextFormat("%.2f", (val))); \
            sy += 25.0f;

        DRAW_SLIDER("Scale",      rig.target.scale,         0.5f,  4.0f);
        sy += 10.0f;
        DRAW_SLIDER("Open",       rig.target.open,          0.0f,  1.2f);
        DRAW_SLIDER("Width",      rig.target.width,         0.1f,  1.5f);
        DRAW_SLIDER("Curve",      rig.target.curve,        -1.0f,  1.0f);
        sy += 10.0f;
        DRAW_SLIDER("Sqze Top",   rig.target.squeezeTop,    0.0f,  1.0f);
        DRAW_SLIDER("Sqze Bot",   rig.target.squeezeBottom, 0.0f,  1.0f);
        sy += 10.0f;
        DRAW_SLIDER("Teeth Y",    rig.target.teethY,       -1.0f,  1.0f);
        DRAW_SLIDER("Teeth W",    rig.target.teethWidth,    0.1f,  0.95f);
        DRAW_SLIDER("Teeth Gap",  rig.target.teethGap,      0.0f,  100.0f);
        sy += 10.0f;
        DRAW_SLIDER("Tongue Up",  rig.target.tongueUp,      0.0f,  1.0f);
        DRAW_SLIDER("Tongue W",   rig.target.tongueWidth,   0.3f,  1.0f);
        DRAW_SLIDER("Tongue X",   rig.target.tongueX,      -1.0f,  1.0f);
        sy += 10.0f;
        DRAW_SLIDER("Asymmetry",  rig.target.asymmetry,    -1.0f,  1.0f);
        DRAW_SLIDER("Squareness", rig.target.squareness,    0.0f,  1.0f);

        if (GuiButton({ innerX, sy, innerW, 30 }, "RESET TO NEUTRAL")) {
             rig.target = MakeDefaultParams();
        }
        sy += 40.0f;

        GuiGroupBox({ startX, startY, w, sy - startY }, "RIG PARAMETERS");

        // Viewport Settings
        float viewY = sy + startY - 100.0f; 
        GuiGroupBox({startX, viewY, w, 120}, "VIEWPORT");
        GuiCheckBox({startX + 10, viewY + 25, 20, 20}, NULL, &showReference);
        GuiLabel({startX + 35, viewY + 25, 120, 20}, "Show Ref");
        
        GuiLabel({startX + 10, viewY + 55, 60, 20}, "Opacity");
        GuiSliderBar({startX + 80, viewY + 55, 160, 20}, NULL, NULL, &refOpacity, 0.0f, 1.0f);
        
        GuiCheckBox({startX + 10, viewY + 85, 20, 20}, NULL, &usePhysics);
        GuiLabel({startX + 35, viewY + 85, 120, 20}, "Use Physics");

        // Shortcuts
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