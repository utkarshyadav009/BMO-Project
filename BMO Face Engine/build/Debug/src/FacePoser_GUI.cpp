// FacePoser_GUI.cpp
// - A Visual Editor for the BMO Rig (Mouth + Eyes)
// - Uses RayGui for Sliders/Buttons

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

// Ensure these filenames match your src folder!
#include "ShaderParametricMouth.cpp" 
#include "ShaderParametricEyes.cpp" 

using json = nlohmann::json;

// ---------------------------------------------------------
// 1. ATLAS LOADER
// ---------------------------------------------------------
struct ReferenceAtlas {
    Texture2D texture;
    std::unordered_map<std::string, Rectangle> frames;
    std::vector<std::string> mouthNames; 

    void Load(const char* img, const char* data) {
        texture = LoadTexture(img);
        std::ifstream f(data);
        if (!f.good()) return; 
        json j = json::parse(f);

        if (j.contains("textures")) {
            for (auto& t : j["textures"]) {
                for (auto& fr : t["frames"]) {
                    std::string name = fr["filename"];
                    bool isMouth = (name.find("_mouth") != std::string::npos) || 
                                   (name.find("mouth_phoneme") != std::string::npos);
                    if (isMouth) {
                        frames[name] = { (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                                         (float)fr["frame"]["w"], (float)fr["frame"]["h"] };
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
void AppendToDatabase(std::string name, FacialParams mp, EyeParams ep) {
    std::ofstream file("visemes_database.txt", std::ios::app);
    if (file.is_open()) {
        file << "visemes[\"" << name << "\"] = {\n";
        file << "  mouth = { " << mp.open << "f, " << mp.width << "f, " << mp.curve << "f, "
             << mp.squeezeTop << "f, " << mp.squeezeBottom << "f, " << mp.teethY << "f, "
             << mp.tongueUp << "f, " << mp.tongueX << "f, " << mp.tongueWidth << "f, "
             << mp.asymmetry << "f, " << mp.squareness << "f, "
             << mp.teethWidth << "f, " << mp.teethGap << "f, "
             << mp.scale << "f, " << mp.outlineThickness << "f },\n";
        file << "  eyes = { " << ep.shapeID << "f, " << ep.bend << "f, " 
             << ep.thickness << "f, " << ep.pupilSize << "f, " 
             << ep.targetLookX << "f, " << ep.targetLookY << "f }\n";
        file << "};\n";
        file.close();
        std::cout << "Saved: " << name << std::endl;
    }
}

// ---------------------------------------------------------
// MAIN EDITOR
// ---------------------------------------------------------
int main() {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 1000, "BMO Face Poser (Mouth + Eyes)");
    SetTargetFPS(60);

    // Styles
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

    // Init Rigs
    ParametricMouth mouthRig;
    mouthRig.Init({ 800, 450 }); 
    ParametricEyes eyeRig;
    eyeRig.Init(); 

    ReferenceAtlas atlas;
    atlas.Load("assets/BMO_Animation_LipSyncSprite.png", "assets/BMO_Animation_Lipsync.json");

    int currentIdx = 0;
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false; 

    // Eye State
    EyeParams currentEye; 

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        float startX = 20.0f; float startY = 20.0f; float w = 300.0f;
        const float rowH = 20.0f; const float rowGap = 10.0f; const float btnH = 30.0f;

        BeginDrawing();
        ClearBackground({131, 220, 169, 255}); 

        // 1. CONTROLS
        GuiGroupBox({startX, startY, w, 120}, "SPRITE BROWSER");
        if (GuiButton({startX + 10, startY + 30, 40, 30}, "<")) {
            currentIdx--; if (currentIdx < 0) currentIdx = (int)atlas.mouthNames.size() - 1;
        }
        if (GuiButton({startX + 250, startY + 30, 40, 30}, ">")) {
            currentIdx = (int)((currentIdx + 1) % (int)atlas.mouthNames.size());
        }
        std::string nameLabel = atlas.mouthNames.empty() ? "NONE" : atlas.mouthNames[currentIdx];
        GuiLabel({startX + 60, startY + 30, 180, 30}, nameLabel.c_str());
        if (GuiButton({startX + 10, startY + 70, w - 20, 30}, "SAVE PRESET (Enter)")) {
            if (!atlas.mouthNames.empty()) AppendToDatabase(atlas.mouthNames[currentIdx], mouthRig.target, currentEye);
        }

        // 2. SLIDERS
        startY += 140.0f; float sy = startY + 25.0f; float innerX = startX + 10; float innerW = w - 20;
        float labelW = 105.0f; float valueW = 60.0f; float colGap = 8.0f;

        #define DRAW_ROW_SLIDER(label, value, mn, mx) \
            GuiLabel({ innerX, sy, labelW, rowH }, label); \
            GuiSliderBar({ innerX + labelW + colGap, sy, innerW - labelW - valueW - colGap * 2.0f, rowH }, NULL, NULL, &(value), (mn), (mx)); \
            GuiLabel({ innerX + innerW - valueW, sy, valueW, rowH }, TextFormat("%.2f", (value))); \
            sy += rowH + rowGap;

        GuiLabel({innerX, sy, innerW, rowH}, "--- MOUTH ---"); sy += 25;
        DRAW_ROW_SLIDER("Scale",      mouthRig.target.scale,         0.5f,  4.0f);
        DRAW_ROW_SLIDER("Open",       mouthRig.target.open,          0.0f,  1.2f);
        DRAW_ROW_SLIDER("Width",      mouthRig.target.width,         0.1f,  1.5f);
        DRAW_ROW_SLIDER("Curve",      mouthRig.target.curve,        -1.0f,  1.0f);
        DRAW_ROW_SLIDER("Sqze Top",   mouthRig.target.squeezeTop,    0.0f,  1.0f);
        DRAW_ROW_SLIDER("Sqze Bot",   mouthRig.target.squeezeBottom, 0.0f,  1.0f);
        DRAW_ROW_SLIDER("Asymmetry",  mouthRig.target.asymmetry,    -1.0f,  1.0f);
        DRAW_ROW_SLIDER("Outline",    mouthRig.target.outlineThickness, 1.0f, 10.0f);
        sy += 10;
        GuiLabel({innerX, sy, innerW, rowH}, "--- EYES ---"); sy += 25;
        DRAW_ROW_SLIDER("Shape ID",   currentEye.shapeID,      0.0f,  5.0f);
        DRAW_ROW_SLIDER("Bend",       currentEye.bend,        -1.0f,  1.0f);
        DRAW_ROW_SLIDER("Thick",      currentEye.thickness,    1.0f, 10.0f);
        DRAW_ROW_SLIDER("Pupil",      currentEye.pupilSize,    0.0f,  1.0f);
        DRAW_ROW_SLIDER("Look X",     currentEye.targetLookX, -50.0f, 50.0f);
        DRAW_ROW_SLIDER("Look Y",     currentEye.targetLookY, -50.0f, 50.0f);

        if (GuiButton({ innerX, sy, innerW, btnH }, "RESET")) {
            mouthRig.target = MakeDefaultParams();
            currentEye = {0.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f};
        }
        sy += btnH + rowGap;
        
        Rectangle cb2 = { startX + 10, sy + 5, 20, 20 };
        GuiCheckBox(cb2, NULL, &usePhysics);
        GuiLabel({ cb2.x + 28, cb2.y, 160, 20 }, "Use Physics / Blink");
        float rigBoxH = (sy - startY) + 40.0f;
        GuiGroupBox({ startX, startY, w, rigBoxH }, "RIG PARAMETERS");

        // --- RENDER ---
        if (showReference && !atlas.mouthNames.empty()) {
            atlas.Draw(atlas.mouthNames[currentIdx], mouthRig.centerPos, 1.0f, refOpacity);
        }

        // DRAW EYES
        EyeParams simEye = currentEye; 
        if (usePhysics) { eyeRig.Update(dt, simEye); }
        else { 
             eyeRig.sLookX.val = simEye.targetLookX; eyeRig.sLookY.val = simEye.targetLookY; 
             eyeRig.sScaleX.val = 1.0f; eyeRig.sScaleY.val = 1.0f; 
        }

        Vector2 mouthCenter = mouthRig.centerPos;
        
        // TUNING PARAMETERS
        float eyeSpacing = 200.0f; 
        float eyeHeight  = 150.0f;  
        float eyeSize    = 50.0f; // [FIX] Smaller Eyes (Was 120.0f)

        Rectangle leftRect = { 
            mouthCenter.x - (eyeSpacing/2) - (eyeSize/2), 
            mouthCenter.y - eyeHeight - (eyeSize/2), 
            eyeSize, eyeSize 
        };

        Rectangle rightRect = { 
            mouthCenter.x + (eyeSpacing/2) - (eyeSize/2), 
            mouthCenter.y - eyeHeight - (eyeSize/2), 
            eyeSize, eyeSize 
        };

        eyeRig.DrawEye(leftRect, simEye, BLACK);
        eyeRig.DrawEye(rightRect, simEye, BLACK);

        // DRAW MOUTH
        mouthRig.usePhysics = usePhysics;
        mouthRig.UpdatePhysics(dt);
        mouthRig.GenerateGeometry();
        mouthRig.Draw();

        if (IsKeyPressed(KEY_LEFT)) currentIdx--;
        if (IsKeyPressed(KEY_RIGHT)) currentIdx++;
        if (currentIdx < 0) currentIdx = (int)atlas.mouthNames.size() - 1;
        if (currentIdx >= (int)atlas.mouthNames.size()) currentIdx = 0;
        if (IsKeyPressed(KEY_ENTER) && !atlas.mouthNames.empty()) AppendToDatabase(atlas.mouthNames[currentIdx], mouthRig.target, currentEye);

        EndDrawing();
    }
    mouthRig.Unload();
    CloseWindow();
    return 0;
}