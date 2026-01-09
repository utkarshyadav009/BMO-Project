#include "raylib.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>

using json = nlohmann::json;

// ---------------------------------------------------------
// DATA STRUCTURES
// ---------------------------------------------------------
struct FrameData {
    std::string name;
    Rectangle originalRect; 
    
    // TWO INDEPENDENT BOXES (Relative to the face top-left)
    Rectangle eyesBox;  // X, Y, W, H
    Rectangle mouthBox; // X, Y, W, H
};

struct FreeformSlicer {
    Texture2D texture{};
    std::vector<FrameData> frames;
    json originalJsonData; 

    void Load(const char* imagePath, const char* originalJsonPath) {
        texture = LoadTexture(imagePath);
        std::ifstream f(originalJsonPath);
        if (!f.is_open()) {
            TraceLog(LOG_ERROR, "JSON NOT FOUND: %s", originalJsonPath);
            return;
        }
        originalJsonData = json::parse(f);

        if (originalJsonData.contains("textures")) {
            for (auto& t : originalJsonData["textures"]) {
                for (auto& fr : t["frames"]) {
                    FrameData fd;
                    fd.name = fr["filename"];
                    fd.originalRect = {
                        (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                        (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                    };
                    
                    // --- DEFAULTS ---
                    float w = fd.originalRect.width;
                    float h = fd.originalRect.height;

                    // Default Eyes: Top 60% of the face
                    fd.eyesBox = { 0, 0, w, h * 0.6f };

                    // Default Mouth: Bottom 40%, centered, slightly narrower
                    float mouthW = w * 0.5f;
                    fd.mouthBox = { (w - mouthW)/2.0f, h * 0.6f, mouthW, h * 0.4f };

                    frames.push_back(fd);
                }
            }
        }
        
        std::sort(frames.begin(), frames.end(), [](const FrameData& a, const FrameData& b) {
            return a.name < b.name;
        });
    }

    void Save(const char* outputPath) {
        json output = originalJsonData;
        std::vector<json> newFramesJson;

        for (const auto& fd : frames) {
            // --- 1. EYES ENTRY ---
            json eyes;
            eyes["filename"] = fd.name + "_eyes";
            eyes["rotated"] = false;
            eyes["trimmed"] = false;
            eyes["frame"] = {
                {"x", fd.originalRect.x + fd.eyesBox.x},
                {"y", fd.originalRect.y + fd.eyesBox.y},
                {"w", fd.eyesBox.width},
                {"h", fd.eyesBox.height}
            };
            eyes["spriteSourceSize"] = {
                {"x", fd.eyesBox.x},
                {"y", fd.eyesBox.y},
                {"w", fd.eyesBox.width},
                {"h", fd.eyesBox.height}
            };
            eyes["sourceSize"] = {{"w",fd.originalRect.width},{"h",fd.originalRect.height}};
            eyes["pivot"] = {{"x",0.5},{"y",0.5}};
            newFramesJson.push_back(eyes);

            // --- 2. MOUTH ENTRY ---
            if (fd.name.find("closed_eyes") == std::string::npos) {
                json mouth;
                mouth["filename"] = fd.name + "_mouth";
                mouth["rotated"] = false;
                mouth["trimmed"] = false;
                mouth["frame"] = {
                    {"x", fd.originalRect.x + fd.mouthBox.x},
                    {"y", fd.originalRect.y + fd.mouthBox.y}, 
                    {"w", fd.mouthBox.width},
                    {"h", fd.mouthBox.height}
                };
                mouth["spriteSourceSize"] = {
                    {"x", fd.mouthBox.x}, 
                    {"y", fd.mouthBox.y}, 
                    {"w", fd.mouthBox.width}, 
                    {"h", fd.mouthBox.height}
                };
                mouth["sourceSize"] = {{"w",fd.originalRect.width},{"h",fd.originalRect.height}};
                mouth["pivot"] = {{"x",0.5},{"y",0.5}};
                newFramesJson.push_back(mouth);
            }
        }

        output["textures"][0]["frames"] = newFramesJson;
        std::ofstream o(outputPath);
        o << std::setw(4) << output << std::endl;
        TraceLog(LOG_INFO, "SAVED TO: %s", outputPath);
    }

    void Unload() { UnloadTexture(texture); }
};

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    InitWindow(1600, 900, "BMO Freeform Slicer");
    SetTargetFPS(60);

    FreeformSlicer slicer;
    // Load your CLEAN (Original) JSON and Texture
    slicer.Load("assets/BMO_SpriteSheet_Texture.png", "assets/BMO_SpriteSheet_Data.json");

    int idx = 0;
    float zoom = 0.8f;
    bool saved = false;
    
    // EDITOR STATE
    bool editingEyes = false; // False = Editing Mouth (Yellow), True = Editing Eyes (Blue)

    // Colors
    Color bgCol = {30, 30, 30, 255};
    Color eyeCol = SKYBLUE; 
    Color mouthCol = YELLOW;

    while (!WindowShouldClose()) {
        // --- INPUT ---
        
        // 1. NAVIGATION
        if (IsKeyPressed(KEY_PERIOD)) { idx++; if(idx >= slicer.frames.size()) idx = 0; saved = false; }
        if (IsKeyPressed(KEY_COMMA))  { idx--; if(idx < 0) idx = slicer.frames.size() - 1; saved = false; }

        FrameData& current = slicer.frames[idx];
        int speed = IsKeyDown(KEY_LEFT_SHIFT) ? 10 : 1;

        // 2. TOGGLE EDIT MODE
        if (IsKeyPressed(KEY_TAB)) editingEyes = !editingEyes;

        // 3. EDIT BOX (WASD for Position, Arrows for Size)
        Rectangle* activeBox = editingEyes ? &current.eyesBox : &current.mouthBox;

        // Move (WASD)
        if (IsKeyDown(KEY_W)) activeBox->y -= speed;
        if (IsKeyDown(KEY_S)) activeBox->y += speed;
        if (IsKeyDown(KEY_A)) activeBox->x -= speed;
        if (IsKeyDown(KEY_D)) activeBox->x += speed;

        // Resize (Arrows)
        if (IsKeyDown(KEY_RIGHT)) activeBox->width += speed;
        if (IsKeyDown(KEY_LEFT))  activeBox->width -= speed;
        if (IsKeyDown(KEY_DOWN))  activeBox->height += speed;
        if (IsKeyDown(KEY_UP))    activeBox->height -= speed;

        // 4. SAVE
        if (IsKeyPressed(KEY_ENTER)) {
            slicer.Save("assets/BMO_Final_Animation.json");
            saved = true;
        }

        // --- RENDER ---
        BeginDrawing();
        ClearBackground(bgCol);

        if (slicer.frames.empty()) { EndDrawing(); continue; }

        // --- LEFT SIDE: EDITOR VIEW ---
        float startX = 50;
        float startY = 100;
        Rectangle srcFull = current.originalRect;
        Rectangle dstFull = { startX, startY, srcFull.width * zoom, srcFull.height * zoom };
        
        DrawTexturePro(slicer.texture, srcFull, dstFull, {0,0}, 0.0f, WHITE);
        DrawRectangleLinesEx(dstFull, 2, GRAY);
        
        // Draw Eyes Box
        Rectangle eyeRectScreen = {
            startX + current.eyesBox.x * zoom,
            startY + current.eyesBox.y * zoom,
            current.eyesBox.width * zoom,
            current.eyesBox.height * zoom
        };
        DrawRectangleLinesEx(eyeRectScreen, 2, editingEyes ? WHITE : Fade(eyeCol, 0.5f));
        if (editingEyes) DrawText("EDITING EYES", eyeRectScreen.x, eyeRectScreen.y - 20, 20, eyeCol);

        // Draw Mouth Box
        Rectangle mouthRectScreen = {
            startX + current.mouthBox.x * zoom,
            startY + current.mouthBox.y * zoom,
            current.mouthBox.width * zoom,
            current.mouthBox.height * zoom
        };
        DrawRectangleLinesEx(mouthRectScreen, 2, !editingEyes ? WHITE : Fade(mouthCol, 0.5f));
        if (!editingEyes) DrawText("EDITING MOUTH", mouthRectScreen.x, mouthRectScreen.y + mouthRectScreen.height + 5, 20, mouthCol);


        // --- RIGHT SIDE: PREVIEW ---
        float previewX = startX + dstFull.width + 150;
        float explodeGap = IsKeyDown(KEY_SPACE) ? 50.0f : 0.0f;

        DrawText("QA PREVIEW (Space to Explode)", previewX, startY - 40, 20, LIGHTGRAY);

        // Render Eyes Layer
        Rectangle srcE = { 
            srcFull.x + current.eyesBox.x, srcFull.y + current.eyesBox.y, 
            current.eyesBox.width, current.eyesBox.height 
        };
        Vector2 posE = { previewX + current.eyesBox.x * zoom, startY + current.eyesBox.y * zoom - explodeGap };
        Rectangle dstE = { posE.x, posE.y, srcE.width * zoom, srcE.height * zoom };
        DrawTexturePro(slicer.texture, srcE, dstE, {0,0}, 0.0f, WHITE);

        // Render Mouth Layer
        Rectangle srcM = { 
            srcFull.x + current.mouthBox.x, srcFull.y + current.mouthBox.y, 
            current.mouthBox.width, current.mouthBox.height 
        };
        Vector2 posM = { previewX + current.mouthBox.x * zoom, startY + current.mouthBox.y * zoom + explodeGap };
        Rectangle dstM = { posM.x, posM.y, srcM.width * zoom, srcM.height * zoom };
        DrawTexturePro(slicer.texture, srcM, dstM, {0,0}, 0.0f, WHITE);

        // --- INSTRUCTIONS ---
        int uiY = 800;
        DrawText(TextFormat("Face: %s", current.name.c_str()), 50, 50, 30, WHITE);
        
        DrawText("TAB: Switch Box (Eyes vs Mouth)", 50, uiY, 20, YELLOW);
        DrawText("WASD: Move Box | ARROWS: Resize Box", 50, uiY+30, 20, WHITE);
        DrawText("ENTER: Save JSON", 500, uiY, 20, GREEN);
        
        if (saved) DrawText("SAVED!", 700, uiY, 20, GREEN);

        EndDrawing();
    }

    slicer.Unload();
    CloseWindow();
    return 0;
}