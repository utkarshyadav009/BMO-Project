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
    Rectangle originalRect; // The full face rectangle in the original atlas
    
    bool isVertical = false; // Toggle for D: faces where eyes are vertical

    // STANDARD MODE (Top/Bottom Split)
    int splitLine;     // Y position relative to the face top
    
    // MOUTH BOX (Manual Crop Region)
    int mouthX; // Offset X from face left
    int mouthY; // Offset Y from face top
    int mouthW; // Width of the cut
    int mouthH; // Height of the cut

    // VERTICAL MODE EYES (Manual Override)
    int eyesX;
    int eyesY;
    int eyesW;
    int eyesH;
};

struct AdvancedSlicer {
    Texture2D texture{};
    std::vector<FrameData> frames;
    json originalJsonData; 

    // Load the ORIGINAL raw Atlas (BMO_SpriteSheet_Data.json)
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
                    
                    // --- INTELLIGENT DEFAULTS ---
                    // 1. Default Split Line (55% down)
                    fd.splitLine = (int)(fd.originalRect.height * 0.55f);
                    
                    // 2. Default Mouth Box (Centered, Bottom Half)
                    fd.mouthW = (int)(fd.originalRect.width * 0.5f); 
                    fd.mouthH = (int)(fd.originalRect.height - fd.splitLine);
                    fd.mouthX = (int)((fd.originalRect.width - fd.mouthW) / 2.0f);
                    fd.mouthY = fd.splitLine;

                    // 3. Default Eyes Box (Full Width, Top Half)
                    fd.eyesX = 0;
                    fd.eyesY = 0;
                    fd.eyesW = (int)fd.originalRect.width;
                    fd.eyesH = fd.splitLine;

                    frames.push_back(fd);
                }
            }
        }
        
        // Sort alphabetically to find faces easily
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
            
            // Logic: Vertical Mode (Manual Box) vs Standard Mode (Split Line)
            int eX = fd.isVertical ? fd.eyesX : 0;
            int eY = fd.isVertical ? fd.eyesY : 0;
            int eW = fd.isVertical ? fd.eyesW : (int)fd.originalRect.width;
            int eH = fd.isVertical ? fd.eyesH : fd.splitLine;

            // ATLAS FRAME (Where to cut)
            eyes["frame"] = {
                {"x", fd.originalRect.x + eX},
                {"y", fd.originalRect.y + eY},
                {"w", eW},
                {"h", eH}
            };
            
            // SPRITE SOURCE (Offsets & Dimensions)
            // FIX APPLIED: "w" and "h" are now the CUT size, not the ORIGINAL size
            eyes["spriteSourceSize"] = {
                {"x", eX},
                {"y", eY},
                {"w", eW},
                {"h", eH}
            };
            // Source Size remains the full original canvas size
            eyes["sourceSize"] = {{"w",fd.originalRect.width},{"h",fd.originalRect.height}};
            eyes["pivot"] = {{"x",0.5},{"y",0.5}};
            
            newFramesJson.push_back(eyes);

            // --- 2. MOUTH ENTRY ---
            // Only create mouth if it's not a blink frame (optional logic, but safe to keep)
            if (fd.name.find("closed_eyes") == std::string::npos) { 
                json mouth;
                mouth["filename"] = fd.name + "_mouth";
                mouth["rotated"] = false;
                mouth["trimmed"] = false;
                
                // ATLAS FRAME
                mouth["frame"] = {
                    {"x", fd.originalRect.x + fd.mouthX},
                    {"y", fd.originalRect.y + fd.mouthY}, 
                    {"w", fd.mouthW},
                    {"h", fd.mouthH}
                };
                
                // SPRITE SOURCE
                // FIX APPLIED: "w" and "h" are the CUT size
                mouth["spriteSourceSize"] = {
                    {"x", fd.mouthX}, 
                    {"y", fd.mouthY}, 
                    {"w", fd.mouthW}, 
                    {"h", fd.mouthH}
                };
                mouth["sourceSize"] = {{"w",fd.originalRect.width},{"h",fd.originalRect.height}};
                mouth["pivot"] = {{"x",0.5},{"y",0.5}};
                
                newFramesJson.push_back(mouth);
            }
        }

        // Write Final JSON
        output["textures"][0]["frames"] = newFramesJson;
        std::ofstream o(outputPath);
        o << std::setw(4) << output << std::endl;
        TraceLog(LOG_INFO, "SAVED FIXED JSON TO: %s", outputPath);
    }

    void Unload() { UnloadTexture(texture); }
};

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    // Large window for comfortable editing
    InitWindow(1600, 900, "BMO Editor & QA Tool (Final Fix)");
    SetTargetFPS(60);

    AdvancedSlicer slicer;
    // LOAD THE RAW DATA (The starting point)
    slicer.Load("assets/BMO_SpriteSheet_Texture.png", "assets/BMO_SpriteSheet_Data.json");

    int idx = 0;
    float zoom = 0.8f;
    bool saved = false;

    // UI Colors
    Color bgCol = {30, 30, 30, 255};
    Color guideCol = GREEN;
    Color cropCol = YELLOW;
    Color eyeCol = SKYBLUE; 

    while (!WindowShouldClose()) {
        // --- INPUT ---
        
        // 1. NAVIGATION
        if (IsKeyPressed(KEY_PERIOD)) { idx++; if(idx >= slicer.frames.size()) idx = 0; saved = false; }
        if (IsKeyPressed(KEY_COMMA))  { idx--; if(idx < 0) idx = slicer.frames.size() - 1; saved = false; }

        FrameData& current = slicer.frames[idx];
        int speed = IsKeyDown(KEY_LEFT_SHIFT) ? 10 : 1; // Shift for fast movement

        // 2. TOGGLE MODES
        if (IsKeyPressed(KEY_V)) current.isVertical = !current.isVertical;

        if (!current.isVertical) {
            // STANDARD MODE (Split Line)
            if (IsKeyDown(KEY_W)) current.splitLine -= speed;
            if (IsKeyDown(KEY_S)) current.splitLine += speed;
            
            // 'R' to reset mouth Y to the split line
            if (IsKeyPressed(KEY_R)) current.mouthY = current.splitLine; 
        } else {
            // VERTICAL MODE (Manual Eyes Box)
            // WASD Moves Eyes
            if (IsKeyDown(KEY_W)) current.eyesY -= speed;
            if (IsKeyDown(KEY_S)) current.eyesY += speed;
            if (IsKeyDown(KEY_A)) current.eyesX -= speed;
            if (IsKeyDown(KEY_D)) current.eyesX += speed;
            
            // IJKL Resizes Eyes
            if (IsKeyDown(KEY_I)) current.eyesH -= speed;
            if (IsKeyDown(KEY_K)) current.eyesH += speed;
            if (IsKeyDown(KEY_J)) current.eyesW -= speed;
            if (IsKeyDown(KEY_L)) current.eyesW += speed;
        }

        // 3. MOUTH EDITING (Always Active)
        // Arrows: Move
        if (IsKeyDown(KEY_UP))    current.mouthY -= speed;
        if (IsKeyDown(KEY_DOWN))  current.mouthY += speed;
        if (IsKeyDown(KEY_LEFT))  current.mouthX -= speed;
        if (IsKeyDown(KEY_RIGHT)) current.mouthX += speed;

        // Q/E (Width) & Z/C (Height)
        if (IsKeyDown(KEY_Q)) current.mouthW -= speed;
        if (IsKeyDown(KEY_E)) current.mouthW += speed;
        if (IsKeyDown(KEY_Z)) current.mouthH -= speed;
        if (IsKeyDown(KEY_C)) current.mouthH += speed;

        // 4. SAVE
        if (IsKeyPressed(KEY_ENTER)) {
            slicer.Save("assets/BMO_Final_Animation.json");
            saved = true;
        }

        // 5. EXPLODE TOGGLE (QA Feature)
        bool explode = IsKeyDown(KEY_SPACE);

        // --- RENDER ---
        BeginDrawing();
        ClearBackground(bgCol);

        if (slicer.frames.empty()) {
            DrawText("No frames found. Check JSON path!", 20, 20, 20, RED);
            EndDrawing(); continue;
        }

        // --- LEFT SIDE: EDITOR VIEW ---
        float startX = 50;
        float startY = 100;
        Rectangle srcFull = current.originalRect;
        Rectangle dstFull = { startX, startY, srcFull.width * zoom, srcFull.height * zoom };
        
        // Draw Original Face
        DrawTexturePro(slicer.texture, srcFull, dstFull, {0,0}, 0.0f, WHITE);
        DrawRectangleLinesEx(dstFull, 2, GRAY);
        
        // Editor Overlays
        if (current.isVertical) {
            DrawText("MODE: VERTICAL (Manual Eyes)", startX, startY - 40, 20, eyeCol);
            // Draw Blue Eye Box
            Rectangle eyeBox = {
                startX + (current.eyesX * zoom),
                startY + (current.eyesY * zoom),
                current.eyesW * zoom,
                current.eyesH * zoom
            };
            DrawRectangleLinesEx(eyeBox, 2, eyeCol);
        } else {
            DrawText("MODE: STANDARD (Split Line)", startX, startY - 40, 20, guideCol);
            // Draw Green Split Line
            float splitY = startY + (current.splitLine * zoom);
            DrawLine(startX, splitY, startX + dstFull.width, splitY, guideCol);
        }

        // Draw Yellow Mouth Box
        Rectangle mouthBox = {
            startX + (current.mouthX * zoom),
            startY + (current.mouthY * zoom),
            current.mouthW * zoom,
            current.mouthH * zoom
        };
        DrawRectangleLinesEx(mouthBox, 2, cropCol);


        // --- RIGHT SIDE: QA PREVIEW ---
        float previewX = startX + dstFull.width + 150;
        float explodeGap = explode ? 50.0f : 0.0f; // Gap for exploded view

        DrawText("QA PREVIEW (Hold SPACE to Explode)", previewX, startY - 40, 20, LIGHTGRAY);

        // 1. Render Eyes (Calculated based on current edit)
        Rectangle pSrcEyes;
        if (!current.isVertical) {
            pSrcEyes = srcFull; pSrcEyes.height = current.splitLine;
        } else {
            pSrcEyes = { srcFull.x + current.eyesX, srcFull.y + current.eyesY, (float)current.eyesW, (float)current.eyesH };
        }
        
        // Eye Position
        Vector2 eyePos = { 
            previewX + (current.isVertical ? current.eyesX * zoom : 0), 
            startY + (current.isVertical ? current.eyesY * zoom : 0) - explodeGap
        };
        Rectangle pDstEyes = { eyePos.x, eyePos.y, pSrcEyes.width * zoom, pSrcEyes.height * zoom };
        
        DrawTexturePro(slicer.texture, pSrcEyes, pDstEyes, {0,0}, 0.0f, WHITE);
        if (explode) DrawRectangleLinesEx(pDstEyes, 2, eyeCol); // Debug outline

        // 2. Render Mouth
        Rectangle pSrcMouth = { 
            srcFull.x + current.mouthX, 
            srcFull.y + current.mouthY, 
            (float)current.mouthW, 
            (float)current.mouthH 
        };
        
        // Mouth Position (Relative to Preview Origin)
        Vector2 mouthPos = {
            previewX + (current.mouthX * zoom),
            startY + (current.mouthY * zoom) + explodeGap
        };
        Rectangle pDstMouth = { mouthPos.x, mouthPos.y, pSrcMouth.width * zoom, pSrcMouth.height * zoom };

        DrawTexturePro(slicer.texture, pSrcMouth, pDstMouth, {0,0}, 0.0f, WHITE);
        if (explode) DrawRectangleLinesEx(pDstMouth, 2, cropCol); // Debug outline


        // --- UI INSTRUCTIONS ---
        int uiY = 800;
        DrawText("NAVIGATE: < , > (Comma/Period)", 50, uiY, 20, WHITE);
        
        DrawText("MOUTH (Arrows, Q/E/Z/C):", 50, uiY+30, 20, cropCol);
        DrawText(TextFormat("X:%d Y:%d W:%d H:%d", current.mouthX, current.mouthY, current.mouthW, current.mouthH), 350, uiY+30, 20, cropCol);

        if (current.isVertical) DrawText("EYES (WASD, IJKL): Manual Box", 50, uiY+60, 20, eyeCol);
        else DrawText("EYES (W/S): Split Line", 50, uiY+60, 20, guideCol);

        DrawText("ENTER: SAVE BMO_Final_Animation.json", 1000, uiY, 30, YELLOW);
        if(saved) DrawText("SAVED!", 1000, uiY+40, 30, GREEN);

        EndDrawing();
    }

    slicer.Unload();
    CloseWindow();
    return 0;
}