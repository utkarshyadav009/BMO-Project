#include "raylib.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <cmath>

using json = nlohmann::json;

// ---------------------------------------------------------
// 1. ASSET LOADER
// ---------------------------------------------------------
struct SpriteAtlas {
    Texture2D texture{};
    std::unordered_map<std::string, Rectangle> frames;

    void Load(const char* imagePath, const char* jsonPath) {
        texture = LoadTexture(imagePath);
        std::ifstream f(jsonPath);
        if (!f.is_open()) {
            TraceLog(LOG_ERROR, "JSON not found: %s", jsonPath);
            return;
        }
        json data = json::parse(f);

        if (data.contains("textures")) {
            for (auto& t : data["textures"]) {
                for (auto& fr : t["frames"]) {
                    std::string name = fr["filename"];
                    frames[name] = {
                        (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                        (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                    };
                }
            }
        }
        TraceLog(LOG_INFO, "Atlas Loaded: %d frames", (int)frames.size());
    }

    void DrawFrame(const std::string& name, int x, int y, int screenW, int screenH, float scaleX, float scaleY) {
        if (frames.find(name) == frames.end()) return; // Skip missing frames

        Rectangle src = frames[name];
        
        // Scale logic for "Juice" (Squash & Stretch)
        // We draw from the CENTER of the screen to apply scale correctly
        float destW = screenW * scaleX;
        float destH = screenH * scaleY;
        
        Rectangle dest = {
            (float)x + (screenW - destW) / 2.0f, // Center X
            (float)y + (screenH - destH) / 2.0f, // Center Y
            destW, 
            destH
        };
        
        DrawTexturePro(texture, src, dest, {0,0}, 0.0f, WHITE);
    }

    void Unload() { UnloadTexture(texture); }
};

// ---------------------------------------------------------
// 2. PHYSICS (Juice)
// ---------------------------------------------------------
struct Spring { float val = 1.0f, vel = 0.0f, target = 1.0f; };

void UpdateSpring(Spring& s, float stiffness, float damping, float dt) {
    float force = (s.target - s.val) * stiffness;
    s.vel = (s.vel + force * dt) * damping;
    s.val += s.vel * dt;
}

// ---------------------------------------------------------
// 3. BMO PUPPET
// ---------------------------------------------------------
struct BMO {
    SpriteAtlas atlas;
    
    // State
    std::string currentMood = "happy_standard"; // e.g. "happy_standard"
    std::string currentMouth = "happy_standard"; 
    
    // Physics
    Spring scaleY;
    Spring scaleX;

    void Init() {
        // LOAD YOUR NEW FILES HERE
        atlas.Load("assets/BMO_AnimationSpriteSheet.png", 
                   "assets/BMO_AnimationSpriteSheetData.json");
    }

    void SetMood(std::string mood) {
        if (currentMood != mood) {
            currentMood = mood;
            // Juice: Small pop when changing mood
            scaleY.vel += 0.05f;
            scaleX.vel -= 0.02f;
        }
    }

    void Update(float dt, bool isTalking) {
        // 1. Breathing Idle
        scaleY.target = 1.0f + sinf(GetTime() * 2.0f) * 0.005f;
        scaleX.target = 1.0f;

        // 2. Talking Squash & Stretch
        if (isTalking) {
            scaleY.target += 0.02f; // Stretch vertically
            scaleX.target -= 0.01f; // Squash horizontally
        }

        // 3. Run Physics
        UpdateSpring(scaleY, 150.0f, 0.6f, dt);
        UpdateSpring(scaleX, 150.0f, 0.6f, dt);
    }

    void Draw(int screenW, int screenH) {
        // Construct filenames from mood
        // Expects: "face_[mood]_eyes" and "face_[mouth_shape]_mouth"
        
        std::string eyesTex = "face_" + currentMood + "_eyes";
        std::string mouthTex = "face_" + currentMouth + "_mouth";

        // Draw Layer 1: EYES (Base)
        atlas.DrawFrame(eyesTex, 0, 0, screenW, screenH, scaleX.val, scaleY.val);

        // Draw Layer 2: MOUTH (Top)
        atlas.DrawFrame(mouthTex, 0, 0, screenW, screenH, scaleX.val, scaleY.val);
    }
};

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    const int W = 1280;
    const int H = 720;
    InitWindow(W, H, "BMO Final Engine");
    SetTargetFPS(60);

    BMO bmo;
    bmo.Init();

    bool isTalking = false;
    float talkTimer = 0.0f;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        // --- CONTROLS ---
        // 1. Mood Switching (Eyes)
        if (IsKeyPressed(KEY_H)) bmo.SetMood("happy_standard");
        if (IsKeyPressed(KEY_S)) bmo.SetMood("sad_standard");
        if (IsKeyPressed(KEY_A)) bmo.SetMood("angry_growl");
        if (IsKeyPressed(KEY_W)) bmo.SetMood("worried_teary");

        // 2. Talk Toggle
        if (IsKeyPressed(KEY_SPACE)) isTalking = !isTalking;

        // --- MOUTH LOGIC (Placeholder for Rhubarb) ---
        // For now, we randomly flap between Closed and Open to test alignment
        if (isTalking) {
            talkTimer += dt;
            if (talkTimer > 0.15f) {
                talkTimer = 0.0f;
                // Toggle between the "Closed" version of the current mood 
                // and a generic "Talking" mouth
                if (bmo.currentMouth == bmo.currentMood) {
                    bmo.currentMouth = "happy_talking"; // Open
                } else {
                    bmo.currentMouth = bmo.currentMood; // Closed
                }
            }
        } else {
            // When silent, mouth matches mood
            bmo.currentMouth = bmo.currentMood; 
        }

        // --- UPDATE & DRAW ---
        bmo.Update(dt, isTalking);

        BeginDrawing();
        ClearBackground({131, 220, 169, 255}); // BMO Green

        bmo.Draw(W, H);

        DrawText("H: Happy | S: Sad | A: Angry | W: Worried", 20, 20, 20, DARKGRAY);
        DrawText(isTalking ? "TALKING..." : "SILENT (Space)", 20, 50, 20, isTalking ? RED : DARKGRAY);

        EndDrawing();
    }

    bmo.atlas.Unload();
    CloseWindow();
    return 0;
}