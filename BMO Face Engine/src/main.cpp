#include "raylib.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
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

    void DrawFrame(const std::string& name, int x, int y, int screenW, int screenH, Vector2 offset, Vector2 scale, Color tint) {
        if (frames.find(name) == frames.end()) return;

        Rectangle src = frames[name];
        
        float destW = screenW * scale.x;
        float destH = screenH * scale.y;
        
        Rectangle dest = {
            (float)x + (screenW - destW) / 2.0f + offset.x, 
            (float)y + (screenH - destH) / 2.0f + offset.y, 
            destW, 
            destH
        };
        
        DrawTexturePro(texture, src, dest, {0,0}, 0.0f, tint);
    }

    void Unload() { UnloadTexture(texture); }
};

// ---------------------------------------------------------
// 2. LIP SYNC PARSER
// ---------------------------------------------------------
struct LipSyncSystem {
    struct Cue { float time; char shape; };
    std::vector<Cue> cues;

    void Load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) { TraceLog(LOG_ERROR, "TSV Missing: %s", path.c_str()); return; }
        
        float t; std::string s;
        cues.clear();
        while (f >> t >> s) {
            cues.push_back({t, s[0]}); 
        }
    }

    std::string GetMouthForTime(float time) {
        if (cues.empty()) return "mouth_phoneme_X"; 

        char shape = 'X';
        for (const auto& cue : cues) {
            if (cue.time > time) break;
            shape = cue.shape;
        }

        switch(shape) {
            case 'A': return "mouth_phoneme_A"; 
            case 'B': return "mouth_phoneme_B"; 
            case 'C': return "mouth_phoneme_C"; 
            case 'D': return "mouth_phoneme_D"; 
            case 'E': return "mouth_phoneme_E"; 
            case 'F': return "mouth_phoneme_F"; 
            case 'G': return "mouth_phoneme_G"; 
            case 'H': return "mouth_phoneme_H"; 
            case 'X': return "mouth_phoneme_X"; 
            default:  return "mouth_phoneme_X";
        }
    }
};

// ---------------------------------------------------------
// 3. PHYSICS HELPER
// ---------------------------------------------------------
struct Spring { float val = 1.0f, vel = 0.0f, target = 1.0f; };

void UpdateSpring(Spring& s, float k, float d, float dt) {
    float f = (s.target - s.val) * k;
    s.vel = (s.vel + f * dt) * d;
    s.val += s.vel * dt;
}

// ---------------------------------------------------------
// 4. BMO PUPPET
// ---------------------------------------------------------
struct BMO {
    SpriteAtlas atlas;
    LipSyncSystem lips;
    
    std::string currentMood = "face_happy_standard"; 
    
    Spring scaleY, scaleX;
    Vector2 lookOffset = {0,0};
    
    // Procedural Animation Offsets
    Vector2 shakeOffset = {0,0};
    float pulseScale = 1.0f;

    float blinkTimer = 0.0f, nextBlink = 3.0f;
    bool isBlinking = false;

    // DEBUG
    bool debugMode = false;
    int debugIndex = 0;
    std::vector<std::string> phonemes = {
        "mouth_phoneme_A", "mouth_phoneme_B", "mouth_phoneme_C", 
        "mouth_phoneme_D", "mouth_phoneme_E", "mouth_phoneme_F", 
        "mouth_phoneme_G", "mouth_phoneme_H", "mouth_phoneme_X",
        "mouth_phoneme_Th", "mouth_phoneme_R", "mouth_phoneme_J"
    };

    void Init() {
        // [FIXED] Using the filename that works for you!
        atlas.Load("assets/BMO_Animation_LipSyncSprite.png", 
                   "assets/BMO_Animation_Lipsync.json");
                   
        lips.Load("assets/output.tsv"); 
    }

    void Update(float dt, float audioTime, bool isPlaying) {
        // --- DEBUG ---
        if (IsKeyPressed(KEY_TAB)) debugMode = !debugMode;
        if (debugMode) {
            if (IsKeyPressed(KEY_RIGHT)) { debugIndex++; if (debugIndex >= phonemes.size()) debugIndex = 0; }
            if (IsKeyPressed(KEY_LEFT)) { debugIndex--; if (debugIndex < 0) debugIndex = phonemes.size() - 1; }
        }

        // --- 1. MOOD FX (Procedural Animation) ---
        float time = (float)GetTime();
        shakeOffset = {0, 0};
        pulseScale = 1.0f;

        // EFFECT: RAGE / WAIL (Shaking)
        if (currentMood.find("angry") != std::string::npos || currentMood.find("wail") != std::string::npos) {
            shakeOffset.x = (float)GetRandomValue(-3, 3);
            shakeOffset.y = (float)GetRandomValue(-3, 3);
        }
        
        // EFFECT: SOBBING (Slow heavy bounce)
        else if (currentMood.find("cry") != std::string::npos || currentMood.find("worried") != std::string::npos) {
            shakeOffset.y = sinf(time * 8.0f) * 5.0f; 
        }

        // EFFECT: SPARKLE / EXCITED (Pulsing)
        else if (currentMood.find("star") != std::string::npos || currentMood.find("excited") != std::string::npos || currentMood.find("kawaii") != std::string::npos) {
            float pulse = sinf(time * 12.0f) * 0.05f; 
            pulseScale = 1.0f + pulse;
        }

        // --- 2. BLINKING ---
        bool canBlink = (currentMood.find("closed") == std::string::npos) && 
                        (currentMood.find("winc") == std::string::npos) &&
                        !debugMode;

        if (canBlink) {
            blinkTimer += dt;
            if (blinkTimer >= nextBlink) {
                isBlinking = true;
                if (blinkTimer >= nextBlink + 0.15f) {
                    isBlinking = false;
                    blinkTimer = 0.0f;
                    nextBlink = GetRandomValue(20, 60) / 10.0f;
                }
            }
        } else {
            isBlinking = false;
        }

        // --- 3. MOUSE LOOK ---
        Vector2 mouse = GetMousePosition();
        Vector2 center = { (float)GetScreenWidth()/2, (float)GetScreenHeight()/2 };
        Vector2 dir = { mouse.x - center.x, mouse.y - center.y };
        float dist = sqrtf(dir.x*dir.x + dir.y*dir.y);
        
        if (dist > 0) {
            float move = fminf(dist * 0.05f, 15.0f); 
            lookOffset = { (dir.x/dist)*move, (dir.y/dist)*move };
        }

        // --- 4. PHYSICS (Squash & Stretch) ---
        float breath = sinf(time * 2.5f) * 0.008f;
        scaleY.target = 1.0f + breath;
        scaleX.target = 1.0f - breath * 0.5f;

        if (isPlaying && !debugMode) {
            std::string shape = lips.GetMouthForTime(audioTime);
            if (shape == "mouth_phoneme_D" || shape == "mouth_phoneme_C") {
                scaleY.target += 0.03f; 
                scaleX.target -= 0.015f;
            }
        }

        UpdateSpring(scaleY, 150.0f, 0.6f, dt);
        UpdateSpring(scaleX, 150.0f, 0.6f, dt);
    }

    void Draw(int W, int H, float audioTime, bool isPlaying) {
        // --- TEXTURE SELECTION ---
        std::string eyesTex = isBlinking ? "face_happy_closed_eyes_eyes" : currentMood + "_eyes";
        std::string mouthTex;
        Vector2 mouthOffset = {0, 0};

        if (debugMode) {
            mouthTex = phonemes[debugIndex];
            if (mouthTex.find("mouth_phoneme_") != std::string::npos && mouthTex != "mouth_phoneme_X") {
                mouthOffset.y = -10.0f; 
            }
        } 
        else if (isPlaying) {
            mouthTex = lips.GetMouthForTime(audioTime);
            if (mouthTex.find("mouth_phoneme_") != std::string::npos && mouthTex != "mouth_phoneme_X") {
                mouthOffset.y = -10.0f; 
            }
        } 
        else {
            mouthTex = currentMood + "_mouth"; 
        }

        // --- COMPOSITE TRANSFORMS ---
        Vector2 finalScale = { scaleX.val * pulseScale, scaleY.val * pulseScale };
        
        Vector2 eyesFinalOffset = { lookOffset.x + shakeOffset.x, lookOffset.y + shakeOffset.y };
        Vector2 mouthFinalOffset = { mouthOffset.x + shakeOffset.x, mouthOffset.y + shakeOffset.y };

        // --- DRAW ---
        atlas.DrawFrame(eyesTex, 0, 0, W, H, eyesFinalOffset, finalScale, WHITE);
        atlas.DrawFrame(mouthTex, 0, 0, W, H, mouthFinalOffset, finalScale, WHITE); 
        
        // DEBUG UI
        if (debugMode) {
            DrawText("DEBUG MODE", 20, H - 60, 20, YELLOW);
            DrawText(TextFormat("Mouth: %s", mouthTex.c_str()), 20, H - 30, 20, WHITE);
        }
    }
};

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    const int W = 1280, H = 720;
    InitWindow(W, H, "BMO Engine: Animated Emotions");
    InitAudioDevice();
    SetTargetFPS(60);

    BMO bmo;
    bmo.Init();

    Music voice = LoadMusicStream("assets/song.wav"); 
    bool isPlaying = false;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        UpdateMusicStream(voice);

        // --- CONTROLS ---
        if (IsKeyPressed(KEY_SPACE)) {
            if (IsMusicStreamPlaying(voice)) { StopMusicStream(voice); isPlaying = false; }
            else { PlayMusicStream(voice); isPlaying = true; }
        }
        
        // --- MOOD SWITCHING ---
        if (IsKeyPressed(KEY_H)) bmo.currentMood = "face_happy_standard";
        if (IsKeyPressed(KEY_S)) bmo.currentMood = "face_sad_standard";
        if (IsKeyPressed(KEY_A)) bmo.currentMood = "face_angry_shout";     // SHAKES
        if (IsKeyPressed(KEY_W)) bmo.currentMood = "face_crying_wail";     // SHAKES
        if (IsKeyPressed(KEY_C)) bmo.currentMood = "face_crying_tears";    // SOBS
        if (IsKeyPressed(KEY_E)) bmo.currentMood = "face_excited_stars";   // PULSES
        if (IsKeyPressed(KEY_K)) bmo.currentMood = "face_kawaii_sparkle";  // PULSES

        float time = GetMusicTimePlayed(voice);
        if (!IsMusicStreamPlaying(voice)) time = 0.0f;
        
        bmo.Update(dt, time, isPlaying);

        BeginDrawing();
        ClearBackground({131, 220, 169, 255}); 
        
        bmo.Draw(W, H, time, isPlaying);
        
        if (!bmo.debugMode) {
            DrawText("SPACE: Audio | TAB: Debug", 20, 20, 20, DARKGRAY);
            DrawText("H: Happy | S: Sad | A: Angry | C: Cry | E: Excited", 20, 50, 20, DARKGRAY);
        }
        
        EndDrawing();
    }

    UnloadMusicStream(voice);
    CloseAudioDevice();
    CloseWindow();
    return 0;
}