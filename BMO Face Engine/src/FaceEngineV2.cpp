#include "raylib.h"
#include "json.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cctype>

// INCLUDE THE SHADER ENGINE
#include "ShaderParametricMouth.cpp"

using json = nlohmann::json;

// ---------------------------------------------------------
// 1. HELPER FUNCTIONS (For Parsing)
// ---------------------------------------------------------
static std::vector<float> ParseFloats(std::string line) {
    std::vector<float> res;
    std::string numStr;
    for (char c : line) {
        // Allow digits, dots, minus signs, and scientific notation 'e'
        if (isdigit(c) || c == '.' || c == '-' || c == 'e') {
            numStr += c;
        } 
        else {
            if (!numStr.empty()) {
                try { res.push_back(std::stof(numStr)); } catch(...) {}
                numStr.clear();
            }
        }
    }
    if (!numStr.empty()) { try { res.push_back(std::stof(numStr)); } catch(...) {} }
    return res;
}

// ---------------------------------------------------------
// 2. VISEME DATABASE (Text File Loader)
// ---------------------------------------------------------
struct VisemeDatabase {
    std::unordered_map<std::string, FacialParams> data;

    void Load(const char* filename) {
        data.clear();
        std::ifstream in(filename);
        if (!in.is_open()) {
            std::cout << "[DB] Error: Could not open " << filename << std::endl;
            return;
        }

        std::string line;
        std::string currentName;
        FacialParams currentParams = MakeDefaultParams();
        bool parsingEntry = false;

        while (std::getline(in, line)) {
            // 1. Detect Entry Name: visemes["NAME"]
            size_t namePos = line.find("visemes[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                if (endPos != std::string::npos) {
                    currentName = line.substr(namePos + 9, endPos - (namePos + 9));
                    currentParams = MakeDefaultParams(); // Reset
                    parsingEntry = true;
                }
            }

            // 2. Detect Values (Assumes values follow the name or are on lines starting with '{')
            // The file format usually has data on the same line or next line inside { ... }
            if (parsingEntry) {
                std::vector<float> v = ParseFloats(line);
                
                // If we found a chunk of numbers, map them to params
                if (v.size() >= 14) { // Heuristic: valid entries have lots of floats
                     if(v.size() > 0) currentParams.open = v[0];
                     if(v.size() > 1) currentParams.width = v[1];
                     if(v.size() > 2) currentParams.curve = v[2];
                     if(v.size() > 3) currentParams.squeezeTop = v[3];
                     if(v.size() > 4) currentParams.squeezeBottom = v[4];
                     if(v.size() > 5) currentParams.teethY = v[5];
                     if(v.size() > 6) currentParams.tongueUp = v[6];
                     if(v.size() > 7) currentParams.tongueX = v[7];
                     if(v.size() > 8) currentParams.tongueWidth = v[8];
                     if(v.size() > 9) currentParams.asymmetry = v[9];
                     if(v.size() > 10) currentParams.squareness = v[10];
                     if(v.size() > 11) currentParams.teethWidth = v[11];
                     if(v.size() > 12) currentParams.teethGap = v[12];
                     if(v.size() > 13) currentParams.scale = v[13];
                     if(v.size() > 14) currentParams.outlineThickness = v[14];
                     if(v.size() > 15) currentParams.sigma = v[15];
                     if(v.size() > 16) currentParams.power = v[16];
                     if(v.size() > 17) currentParams.maxLiftValue = v[17];
                     
                     // Store and reset
                     data[currentName] = currentParams;
                     parsingEntry = false; 
                }
            }
        }
        std::cout << "[DB] Loaded " << data.size() << " visemes from disk.\n";
    }

    FacialParams Get(const std::string& name) {
        // Direct lookup
        if (data.count(name)) return data[name];
        
        // Suffix fallback (e.g., if code asks for "angry" but file has "angry_mouth")
        if (data.count(name + "_mouth")) return data[name + "_mouth"];

        // Safety fallback
        if (data.count("face_happy_standard_mouth")) return data["face_happy_standard_mouth"];
        
        return MakeDefaultParams();
    }
};

// ---------------------------------------------------------
// 3. ASSET LOADER (Sprites)
// ---------------------------------------------------------
struct SpriteAtlas {
    Texture2D texture{};
    std::unordered_map<std::string, Rectangle> frames;

    void Load(const char* imagePath, const char* jsonPath) {
        texture = LoadTexture(imagePath);
        std::ifstream f(jsonPath);
        if (!f.is_open()) return;
        json data = json::parse(f, nullptr, false);
        if (data.is_discarded()) return;

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
    }

    void DrawFrame(const std::string& name, int x, int y, int screenW, int screenH, Vector2 offset, Vector2 scale, Color tint) {
        if (frames.find(name) == frames.end()) return;
        Rectangle src = frames[name];
        
        float destW = screenW * scale.x;
        float destH = screenH * scale.y;
        
        Rectangle dest = {
            (float)x + (screenW - destW) / 2.0f + offset.x, 
            (float)y + (screenH - destH) / 2.0f + offset.y, 
            destW, destH
        };
        DrawTexturePro(texture, src, dest, {0,0}, 0.0f, tint);
    }
    void Unload() { UnloadTexture(texture); }
};

// ---------------------------------------------------------
// 4. LIP SYNC PARSER
// ---------------------------------------------------------
struct LipSyncSystem {
    struct Cue { float time; char shape; };
    std::vector<Cue> cues;

    void Load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return;
        
        float t; std::string s;
        cues.clear();
        while (f >> t >> s) cues.push_back({t, s[0]}); 
    }

    std::string GetMouthForTime(float time) {
        if (cues.empty()) return "mouth_phoneme_X"; 
        char shape = 'X';
        for (const auto& cue : cues) {
            if (cue.time > time) break;
            shape = cue.shape;
        }
        // Map char to specific phoneme keys in DB
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
// 5. BMO PUPPET
// ---------------------------------------------------------
struct BMO {
    SpriteAtlas atlas;
    LipSyncSystem lips;
    ParametricMouth rig;
    VisemeDatabase db; // <-- Using the new file-based DB
    
    std::string currentMood = "face_happy_standard"; 
    
    // Procedural Animation
    float pulseScale = 1.0f;
    Vector2 shakeOffset = {0,0};
    
    // Blinking
    float blinkTimer = 0.0f, nextBlink = 3.0f;
    bool isBlinking = false;

    bool debugMode = false;

    void Init(int screenW, int screenH) {
        atlas.Load("assets/BMO_Animation_LipSyncSprite.png", 
                   "assets/BMO_Animation_Lipsync.json");
        lips.Load("assets/output.tsv"); 

        // 1. Initialize Rig
        rig.Init({ (float)screenW/2.0f, (float)screenH/2.0f + 100.0f });
        
        // 2. Load Database from file
        db.Load("visemes_database.txt");
        
        // 3. Set Initial State
        rig.target = db.Get(currentMood + "_mouth");
        rig.current = rig.target;
    }

    void Update(float dt, float audioTime, bool isPlaying) {
        if (IsKeyPressed(KEY_TAB)) debugMode = !debugMode;

        // --- PROC ANIM ---
        float time = (float)GetTime();
        shakeOffset = {0, 0};
        pulseScale = 1.0f;

        // Effects based on mood string
        if (currentMood.find("angry") != std::string::npos || currentMood.find("wail") != std::string::npos) {
            shakeOffset.x = (float)GetRandomValue(-3, 3);
            shakeOffset.y = (float)GetRandomValue(-3, 3);
        }
        else if (currentMood.find("cry") != std::string::npos) {
            shakeOffset.y = sinf(time * 8.0f) * 5.0f; 
        }
        else if (currentMood.find("star") != std::string::npos || currentMood.find("excited") != std::string::npos) {
            float pulse = sinf(time * 12.0f) * 0.05f; 
            pulseScale = 1.0f + pulse;
        }

        // --- BLINK ---
        bool canBlink = (currentMood.find("closed") == std::string::npos) && (currentMood.find("winc") == std::string::npos);
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
        } else isBlinking = false;

        // --- TARGET MOUTH LOGIC ---
        std::string targetName;

        if (isPlaying) {
             // If song is playing, get Phoneme
             targetName = lips.GetMouthForTime(audioTime);
        } else {
             // If idle, get Mood Mouth
             targetName = currentMood + "_mouth";
        }

        // Retrieve from Database
        rig.target = db.Get(targetName);

        // Apply Position Shake
        Vector2 basePos = { (float)GetScreenWidth()/2.0f, (float)GetScreenHeight()/2.0f + 100.0f };
        rig.centerPos = { basePos.x + shakeOffset.x, basePos.y + shakeOffset.y };
        
        // Apply Physics & Generate
        rig.UpdatePhysics(dt);
        rig.GenerateGeometry();
    }

    void Draw(int W, int H) {
        // Draw Eyes (Sprite)
        std::string eyesTex = isBlinking ? "face_happy_closed_eyes_eyes" : currentMood + "_eyes";
        Vector2 finalScale = { pulseScale, pulseScale };
        atlas.DrawFrame(eyesTex, 0, 0, W, H, shakeOffset, finalScale, WHITE);

        // Draw Mouth (Shader)
        rig.Draw();
    }
    
    void Unload() {
        atlas.Unload();
        rig.Unload();
    }
};

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    const int W = 1280, H = 720;
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(W, H, "BMO Engine: Data-Driven");
    InitAudioDevice();
    SetTargetFPS(60);

    BMO bmo;
    bmo.Init(W, H);

    Music voice = LoadMusicStream("assets/song.wav"); 
    bool isPlaying = false;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        UpdateMusicStream(voice);

        // --- INPUT ---
        if (IsKeyPressed(KEY_SPACE)) {
            if (IsMusicStreamPlaying(voice)) { StopMusicStream(voice); isPlaying = false; }
            else { PlayMusicStream(voice); isPlaying = true; }
        }
        
        // Reload DB on 'R' (Great for tweaking while running!)
        if (IsKeyPressed(KEY_R)) {
            bmo.db.Load("visemes_database.txt");
            TraceLog(LOG_INFO, "Visemes Reloaded!");
        }

        // Moods
        if (IsKeyPressed(KEY_H)) bmo.currentMood = "face_happy_standard";
        if (IsKeyPressed(KEY_S)) bmo.currentMood = "face_sad_standard";
        if (IsKeyPressed(KEY_A)) bmo.currentMood = "face_angry_shout";
        if (IsKeyPressed(KEY_W)) bmo.currentMood = "face_crying_wail";
        if (IsKeyPressed(KEY_C)) bmo.currentMood = "face_crying_tears";
        if (IsKeyPressed(KEY_E)) bmo.currentMood = "face_excited_stars";
        if (IsKeyPressed(KEY_K)) bmo.currentMood = "face_kawaii_sparkle";
        if (IsKeyPressed(KEY_Z)) bmo.currentMood = "face_hypnotized";
        if (IsKeyPressed(KEY_X)) bmo.currentMood = "face_shocked_pale";

        float time = GetMusicTimePlayed(voice);
        if (!IsMusicStreamPlaying(voice)) time = 0.0f;
        
        bmo.Update(dt, time, isPlaying);

        BeginDrawing();
        ClearBackground({201, 228, 195, 255}); // BMO Face Color

        
        bmo.Draw(W, H);
        
        DrawText("SPACE: Audio | R: Reload Database", 20, 20, 20, DARKGRAY);
        DrawText("H,S,A,W,C,E,K,Z,X for Moods", 20, 50, 20, DARKGRAY);
        
        if (bmo.debugMode) DrawText("DEBUG MODE", 20, H - 40, 20, YELLOW);
        
        EndDrawing();
    }

    UnloadMusicStream(voice);
    bmo.Unload();
    CloseAudioDevice();
    CloseWindow();
    return 0;
}