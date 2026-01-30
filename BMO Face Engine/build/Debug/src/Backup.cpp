#include "raylib.h"
#include "json.hpp"
#include <fstream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <raymath.h> // Required for Vector2 operations

using json = nlohmann::json;

// ---------------------------------------------------------
// 1. SPRITE ATLAS (For Eyes and Body)
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

        // Support both array structures (TexturePacker generic)
        if (data.contains("textures")) {
            for (auto& t : data["textures"]) {
                for (auto& fr : t["frames"]) {
                    frames[fr["filename"]] = {
                        (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                        (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                    };
                }
            }
        } else if (data.contains("frames")) {
            // Standard JSON Array format
            for (auto& fr : data["frames"]) {
                 frames[fr["filename"]] = {
                    (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                    (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                };
            }
        }
        TraceLog(LOG_INFO, "Atlas Loaded: %d frames", (int)frames.size());
    }

    void Unload() { UnloadTexture(texture); }

    // Draws a sprite with position, scale, and rotation (for juicy physics)
    void DrawPartJuicy(const std::string& name, Vector2 pos, Vector2 scale, float rot) {
        if (frames.find(name) == frames.end()) {
            // Draw Magenta Box if missing
            DrawRectangleLines(pos.x - 50, pos.y - 50, 100, 100, MAGENTA);
            return;
        }

        Rectangle src = frames[name];
        Rectangle dst = { pos.x, pos.y, src.width * scale.x, src.height * scale.y };
        Vector2 origin = { dst.width / 2.0f, dst.height / 2.0f };

        DrawTexturePro(texture, src, dst, origin, rot, WHITE);
    }
};

// ---------------------------------------------------------
// 2. PHYSICS HELPERS
// ---------------------------------------------------------
struct Spring { 
    float val = 0.0f; 
    float vel = 0.0f; 
    float target = 0.0f; 
};

void UpdateSpring(Spring& s, float stiffness, float damping, float dt) {
    float force = (s.target - s.val) * stiffness;
    s.vel = (s.vel + force * dt) * damping;
    s.val += s.vel * dt;
}

Vector2 GetBezierPoint(Vector2 p0, Vector2 p1, Vector2 p2, float t) {
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;
    return {
        uu * p0.x + 2 * u * t * p1.x + tt * p2.x,
        uu * p0.y + 2 * u * t * p1.y + tt * p2.y
    };
}

void DrawBezierQuad(Vector2 p0, Vector2 p1, Vector2 p2, float thick, Color col, int segments = 24) {
    Vector2 prev = p0;
    for (int i = 1; i <= segments; i++) {
        float t = (float)i / (float)segments;
        Vector2 cur = GetBezierPoint(p0, p1, p2, t);
        DrawLineEx(prev, cur, thick, col);
        prev = cur;
    }
}


float Rand01() { return GetRandomValue(0, 10000) / 10000.0f; }

// ---------------------------------------------------------
// 3. SPEECH ENVELOPE (The "Rhythm" of talking)
// ---------------------------------------------------------
struct SpeechCadence {
    float timer = 0.0f;
    float nextPulse = 0.1f;
    float env = 0.0f;       // 0.0 (Silent) to 1.0 (Loud)
    float envTarget = 0.0f; 
    
    void Update(bool talking, float dt) {
        if (!talking) {
            envTarget = 0.0f;
            // Decay silence
            env += (envTarget - env) * 10.0f * dt;
            return;
        }

        timer += dt;
        if (timer >= nextPulse) {
            timer = 0.0f;
            // Randomize rhythm (Fast syllables vs pauses)
            if (Rand01() < 0.2f) nextPulse = 0.15f + Rand01() * 0.2f; // Pause
            else                 nextPulse = 0.05f + Rand01() * 0.1f; // Fast

            // Set new target volume
            float emphasis = 0.7f + Rand01() * 0.5f; 
            envTarget = emphasis; 
        } else {
            // Natural decay between syllables
            envTarget *= 0.90f; 
        }

        // Smoothly move current envelope to target
        env += (envTarget - env) * 20.0f * dt;
        // Clamp
        if (env < 0.0f) env = 0.0f;
        if (env > 1.0f) env = 1.0f;
    }
};


// ---------------------------------------------------------
// 4. PROCEDURAL BEZIER MOUTH RIG
// ---------------------------------------------------------
struct BMOMouthRig {
    // TARGETS
    float openT = 0.0f;     // 0..1
    float curveT = 0.0f;    // -1 (Sad/Angry) .. 0 (Flat) .. 1 (Happy/Scared)
    float widthT = 0.5f;    // 0..1
    float teethT = 0.0f;    // 0..1

    // CURRENT STATE (Smoothed)
    float open = 0.0f;
    float curve = 0.0f;
    float width = 0.0f;
    float teeth = 0.0f;

    // Colors
    Color colLine = BLACK;
    Color colCavity = {36, 61, 38, 255};    // Dark Green
    Color colTongue = {156, 178, 114, 255}; // Muted Green
    Color colTeeth  = WHITE;

    void Update(float dt) {
        float speed = 15.0f * dt;
        open  += (openT - open) * speed;
        curve += (curveT - curve) * speed;
        width += (widthT - width) * speed;
        teeth += (teethT - teeth) * speed;
    }

    Vector2 GetBezierPoint(Vector2 p0, Vector2 p1, Vector2 p2, float t) {
        float u = 1.0f - t;
        float tt = t * t;
        float uu = u * u;
        return {
            uu * p0.x + 2 * u * t * p1.x + tt * p2.x,
            uu * p0.y + 2 * u * t * p1.y + tt * p2.y
        };
    }

    void Draw(Vector2 center, float scale) {
        float currentWidth = 140.0f * scale * (0.8f + width * 0.4f);
        float halfW = currentWidth / 2.0f;

        // Shape Logic
        // Curve determines the arch of the top lip
        float topLipY = -10.0f * scale;
        float topControlY = topLipY - (curve * 35.0f * scale); 

        // Open determines bottom lip depth
        float bottomLipY = topLipY + (open * 110.0f * scale);
        float bottomControlY = bottomLipY + (40.0f * scale);

        // Control Points
        Vector2 cornerLeft  = { center.x - halfW, center.y + topLipY };
        Vector2 cornerRight = { center.x + halfW, center.y + topLipY };
        Vector2 topControl     = { center.x, center.y + topControlY };
        Vector2 bottomControl  = { center.x, center.y + bottomControlY };

        // 1. FILL CAVITY (Triangle Fan for Polygon)
        if (open > 0.05f) {
            std::vector<Vector2> points;
            Vector2 polyCenter = { center.x, center.y + (topLipY + bottomLipY)/2.0f };
            points.push_back(polyCenter); // Center first

            // Top Edge
            for (int i = 0; i <= 20; i++) 
                points.push_back(GetBezierPoint(cornerLeft, topControl, cornerRight, (float)i/20.0f));
            
            // Bottom Edge (Reverse)
            for (int i = 0; i <= 20; i++) 
                points.push_back(GetBezierPoint(cornerRight, bottomControl, cornerLeft, (float)i/20.0f));
            
            // Close loop
            points.push_back(GetBezierPoint(cornerLeft, topControl, cornerRight, 0.0f));

            DrawTriangleFan(points.data(), points.size(), colCavity);
        }

        // 2. TONGUE
        if (open > 0.3f) {
            float tongueH = 40.0f * scale * open;
            DrawEllipse((int)center.x, (int)(center.y + bottomLipY - tongueH*0.3f), 
                        (int)(halfW * 0.5f), (int)tongueH, colTongue);
        }

        // 3. TEETH
        if (teeth > 0.1f && open > 0.1f) {
            float teethH = 25.0f * scale * teeth;
            DrawRectangle((int)(center.x - halfW*0.6f), (int)(center.y + topLipY), 
                          (int)(halfW*1.2f), (int)teethH, colTeeth);
        }

        // 4. OUTLINES
        float thick = 8.0f * scale;
        DrawBezierQuad(cornerLeft, topControl, cornerRight, thick, colLine);

        if (open > 0.05f) {
            DrawBezierQuad(cornerRight, bottomControl, cornerLeft, thick, colLine);
        } else {
            // Closed line with slight curve
            Vector2 closedControl = { center.x, center.y + topLipY + (curve * 10.0f) };
            DrawBezierQuad(cornerLeft, closedControl, cornerRight, thick, colLine);
        }
    }
};

// ---------------------------------------------------------
// 5. MAIN
// ---------------------------------------------------------
int main() {
    const int W = 1280;
    const int H = 720;
    InitWindow(W, H, "BMO Face Engine - Final Procedural Rig");
    SetTargetFPS(60);

    // Load Assets
    SpriteAtlas atlas;
    // IMPORTANT: Make sure this path is correct for your setup
    atlas.Load("assets/BMO_SpriteSheet_Texture.png", "assets/BMO_Split_Data.json");

    // State
    bool isTalking = false;
    std::string currentEyes = "face_happy_standard_eyes";

    // Components
    SpeechCadence speech;
    BMOMouthRig mouth;
    
    // Physics Springs (Squash & Stretch)
    Spring scaleY { 1.0f, 0.0f, 1.0f };
    Spring scaleX { 1.0f, 0.0f, 1.0f };
    
    // BMO Colors
    Color bg = {131, 220, 169, 255};

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();

        // ---------------- INPUT ----------------
        if (IsKeyPressed(KEY_SPACE)) isTalking = !isTalking;

        if (IsKeyPressed(KEY_H)) {
            // HAPPY (Flat D-Shape)
            currentEyes = "face_happy_standard_eyes";
            mouth.curveT = 0.0f;  // Flat top
            mouth.teethT = 1.0f;  // Show teeth
            mouth.openT = 0.0f;   // Closed initially
        }
        if (IsKeyPressed(KEY_S)) {
            // SCARED/WORRIED (Bean Shape - Arched up)
            currentEyes = "face_worried_teary_eyes";
            mouth.curveT = 0.8f;  // Curve up
            mouth.teethT = 1.0f;
            mouth.openT = 0.0f;
        }
        if (IsKeyPressed(KEY_A)) {
            // ANGRY (Kidney Shape - Dipped down)
            currentEyes = "face_angry_shout_eyes";
            mouth.curveT = -0.8f; // Curve down
            mouth.teethT = 0.0f;  // No teeth
            mouth.openT = 0.0f;
        }

        // ---------------- UPDATE ----------------
        // 1. Speech
        speech.Update(isTalking, dt);

        // 2. Drive Mouth Rig
        // Open based on speech volume
        mouth.openT = speech.env;
        // Widen slightly when loud
        mouth.widthT = 0.5f + (speech.env * 0.2f); 

        mouth.Update(dt);

        // 3. Drive Physics (Juice)
        // Face stretches vertically when talking
        scaleY.target = 1.0f + (speech.env * 0.05f);
        scaleX.target = 1.0f - (speech.env * 0.03f); // Squash horizontally
        
        // Add subtle breathing
        scaleY.target += sinf(GetTime() * 2.0f) * 0.005f;

        UpdateSpring(scaleY, 150.0f, 0.6f, dt);
        UpdateSpring(scaleX, 150.0f, 0.6f, dt);

        // ---------------- DRAW ----------------
        BeginDrawing();
        ClearBackground(bg);

        float centerX = W / 2.0f;
        float centerY = H / 2.0f;

        // 1. Draw Eyes (Sprite) - Top Half
        // Offset -150 to sit above center
        atlas.DrawPartJuicy(
            currentEyes, 
            { centerX, centerY - 150.0f }, 
            { scaleX.val, scaleY.val }, 
            0.0f
        );

        // 2. Draw Mouth (Procedural Rig) - Bottom Half
        // Offset +120 to sit below center
        // We pass the scaleY spring to the mouth size too for extra bounce
        mouth.Draw({ centerX, centerY + 120.0f }, scaleY.val);

        // UI
        DrawText(isTalking ? "TALKING" : "SILENT", 20, 20, 20, isTalking ? RED : DARKGRAY);
        DrawText("Controls: H (Happy), S (Scared), A (Angry), SPACE (Talk)", 20, 50, 20, DARKGRAY);

        EndDrawing();
    }

    atlas.Unload();
    CloseWindow();
    return 0;
}