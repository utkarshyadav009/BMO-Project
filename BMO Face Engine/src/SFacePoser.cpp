// SFacePoser.cpp
// DRIVER FILE: Links the GUI to the Shader-Based Mouth Engine

#include "raylib.h"

// 1. SETUP RAYGUI
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "json.hpp"
#include <fstream>
#include <sstream>
#include <cctype>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

// Include the NEW Shader-based Engine
// Ensure ParametricMouth.cpp is in the same folder (src/)
#include "ShaderParametricMouth.cpp" 

using json = nlohmann::json;



//Helper Functions
// helper: trim
static inline std::string Trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// helper: remove trailing 'f' and whitespace
static inline std::string StripFloatSuffix(std::string tok) {
    tok = Trim(tok);
    if (!tok.empty() && (tok.back() == 'f' || tok.back() == 'F')) tok.pop_back();
    tok = Trim(tok);
    return tok;
}

static bool ParseVisemeLine(const std::string& line, std::vector<float>& outVals) {
    outVals.clear();

    // Must contain { ... }
    size_t lb = line.find('{');
    size_t rb = line.find('}');
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return false;

    std::string inside = line.substr(lb + 1, rb - lb - 1);

    // Split by commas
    std::stringstream ss(inside);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = StripFloatSuffix(tok);
        if (tok.empty()) continue;

        try {
            outVals.push_back(std::stof(tok));
        } catch (...) {
            return false; // failed parse
        }
    }

    return !outVals.empty();
}

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
        << p.scale << "f, "<<p.outlineThickness<<"f };\n";
        file.close();
        std::cout << "Saved: " << name << std::endl;
    }
}

bool LoadTestFace(FacialParams& p, int viseme)
{
    std::ifstream in("visemes_database.txt");
    if (!in.is_open()) {
        std::cout << "Can't open input file!\n";
        return false;
    }

    std::string line;
    int visemeCount = 0;

    while (std::getline(in, line)) {
        line = Trim(line);
        if (line.empty()) continue;

        // Only count actual viseme assignment lines
        if (line.find("visemes[\"") == std::string::npos) continue;
        if (line.find('{') == std::string::npos) continue;
        if (line.find('}') == std::string::npos) continue;

        visemeCount++;

        viseme = visemeCount;
        // 4th viseme
        if (visemeCount == 4) {
            std::vector<float> vals;
            if (!ParseVisemeLine(line, vals)) {
                std::cout << "Failed to parse viseme line #4\n";
                return false;
            }

            // Your save order (AppendToDatabase):
            // open, width, curve, squeezeTop, squeezeBottom,
            // teethY,
            // tongueUp, tongueX, tongueWidth,
            // asymmetry, squareness,
            // teethWidth, teethGap,
            // scale, outlineThickness

            if (vals.size() < 15) {
                std::cout << "Viseme #4 has " << vals.size()
                          << " values, expected 15.\n";
                return false;
            }

            p.open            = vals[0];
            p.width           = vals[1];
            p.curve           = vals[2];
            p.squeezeTop      = vals[3];
            p.squeezeBottom   = vals[4];
            p.teethY          = vals[5];
            p.tongueUp        = vals[6];
            p.tongueX         = vals[7];
            p.tongueWidth     = vals[8];
            p.asymmetry       = vals[9];
            p.squareness      = vals[10];
            p.teethWidth      = vals[11];
            p.teethGap        = vals[12];
            p.scale           = vals[13];
            p.outlineThickness= vals[14];

            std::cout << "Loaded 4th viseme into FacialParams.\n";
            return true;
        }
    }

    std::cout << "File has only " << visemeCount << " viseme entries; cannot load 4th.\n";
    return false;
}
// ---------------------------------------------------------
// VISEME DATABASE MANAGER (ROBUST & DEBUGGABLE)
// ---------------------------------------------------------
struct LoadedViseme {
    std::string name;
    FacialParams mouthParams;
};

struct VisemeDatabase {
    std::vector<LoadedViseme> entries;
    std::string dropdownStr;

    // Helper: Extract all numbers from a string, ignoring everything else
    std::vector<float> ParseFloats(std::string line) {
        std::vector<float> res;
        std::string numStr;
        for (char c : line) {
            // Allow digits, dots, minus signs, and scientific notation 'e'
            if (isdigit(c) || c == '.' || c == '-' || c == 'e') {
                numStr += c;
            } 
            else {
                // If we hit a separator and have a number buffered, parse it
                if (!numStr.empty()) {
                    try { res.push_back(std::stof(numStr)); } catch(...) {}
                    numStr.clear();
                }
            }
        }
        // Capture last number if line ends with one
        if (!numStr.empty()) { try { res.push_back(std::stof(numStr)); } catch(...) {} }
        return res;
    }

    void Load(const char* filename) {
        entries.clear();
        dropdownStr = "";
        
        std::ifstream in(filename);
        if (!in.is_open()) {
            std::cout << "[DB] Error: Could not open " << filename << std::endl;
            return;
        }

        std::string line;
        LoadedViseme currentItem;
        bool parsingEntry = false;
        int lineNum = 0;

        while (std::getline(in, line)) {
            lineNum++;
            
            // 1. Detect New Entry: visemes["NAME"]
            size_t namePos = line.find("visemes[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                if (endPos != std::string::npos) {
                    currentItem.name = line.substr(namePos + 9, endPos - (namePos + 9));
                    currentItem.mouthParams = MakeDefaultParams(); // Default state
                    parsingEntry = true;
                    // std::cout << "[DB] Found Entry: " << currentItem.name << std::endl;
                }
            }

            // 2. Detect Mouth Data
            // We relaxed the check: just looks for "mouth" and "{" on the same line
            if (parsingEntry && line.find("mouth") != std::string::npos && line.find("{") != std::string::npos) {
                std::vector<float> v = ParseFloats(line);
                
                // DEBUG: Tell us what happened
                if (v.size() < 15) {
                    std::cout << "[DB] WARNING on line " << lineNum << " (" << currentItem.name << "): "
                              << "Expected 15 values, found " << v.size() << ". (Using defaults for missing)\n";
                }

                // Safely assign whatever data we found (Partial Load)
                if(v.size() > 0) currentItem.mouthParams.open = v[0];
                if(v.size() > 1) currentItem.mouthParams.width = v[1];
                if(v.size() > 2) currentItem.mouthParams.curve = v[2];
                if(v.size() > 3) currentItem.mouthParams.squeezeTop = v[3];
                if(v.size() > 4) currentItem.mouthParams.squeezeBottom = v[4];
                if(v.size() > 5) currentItem.mouthParams.teethY = v[5];
                if(v.size() > 6) currentItem.mouthParams.tongueUp = v[6];
                if(v.size() > 7) currentItem.mouthParams.tongueX = v[7];
                if(v.size() > 8) currentItem.mouthParams.tongueWidth = v[8];
                if(v.size() > 9) currentItem.mouthParams.asymmetry = v[9];
                if(v.size() > 10) currentItem.mouthParams.squareness = v[10];
                if(v.size() > 11) currentItem.mouthParams.teethWidth = v[11];
                if(v.size() > 12) currentItem.mouthParams.teethGap = v[12];
                if(v.size() > 13) currentItem.mouthParams.scale = v[13];
                if(v.size() > 14) currentItem.mouthParams.outlineThickness = v[14];
            }

            // 3. Detect End of Entry
            if (parsingEntry && line.find("};") != std::string::npos) {
                entries.push_back(currentItem);
                parsingEntry = false;
            }
        }

        // Build Dropdown String
        for (size_t i = 0; i < entries.size(); i++) {
            dropdownStr += entries[i].name;
            if (i < entries.size() - 1) dropdownStr += ";";
        }
        
        std::cout << "[DB] Successfully loaded " << entries.size() << " entries.\n";
    }
};
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
    // [FIX] DROPDOWN SPECIFIC STYLES
    // Make the background dark gray and text white so it's readable
    GuiSetStyle(DROPDOWNBOX, BASE_COLOR_NORMAL, ColorToInt({ 40, 40, 40, 255 }));   // Dark Gray Box
    GuiSetStyle(DROPDOWNBOX, BASE_COLOR_FOCUSED, ColorToInt({ 60, 60, 60, 255 })); // Lighter when clicked
    GuiSetStyle(DROPDOWNBOX, TEXT_COLOR_NORMAL, ColorToInt(WHITE));                // White Text
    GuiSetStyle(DROPDOWNBOX, TEXT_COLOR_FOCUSED, ColorToInt(WHITE));               // White Text when open

    ParametricMouth rig;
    // Initial Position (Right side of screen)
    rig.Init({ 800, 360 }); 
    VisemeDatabase db;
    db.Load("visemes_database.txt"); 

    int dropdownActive = 0;       // Selected index
    bool dropdownEditMode = false; // Is menu open?

    ReferenceAtlas atlas;
    atlas.Load("assets/BMO_Animation_LipSyncSprite.png",
               "assets/BMO_Animation_Lipsync.json");

    int currentIdx = 0;
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false; 
    Vector2 atlasPos = { rig.centerPos.x, rig.centerPos.y + 150.0f };


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
            atlas.Draw(atlas.mouthNames[currentIdx], atlasPos, 1.0f, refOpacity);
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
        DRAW_SLIDER("OutlineThickness",      rig.target.outlineThickness,         1.f,  4.0f);
        sy += 10.0f;
        DRAW_SLIDER("Open",       rig.target.open,          0.0f,  1.2f);
        DRAW_SLIDER("Width",      rig.target.width,         0.1f,  1.5f);
        DRAW_SLIDER("Curve",      rig.target.curve,        -1.0f,  1.0f);
        sy += 10.0f;
        DRAW_SLIDER("Sqze Top",   rig.target.squeezeTop,    0.0f,  1.0f);
        DRAW_SLIDER("Sqze Bot",   rig.target.squeezeBottom, 0.0f,  1.0f);
        sy += 10.0f;
        DRAW_SLIDER("Squeeze Sigma",   rig.sigma, 0.0f,  1.0f);
        DRAW_SLIDER("Squeeze Power",   rig.power, 0.0f,  10.0f);
        DRAW_SLIDER("Squeeze lift",   rig.maxLiftValue, 0.0f,  1.0f);
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
        // --- [NEW] DATABASE UI SECTION ---
        
        // 1. Capture the rectangle where the Dropdown will go
        //    (We draw it later, but we need the position now)
        Rectangle dropdownRect = { innerX, sy, innerW - 35, 30 };
        Rectangle reloadRect   = { innerX + innerW - 30, sy, 30, 30 };

        // 2. Draw RELOAD button (Tiny "R" to refresh file)
        if (GuiButton(reloadRect, "R")) {
            db.Load("visemes_database.txt");
        }
        
        // 3. Draw LOAD SELECTED button (Below the dropdown area)
        if (GuiButton({ innerX, sy + 35, innerW, 30 }, "LOAD SELECTED")) {
            if (!db.entries.empty() && dropdownActive < db.entries.size()) {
                rig.target = db.entries[dropdownActive].mouthParams;
                // Snap physics so it doesn't "drift"
                rig.current = rig.target; 
            }
        }
        
        sy += 80.0f; // Make space for the dropdown + load button

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

        if (GuiDropdownBox(dropdownRect, db.dropdownStr.c_str(), &dropdownActive, dropdownEditMode)) {
            dropdownEditMode = !dropdownEditMode;
        }

        EndDrawing();
    }

    rig.Unload();
    CloseWindow();
    return 0;
}