// FinalFacePoser.cpp
// UNIFIED DRIVER: Combined Eye and Mouth Editor
// USAGE: Compile with raylib, raygui, and json.hpp

#include "raylib.h"

// 1. SETUP RAYGUI
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "json.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <filesystem>

// Utility header (Assumed to exist per prompt, provides GlobalScaler)
#include "utility.h"

// ---------------------------------------------------------
// ENGINE INCLUDE
// ---------------------------------------------------------
// The authoritative unified shader backend
#include "ShaderParametricFace.cpp" 

using json = nlohmann::json;

// ---------------------------------------------------------
// UI CONSTANTS & HELPERS
// ---------------------------------------------------------
namespace UI {
    const float START_X = 20.0f;
    const float START_Y = 60.0f; // Shifted down for Tab Bar
    const float PANEL_WIDTH = 340.0f;
    const float LABEL_WIDTH = 90.0f;
    const float VAL_WIDTH = 40.0f;
    const float ROW_HEIGHT = 25.0f;
    
    float GetSliderWidth() {
        return PANEL_WIDTH - LABEL_WIDTH - VAL_WIDTH - 30.0f;
    }

    void Slider(const char* text, float* var, float min, float max, float& yPos) {
        GuiLabel({START_X + 10, yPos, LABEL_WIDTH, 20}, text);
        GuiSliderBar({START_X + 10 + LABEL_WIDTH, yPos, GetSliderWidth(), 20}, NULL, NULL, var, min, max);
        GuiLabel({START_X + 10 + LABEL_WIDTH + GetSliderWidth() + 5, yPos, VAL_WIDTH, 20}, TextFormat("%.2f", *var));
        yPos += ROW_HEIGHT;
    }

    bool Checkbox(const char* text, bool* var, float xOffset, float yPos) {
        return GuiCheckBox({START_X + xOffset, yPos, 20, 20}, text, var);
    }
}

// ---------------------------------------------------------
// UNIFIED DATA STRUCTURES
// ---------------------------------------------------------

// Holds the complete state of the face for saving/loading
struct FaceState {
    EyeParams eyes;
    MouthParams mouth;
    
    FaceState() {
        eyes = EyeParams(); 
        // Ensure defaults match preferred starting state
        eyes.scaleX = 1.0f; eyes.scaleY = 1.0f; eyes.spacing = 200.0f;
        
        mouth = MouthParams(); // Defaults handled in struct definition or Init
    }
};

// ---------------------------------------------------------
// 1. UNIFIED ATLAS LOADER
// ---------------------------------------------------------
struct ReferenceAtlas {
    Texture2D texture;
    std::unordered_map<std::string, Rectangle> frames;
    std::vector<std::string> faceNames;


    void Load(const char* img, const char* data) {
        texture = LoadTexture(img);
        if (texture.id == 0) {
            std::cerr << "[Atlas] Error loading texture: " << img << std::endl;
            return;
        }

        std::ifstream f(data);
        if (!f.good()) {
            std::cerr << "[Atlas] Error loading JSON: " << data << std::endl;
            return;
        }

        try {
            json j = json::parse(f);
            if (j.contains("textures")) {
                for (auto& t : j["textures"]) {
                    for (auto& fr : t["frames"]) {
                        std::string name = fr["filename"];
                        std::string lowerName = name;
                        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                        
                        frames[name] = {
                            (float)fr["frame"]["x"], (float)fr["frame"]["y"],
                            (float)fr["frame"]["w"], (float)fr["frame"]["h"]
                        };

               
                        faceNames.push_back(name);
                    }
                }
            }
            std::sort(faceNames.begin(), faceNames.end());
            
            std::cout << "[Atlas] Loaded " << faceNames.size() << " Faces." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Atlas] JSON Parse Error: " << e.what() << std::endl;
        }
    }

    void Draw(const std::string& name, Vector2 pos, float scale, float alpha) {
        if (name.empty() || frames.find(name) == frames.end()) return;
        Rectangle src = frames[name];
        Rectangle dest = { pos.x, pos.y, src.width * scale, src.height * scale };
        Vector2 origin = { dest.width/2, dest.height/2 };
        DrawTexturePro(texture, src, dest, origin, 0.0f, Fade(WHITE, alpha));
    }
};

// ---------------------------------------------------------
// 2. UNIFIED DATABASE MANAGER
// ---------------------------------------------------------
struct FaceDatabase {
    struct Entry {
        std::string name;
        FaceState state;
    };
    
    std::vector<Entry> entries;
    std::string dropdownStr;

    // Helper: Parse all floats from a string regardless of delimiters
    std::vector<float> ParseFloats(const std::string& line) {
        std::vector<float> res;
        std::string numStr;
        for (char c : line) {
            if (isdigit(c) || c == '.' || c == '-' || c == 'e') {
                numStr += c;
            } else {
                if (!numStr.empty()) {
                    try { res.push_back(std::stof(numStr)); } catch(...) {}
                    numStr.clear();
                }
            }
        }
        if (!numStr.empty()) { try { res.push_back(std::stof(numStr)); } catch(...) {} }
        return res;
    }

    void Load(const char* filename) {
        entries.clear();
        dropdownStr = "";
        
        std::ifstream in(filename);
        if (!in.is_open()) return;

        std::string line;
        while (std::getline(in, line)) {
            // Check for entry start: faces["name"] = { ... }
            size_t namePos = line.find("faces[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                if (endPos != std::string::npos) {
                    Entry e;
                    e.name = line.substr(namePos + 7, endPos - (namePos + 7));
                    
                    std::vector<float> v = ParseFloats(line);
                    
                    int idx = 0;
                    auto& ep = e.state.eyes;
                    auto& mp = e.state.mouth;

                    // --- MAPPING EYES ---
                    if (v.size() > idx) ep.eyeShapeID = v[idx++];
                    if (v.size() > idx) ep.bend = v[idx++];
                    if (v.size() > idx) ep.eyeThickness = v[idx++];
                    if (v.size() > idx) ep.eyeSide = v[idx++];
                    if (v.size() > idx) ep.scaleX = v[idx++];
                    if (v.size() > idx) ep.scaleY = v[idx++];
                    if (v.size() > idx) ep.angle = v[idx++];
                    if (v.size() > idx) ep.spacing = v[idx++];
                    if (v.size() > idx) ep.squareness = v[idx++];
                    
                    if (v.size() > idx) ep.stressLevel = v[idx++];
                    if (v.size() > idx) ep.gloomLevel = v[idx++];
                    if (v.size() > idx) ep.distortMode = (int)v[idx++];
                    if (v.size() > idx) ep.spiralSpeed = v[idx++];
                    
                    if (v.size() > idx) ep.lookX = v[idx++];
                    if (v.size() > idx) ep.lookY = v[idx++];
                    
                    if (v.size() > idx) ep.showBrow = (bool)v[idx++];
                    if (v.size() > idx) ep.useLowerBrow = (bool)v[idx++];
                    if (v.size() > idx) ep.eyebrowThickness = v[idx++];
                    if (v.size() > idx) ep.eyebrowLength = v[idx++];
                    if (v.size() > idx) ep.eyebrowSpacing = v[idx++];
                    if (v.size() > idx) ep.eyebrowX = v[idx++];
                    if (v.size() > idx) ep.eyebrowY = v[idx++];
                    if (v.size() > idx) ep.browScale = v[idx++];
                    if (v.size() > idx) ep.browSide = v[idx++];
                    if (v.size() > idx) ep.browAngle = v[idx++];
                    if (v.size() > idx) ep.browBend = v[idx++];
                    if (v.size() > idx) ep.browBendOffset = v[idx++];
                    
                    if (v.size() > idx) ep.showTears = (bool)v[idx++];
                    if (v.size() > idx) ep.showBlush = (bool)v[idx++];
                    if (v.size() > idx) ep.tearsLevel = v[idx++];
                    if (v.size() > idx) ep.blushMode = (int)v[idx++];
                    if (v.size() > idx) ep.blushScale = v[idx++];
                    if (v.size() > idx) ep.blushX = v[idx++];
                    if (v.size() > idx) ep.blushY = v[idx++];
                    if (v.size() > idx) ep.blushSpacing = v[idx++];
                    
                    if (v.size() > idx) ep.pixelation = v[idx++];

                    // --- MAPPING MOUTH ---
                    if (v.size() > idx) mp.open = v[idx++];
                    if (v.size() > idx) mp.width = v[idx++];
                    if (v.size() > idx) mp.curve = v[idx++];
                    if (v.size() > idx) mp.squeezeTop = v[idx++];
                    if (v.size() > idx) mp.squeezeBottom = v[idx++];
                    if (v.size() > idx) mp.teethY = v[idx++];
                    if (v.size() > idx) mp.tongueUp = v[idx++];
                    if (v.size() > idx) mp.tongueX = v[idx++];
                    if (v.size() > idx) mp.tongueWidth = v[idx++];
                    if (v.size() > idx) mp.asymmetry = v[idx++];
                    if (v.size() > idx) mp.squareness = v[idx++];
                    if (v.size() > idx) mp.teethWidth = v[idx++];
                    if (v.size() > idx) mp.teethGap = v[idx++];
                    if (v.size() > idx) mp.scale = v[idx++];
                    if (v.size() > idx) mp.outlineThickness = v[idx++];
                    if (v.size() > idx) mp.sigma = v[idx++];
                    if (v.size() > idx) mp.power = v[idx++];
                    if (v.size() > idx) mp.maxLiftValue = v[idx++];
                    if (v.size() > idx) mp.lookX = v[idx++];
                    if (v.size() > idx) mp.lookY = v[idx++];
                    if (v.size() > idx) mp.stressLines = v[idx++];
                    if (v.size() > idx) mp.showInnerMouth = (bool)v[idx++];
                    if (v.size() > idx) mp.isThreeShape = (bool)v[idx++];
                    if (v.size() > idx) mp.isDShape = (bool)v[idx++];

                    entries.push_back(e);
                }
            }
        }

        // Rebuild Dropdown
        for (size_t i = 0; i < entries.size(); i++) {
            dropdownStr += entries[i].name;
            if (i < entries.size() - 1) dropdownStr += ";";
        }
        std::cout << "[Database] Loaded " << entries.size() << " combined presets." << std::endl;
    }

    void Save(const char* filename, std::string name, const FaceState& s) {
        // Read file to memory to handle replacement
        std::ifstream infile(filename);
        std::vector<std::string> lines;
        std::string line;
        bool found = false;
        
        // Prepare New Line
        std::stringstream ss;
        ss << "faces[\"" << name << "\"] = { ";
        
        // Eyes
        const EyeParams& p = s.eyes;
        ss << p.eyeShapeID << "f, " << p.bend << "f, " << p.eyeThickness << "f, " << p.eyeSide << "f, "
           << p.scaleX << "f, " << p.scaleY << "f, " << p.angle << "f, " << p.spacing << "f, " << p.squareness << "f, "
           << p.stressLevel << "f, " << p.gloomLevel << "f, " << p.distortMode << ", " << p.spiralSpeed << "f, "
           << p.lookX << "f, " << p.lookY << "f, "
           << (int)p.showBrow << ", " << (int)p.useLowerBrow << ", " << p.eyebrowThickness << "f, " << p.eyebrowLength << "f, "
           << p.eyebrowSpacing << "f, " << p.eyebrowX << "f, " << p.eyebrowY << "f, " << p.browScale << "f, " << p.browSide << "f, " 
           << p.browAngle << "f, " << p.browBend << "f, " << p.browBendOffset << "f, "
           << (int)p.showTears << ", " << (int)p.showBlush << ", " << p.tearsLevel << "f, " << p.blushMode << ", "
           << p.blushScale << "f, " << p.blushX << "f, " << p.blushY << "f, " << p.blushSpacing << "f, " 
           << p.pixelation << "f, ";

        // Mouth
        const MouthParams& m = s.mouth;
        ss << m.open << "f, " << m.width << "f, " << m.curve << "f, " << m.squeezeTop << "f, " << m.squeezeBottom << "f, "
           << m.teethY << "f, " << m.tongueUp << "f, " << m.tongueX << "f, " << m.tongueWidth << "f, "
           << m.asymmetry << "f, " << m.squareness << "f, " << m.teethWidth << "f, " << m.teethGap << "f, "
           << m.scale << "f, " << m.outlineThickness << "f, " << m.sigma << "f, " << m.power << "f, " << m.maxLiftValue << "f, " 
           << m.lookX << "f, " << m.lookY << "f, "<< m.stressLines << "f, " << (int)m.showInnerMouth << "f, "<< (int)m.isThreeShape << "f, "<< (int)m.isDShape << " };";

        std::string newLine = ss.str();

        if (infile.is_open()) {
            while (std::getline(infile, line)) {
                if (line.find("faces[\"" + name + "\"]") != std::string::npos) {
                    lines.push_back(newLine);
                    found = true;
                } else {
                    lines.push_back(line);
                }
            }
            infile.close();
        }
        if (!found) lines.push_back(newLine);

        std::ofstream outfile(filename);
        for (const auto& l : lines) outfile << l << "\n";

        std::cout << "[Database] Saved: " << name << std::endl;
        Load(filename);
    }
};

// ---------------------------------------------------------
// EDITOR STATE
// ---------------------------------------------------------
struct EditorState {
    FaceState current;
    
    // Atlas Selection Indices
    int faceRefIdx = 0;
    
    // Editor Settings
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false;
    bool enableGUI = true;
    bool debugBoxes = false;
    int tabIndex = 0; 
    
    // Dropdown UI
    int dropdownActive = 0;
    bool dropdownEditMode = false;
    
    void CycleFace(ReferenceAtlas& atlas, int dir ){
		if(atlas.faceNames.empty()) return;
		faceRefIdx += dir;
		if(faceRefIdx < 0) faceRefIdx = (int)atlas.faceNames.size() - 1;
		if(faceRefIdx >= (int)atlas.faceNames.size()) faceRefIdx = 0;
	}
};

// ---------------------------------------------------------
// UI SUB-PANELS
// ---------------------------------------------------------
void DrawEyeControls(float& y, EyeParams& p) {
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 350}, "EYE SHAPE"); y += 20.0f;

    int shapeInt = (int)p.eyeShapeID;
    UI::Slider("Shape ID", &p.eyeShapeID, 0.0f, 12.0f, y);
    const char* shapeNames[] = { "Dot", "Line", "Arc", "Cross", "Star", "Heart", "Spiral", "Chevron", "Shuriken", "Kawaii", "Shocked", "Teary", "Colon Eyes" };
    if(shapeInt >= 0 && shapeInt <= 12) GuiLabel({UI::START_X + 10 + UI::LABEL_WIDTH, y - 25, UI::GetSliderWidth(), 20}, shapeNames[shapeInt]);

    if(p.eyeShapeID > 5.5f && p.eyeShapeID < 6.5f) UI::Slider("Spiral Spd", &p.spiralSpeed, -10.0f, 10.0f, y);

    UI::Slider("Bend", &p.bend, -2.0f, 2.0f, y);
    UI::Slider("Thickness", &p.eyeThickness, 1.0f, 30.0f, y);
    y += 5;
    UI::Slider("Scale X", &p.scaleX, 0.1f, 10.0f, y);
    UI::Slider("Scale Y", &p.scaleY, 0.1f, 10.0f, y);
    UI::Slider("Spacing", &p.spacing, 0.0f, 1000.0f, y);
    UI::Slider("Look X", &p.lookX, -300.0f, 300.0f, y);
    UI::Slider("Look Y", &p.lookY, -300.0f, 300.0f, y);
    UI::Slider("Angle", &p.angle, -180.0f, 180.0f, y);
    UI::Slider("Squareness", &p.squareness, 0.0f, 1.0f, y);
    UI::Slider("Pixelation", &p.pixelation,1.0f, 8.0f, y);
    y += 20.0f;

    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 480}, "EYE FX"); y += 20.0f;
    UI::Checkbox("Brows", &p.showBrow, 10, y);
    UI::Checkbox("Tears", &p.showTears, 90, y);
    UI::Checkbox("Blush", &p.showBlush, 170, y); y += 30.0f;

    if (p.showBrow) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- BROW SETTINGS -"); y+= 20;
        UI::Slider("Thick", &p.eyebrowThickness, 1, 20, y); UI::Slider("Len", &p.eyebrowLength, 0.5f, 2.0f, y);
        UI::Slider("Spacing", &p.eyebrowSpacing, -100.0f, 100.0f, y); UI::Slider("Pos X", &p.eyebrowX, -10, 10, y);
        UI::Slider("Pos Y", &p.eyebrowY, -10, 10, y); UI::Slider("Scale", &p.browScale, 0.5f, 2.0f, y);
        UI::Slider("Angle", &p.browAngle, -45.0f, 45.0f, y); UI::Slider("Bend", &p.browBend, -2.0f, 2.0f, y);
        UI::Slider("Bend Off", &p.browBendOffset, 0.0f, 0.99f, y); UI::Checkbox("Use Lower Brow", &p.useLowerBrow, 10, y); y += 35;
    }
    if (p.showTears) { GuiLabel({UI::START_X+10, y, 200, 20}, "- TEAR SETTINGS -"); y+= 20; UI::Slider("Level", &p.tearsLevel, 0, 1, y); }
    if(p.showBlush) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- BLUSH SETTINGS -"); y+= 20;
        UI::Slider("Scale", &p.blushScale, 0.5f, 3.0f, y); UI::Slider("Pos X", &p.blushX, -10.0f, 10.0f, y);
        UI::Slider("Pos Y", &p.blushY, -10.0f, 10.0f, y); UI::Slider("Space", &p.blushSpacing, -100.0f, 100.0f, y);
        GuiLabel({UI::START_X+10, y, UI::LABEL_WIDTH, 20}, "Blush Mode");
        if(GuiButton({UI::START_X+10+UI::LABEL_WIDTH,  y, 80, 20}, p.blushMode == 0 ? "Pink" : (p.blushMode == 1 ? "Green" : "Yellow"))) p.blushMode = (p.blushMode + 1) % 3;
        y += 35;
    }
    GuiLabel({UI::START_X+10, y, UI::PANEL_WIDTH, 20}, "--- SURFACE FX ---"); y += 20;
    UI::Slider("Stress", &p.stressLevel, 0.0f, 1.0f, y); UI::Slider("Gloom", &p.gloomLevel, 0.0f, 1.0f, y);
    bool distMode = (p.distortMode == 1);
    if (UI::Checkbox("Squash/Stretch", &distMode, 10, y)) p.distortMode = distMode ? 1 : 0;
}

void DrawMouthControls(float& y, MouthParams& p) {
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 480}, "MOUTH SETTINGS"); y += 20.0f;
    UI::Slider("Scale", &p.scale, 0.5f, 10.0f, y);
    UI::Slider("Look X", &p.lookX, -250.0f, 250.0f, y);
    UI::Slider("Look Y", &p.lookY, -250.0f, 250.0f, y);
    UI::Slider("Mouth Angle", &p.mouthAngle, -180.0f, 180.0f, y);
    UI::Slider("Outline", &p.outlineThickness, 1.f, 10.0f, y); y += 10;
    UI::Slider("Open", &p.open, 0.0f, 1.2f, y); 
    UI::Slider("Width", &p.width, 0.1f, 1.5f, y);
    UI::Slider("Curve", &p.curve, -5.0f, 5.0f, y); y += 10;
    UI::Slider("Sqze Top", &p.squeezeTop, -1.0f, 1.0f, y); UI::Slider("Sqze Bot", &p.squeezeBottom, -1.0, 1.0f, y); y += 10;
    UI::Slider("Sqze Sigma", &p.sigma, 0.0f, 1.0f, y); UI::Slider("Sqze Pow", &p.power, 0.0f, 10.0f, y);
    UI::Slider("Sqze Lift", &p.maxLiftValue, 0.0f, 1.0f, y); y += 10;
    UI::Slider("Teeth Y", &p.teethY, -1.0f, 1.0f, y); UI::Slider("Teeth W", &p.teethWidth, 0.1f, 1.0f, y);
    UI::Slider("Teeth Gap",&p.teethGap, 0.0f, 100.0f, y); y += 10;
    UI::Slider("Tongue Up", &p.tongueUp, 0.0f, 1.0f, y); UI::Slider("Tongue W", &p.tongueWidth, 0.3f, 1.0f, y);
    UI::Slider("Tongue X", &p.tongueX, -1.0f, 1.0f, y); y += 10;
    UI::Slider("Asymmetry", &p.asymmetry, -1.0f, 1.0f, y); UI::Slider("Squareness", &p.squareness, 0.0f, 1.0f, y);y+=10;
    UI::Slider("Stress Lns", &p.stressLines, 0.0f, 1.0f, y);y+=10;
    UI::Checkbox("Show Inner Mouth", &p.showInnerMouth, 10, y); y+=20;
    UI::Checkbox("3 Shape", &p.isThreeShape, 10, y);y+=20;
    UI::Checkbox("D Shape", &p.isDShape, 10, y);
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 950, "BMO Face Poser: Final");
    SetTargetFPS(60);

    // Style Setup
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt(BLACK));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
    GuiSetStyle(SLIDER, BASE_COLOR_NORMAL, ColorToInt({ 60, 60, 60, 255 }));
    GuiSetStyle(SLIDER, BASE_COLOR_FOCUSED, ColorToInt({ 120, 180, 255, 255 }));
    GuiSetStyle(DROPDOWNBOX, BASE_COLOR_NORMAL, ColorToInt({ 40, 40, 40, 255 }));
    GuiSetStyle(DROPDOWNBOX, TEXT_COLOR_NORMAL, ColorToInt(WHITE));

    // Initialize Engine
    FaceSystem engine;
    engine.Init();
    
    // Load Assets
    ReferenceAtlas atlas;
    atlas.Load("assets/BMO_SpriteSheet_Texture.png", "assets/BMO_SpriteSheet_Data.json");

    FaceDatabase db;
    // Attempt to load unified database, fallbacks handled by struct defaults
    db.Load("face_database.txt");

    EditorState state;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        // Assume GlobalScaler is available from utility.h as per environment
        GlobalScaler.Update();

        // Calculate positions
        Vector2 center = { (float)GetScreenWidth() * 0.5f, (float)GetScreenHeight() * 0.5f };
        float mouthOffset = 100.0f * GlobalScaler.scale;
        // Mouth is offset slightly lower than eyes
        Vector2 mouthPos = { center.x, center.y + mouthOffset };

        // -------------------------------------------------
        // LOGIC
        // -------------------------------------------------
        engine.usePhysics = state.usePhysics;
        engine.debugBoxes = state.debugBoxes;
        
        // Push state to engine
        engine.Update(dt, state.current.eyes, state.current.mouth);

        // -------------------------------------------------
        // DRAWING
        // -------------------------------------------------
        BeginDrawing();
        ClearBackground({201, 228, 195, 255}); // BMO Green

        // 1. Reference Layer (Background)
        if (state.showReference) {
            if (!atlas.faceNames.empty()) 
                atlas.Draw(atlas.faceNames[state.faceRefIdx], center, 1.0f, state.refOpacity);
        }

        // 2. Procedural Face Layer (Midground)
        // Note: Engine draws internally to a texture then presents it
        engine.Draw(center, mouthPos, state.current.eyes, state.current.mouth, BLACK);

        // 3. UI Layer (Foreground)
        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();
        
        // Sprite Navigation Shortcuts
        if (IsKeyPressed(KEY_RIGHT)) {
            if (state.tabIndex == 0) state.CycleFace(atlas, 1);
        }
        if (IsKeyPressed(KEY_LEFT)) {
            if (state.tabIndex == 0) state.CycleFace(atlas, -1);
        }

        if (state.enableGUI) {
            float y = UI::START_Y;

            // Tab Bar
            GuiToggleGroup({UI::START_X, 20, 120, 30}, "EYES;MOUTH", &state.tabIndex);

            // -- LEFT COLUMN: REFERENCE --
            GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 110}, "SPRITE REFERENCE");
            if (GuiButton({UI::START_X + 10, y + 30, 40, 30}, "<")) {
                if (state.tabIndex == 0) state.CycleFace(atlas, -1);
            }
            if (GuiButton({UI::START_X + 270, y + 30, 40, 30}, ">")) {
                if (state.tabIndex == 0) state.CycleFace(atlas, 1);
            }

            std::string refLabel = atlas.faceNames.empty() ? "NONE" : atlas.faceNames[state.faceRefIdx];
			GuiLabel({UI::START_X + 60, y + 30, 200, 30}, refLabel.c_str());
            
            GuiLabel({UI::START_X + 60, y + 30, 200, 30}, refLabel.c_str());

            if (GuiButton({UI::START_X + 10, y + 70, UI::PANEL_WIDTH - 20, 30}, "SAVE PRESET (Enter)")) {
                db.Save("face_database.txt", refLabel, state.current);
            }
            y += 120.0f;

            // -- LEFT COLUMN: PARAMETER CONTROLS --
            if (state.tabIndex == 0) {
                DrawEyeControls(y, state.current.eyes);
            } else {
                DrawMouthControls(y, state.current.mouth);
            }

            // -- RIGHT COLUMN: DATABASE & GLOBAL --
            float screenW = (float)GetScreenWidth();
            float screenH = (float)GetScreenHeight();
            float panelW = 250.0f;
            float vx = screenW - panelW - 16;
            float vy = 16;

            GuiGroupBox({ vx, vy, panelW, 120 }, "DATABASE LOAD");
            if (GuiDropdownBox({ vx + 10, vy + 30, panelW - 20, 25 }, db.dropdownStr.c_str(), &state.dropdownActive, state.dropdownEditMode)) {
                state.dropdownEditMode = !state.dropdownEditMode;
            }
            
            if (GuiButton({ vx + 10, vy + 65, 160, 30 }, "LOAD SELECTED")) {
                if (!db.entries.empty() && state.dropdownActive < (int)db.entries.size()) {
                    state.current = db.entries[state.dropdownActive].state;
                }
            }
            if (GuiButton({ vx + panelW - 60, vy + 65, 50, 30 }, "RLD")) db.Load("face_database.txt");

            // Viewport Settings
            float vyView = vy + 120 + 12.0f;
            GuiGroupBox({ vx, vyView, panelW, 110 }, "VIEWPORT");
            GuiCheckBox({ vx + 10, vyView + 25, 20, 20 }, "Show Ref", &state.showReference);
            GuiSliderBar({ vx + 80, vyView + 55, 100, 20 }, "Opac", nullptr, &state.refOpacity, 0.0f, 1.0f);
            GuiCheckBox({ vx + 10, vyView + 80, 20, 20 }, "Test Physics", &state.usePhysics);

            // Bottom Right Toggles
            GuiCheckBox({screenW - 180, screenH - 40, 20, 20}, "Debug Boxes", &state.debugBoxes);
            GuiCheckBox({screenW - 180, screenH - 70, 20, 20}, "Enable GUI", &state.enableGUI);
        }
        else {
             // Minimal UI to re-enable
             GuiCheckBox({(float)GetScreenWidth() - 180, (float)GetScreenHeight() - 70, 20, 20}, "Enable GUI", &state.enableGUI);
        }

        // Global Save Shortcut
        if (IsKeyPressed(KEY_ENTER)) {
             std::string name = "custom";
             if (state.tabIndex == 0 && !atlas.faceNames.empty()) name = atlas.faceNames[state.faceRefIdx];
             db.Save("face_database.txt", name, state.current);
        }

        EndDrawing();
    }

    engine.Unload();
    CloseWindow();
    return 0;
}