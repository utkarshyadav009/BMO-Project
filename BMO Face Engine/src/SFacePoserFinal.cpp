// SFacePoser.cpp
// UNIFIED DRIVER FILE: Links GUI to Shader-Based Eye and Mouth Engines
// USAGE: Compile this to create the Face Editor tool.

#include "raylib.h"
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

// Include the Refactored Unified Engine
#include "ShaderParametricFace.cpp" 

using json = nlohmann::json;

// ---------------------------------------------------------
// UI CONSTANTS & HELPERS
// ---------------------------------------------------------
namespace UI {
    const float START_X = 20.0f;
    const float START_Y = 50.0f; // Shifted down for tabs
    const float PANEL_WIDTH = 340.0f;
    const float LABEL_WIDTH = 90.0f;
    const float VAL_WIDTH = 40.0f;
    const float ROW_HEIGHT = 25.0f;
    
    float GetSliderWidth() { return PANEL_WIDTH - LABEL_WIDTH - VAL_WIDTH - 30.0f; }

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
// DATA MANAGERS
// ---------------------------------------------------------
struct ReferenceAtlas {
    Texture2D texture;
    std::unordered_map<std::string, Rectangle> frames;
    std::vector<std::string> eyeNames; 
    std::vector<std::string> mouthNames;

    void Load(const char* img, const char* data) {
        texture = LoadTexture(img);
        if (texture.id == 0) return;
        std::ifstream f(data);
        if (!f.good()) return;
        try {
            json j = json::parse(f);
            if (j.contains("textures")) {
                for (auto& t : j["textures"]) {
                    for (auto& fr : t["frames"]) {
                        std::string name = fr["filename"];
                        std::string lower = name;
                        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                        Rectangle r = { (float)fr["frame"]["x"], (float)fr["frame"]["y"], (float)fr["frame"]["w"], (float)fr["frame"]["h"] };
                        frames[name] = r;

                        if (lower.find("eye") != std::string::npos) eyeNames.push_back(name);
                        if (lower.find("mouth") != std::string::npos || lower.find("phoneme") != std::string::npos) mouthNames.push_back(name);
                    }
                }
            }
            std::sort(eyeNames.begin(), eyeNames.end());
            std::sort(mouthNames.begin(), mouthNames.end());
        } catch (...) { std::cerr << "[Atlas] Parse Error\n"; }
    }

    void Draw(std::string name, Vector2 pos, float scale, float alpha) {
        if (frames.find(name) == frames.end()) return;
        Rectangle src = frames[name];
        Rectangle dest = { pos.x, pos.y, src.width * scale, src.height * scale };
        Vector2 origin = { dest.width/2, dest.height/2 };
        DrawTexturePro(texture, src, dest, origin, 0.0f, Fade(WHITE, alpha));
    }
};

struct EyeDatabase {
    struct Entry { std::string name; Eyes::EyeParams params; };
    std::vector<Entry> entries;
    std::string dropdownStr;

    void Load(const char* filename) {
        entries.clear(); dropdownStr = "";
        std::ifstream in(filename); if (!in.is_open()) return;
        std::string line;
        while (std::getline(in, line)) {
            size_t namePos = line.find("eyes[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                if (endPos != std::string::npos) {
                    Entry e; e.params = Eyes::EyeParams();
                    e.name = line.substr(namePos + 6, endPos - (namePos + 6));
                    size_t bracePos = line.find("{");
                    if (bracePos != std::string::npos) {
                        std::string valStr = line.substr(bracePos + 1);
                        std::replace(valStr.begin(), valStr.end(), '}', ' '); 
                        std::replace(valStr.begin(), valStr.end(), ',', ' ');
                        std::replace(valStr.begin(), valStr.end(), 'f', ' ');
                        std::stringstream ss(valStr);
                        int tShowBrow, tUseLowerBrow, tShowTears, tShowBlush, tDistortMode, tBlushMode;
                        ss >> e.params.eyeShapeID >> e.params.bend >> e.params.eyeThickness >> e.params.eyeSide >> e.params.scaleX >> e.params.scaleY
                           >> e.params.angle >> e.params.spacing >> e.params.squareness
                           >> e.params.stressLevel >> e.params.gloomLevel >> tDistortMode >> e.params.spiralSpeed
                           >> e.params.lookX >> e.params.lookY
                           >> tShowBrow >> tUseLowerBrow >> e.params.eyebrowThickness >> e.params.eyebrowLength >> e.params.eyebrowSpacing
                           >> e.params.eyebrowX >> e.params.eyebrowY >> e.params.browScale >> e.params.browSide >> e.params.browAngle 
                           >> e.params.browBend >> e.params.browBendOffset
                           >> tShowTears >> tShowBlush >> e.params.tearsLevel >> tBlushMode
                           >> e.params.blushScale >> e.params.blushX >> e.params.blushY >> e.params.blushSpacing >> e.params.pixelation;
                        e.params.distortMode = tDistortMode; e.params.showBrow = (bool)tShowBrow;
                        e.params.useLowerBrow = (bool)tUseLowerBrow; e.params.showTears = (bool)tShowTears;
                        e.params.showBlush = (bool)tShowBlush; e.params.blushMode = tBlushMode;
                        if (e.params.scaleX == 0) e.params.scaleX = 1.0f; if (e.params.scaleY == 0) e.params.scaleY = 1.0f;
                        entries.push_back(e);
                    }
                }
            }
        }
        for (size_t i = 0; i < entries.size(); i++) { dropdownStr += entries[i].name; if (i < entries.size() - 1) dropdownStr += ";"; }
    }

    void Save(const char* filename, std::string name, Eyes::EyeParams p) {
        std::ifstream infile(filename); std::vector<std::string> lines; std::string line; bool found = false;
        std::stringstream newLine;
        newLine << "eyes[\"" << name << "\"] = { "
                << p.eyeShapeID << "f, " << p.bend << "f, " << p.eyeThickness << "f, " << p.eyeSide << "f, "
                << p.scaleX << "f, " << p.scaleY << "f, " << p.angle << "f, " << p.spacing << "f, " << p.squareness << "f, "
                << p.stressLevel << "f, " << p.gloomLevel << "f, " << p.distortMode << ", " << p.spiralSpeed << "f, "
                << p.lookX << "f, " << p.lookY << "f, "
                << (int)p.showBrow << ", " << (int)p.useLowerBrow << ", " << p.eyebrowThickness << "f, " << p.eyebrowLength << "f, "
                << p.eyebrowSpacing << "f, " << p.eyebrowX << "f, " << p.eyebrowY << "f, " << p.browScale << "f, " << p.browSide << "f, " 
                << p.browAngle << "f, " << p.browBend << "f, " << p.browBendOffset << "f, "
                << (int)p.showTears << ", " << (int)p.showBlush << ", " << p.tearsLevel << "f, " << p.blushMode << ", "
                << p.blushScale << "f, " << p.blushX << "f, " << p.blushY << "f, " << p.blushSpacing << "f, " << p.pixelation << "f };"; 
        if (infile.is_open()) {
            while (std::getline(infile, line)) {
                if (line.find("eyes[\"" + name + "\"]") != std::string::npos) { lines.push_back(newLine.str()); found = true; } 
                else lines.push_back(line);
            }
            infile.close();
        }
        if (!found) lines.push_back(newLine.str());
        std::ofstream outfile(filename); for (const auto& l : lines) outfile << l << "\n";
        Load(filename);
    }
};

struct MouthDatabase {
    struct Entry { std::string name; Mouth::MouthParams params; };
    std::vector<Entry> entries;
    std::string dropdownStr;

    void Load(const char* filename) {
        entries.clear(); dropdownStr = "";
        std::ifstream in(filename); if (!in.is_open()) return;
        std::string line; Entry cur; bool parsing = false;
        while (std::getline(in, line)) {
            size_t namePos = line.find("visemes[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                cur.name = line.substr(namePos + 9, endPos - (namePos + 9));
                cur.params = Mouth::MakeDefaultParams(); parsing = true;
            }
            if (parsing && line.find("mouth") != std::string::npos && line.find("{") != std::string::npos) {
                std::vector<float> v; std::string num;
                for (char c : line) {
                    if (isdigit(c) || c=='.' || c=='-' || c=='e') num+=c;
                    else if (!num.empty()) { try{v.push_back(std::stof(num));}catch(...){} num.clear(); }
                }
                if (!num.empty()) try{v.push_back(std::stof(num));}catch(...){}
                if(v.size()>0) cur.params.open = v[0]; if(v.size()>1) cur.params.width = v[1];
                if(v.size()>2) cur.params.curve = v[2]; if(v.size()>3) cur.params.squeezeTop = v[3];
                if(v.size()>4) cur.params.squeezeBottom = v[4]; if(v.size()>5) cur.params.teethY = v[5];
                if(v.size()>6) cur.params.tongueUp = v[6]; if(v.size()>7) cur.params.tongueX = v[7];
                if(v.size()>8) cur.params.tongueWidth = v[8]; if(v.size()>9) cur.params.asymmetry = v[9];
                if(v.size()>10) cur.params.squareness = v[10]; if(v.size()>11) cur.params.teethWidth = v[11];
                if(v.size()>12) cur.params.teethGap = v[12]; if(v.size()>13) cur.params.scale = v[13];
                if(v.size()>14) cur.params.outlineThickness = v[14]; if(v.size()>15) cur.params.sigma = v[15];
                if(v.size()>16) cur.params.power = v[16]; if(v.size()>17) cur.params.maxLiftValue = v[17];
            }
            if (parsing && line.find("};") != std::string::npos) { entries.push_back(cur); parsing = false; }
        }
        for (size_t i = 0; i < entries.size(); i++) { dropdownStr += entries[i].name; if (i < entries.size() - 1) dropdownStr += ";"; }
    }

    void Save(std::string name, Mouth::MouthParams p) {
        std::ifstream infile("visemes_database.txt"); std::ofstream tempfile("visemes_database.tmp");
        if (!infile.is_open() || !tempfile.is_open()) return;
        bool found = false; std::string line;
        while (std::getline(infile, line)) {
            size_t namePos = line.find("visemes[\"");
            if (namePos != std::string::npos) {
                size_t endPos = line.find("\"]");
                if (line.substr(namePos + 9, endPos - (namePos + 9)) == name) {
                    tempfile << "visemes[\"" << name << "\"] = { "
                        << p.open << "f, " << p.width << "f, " << p.curve << "f, " << p.squeezeTop << "f, " << p.squeezeBottom << "f, "
                        << p.teethY << "f, " << p.tongueUp << "f, " << p.tongueX << "f, " << p.tongueWidth << "f, "
                        << p.asymmetry << "f, " << p.squareness << "f, " << p.teethWidth << "f, " << p.teethGap << "f, "
                        << p.scale << "f, " << p.outlineThickness << "f, " << p.sigma << "f, " << p.power << "f, " << p.maxLiftValue << "f };\n";
                    found = true; continue;
                }
            }
            tempfile << line << "\n";
        }
        if (!found) {
            tempfile << "visemes[\"" << name << "\"] = { "
                << p.open << "f, " << p.width << "f, " << p.curve << "f, " << p.squeezeTop << "f, " << p.squeezeBottom << "f, "
                << p.teethY << "f, " << p.tongueUp << "f, " << p.tongueX << "f, " << p.tongueWidth << "f, "
                << p.asymmetry << "f, " << p.squareness << "f, " << p.teethWidth << "f, " << p.teethGap << "f, "
                << p.scale << "f, " << p.outlineThickness << "f, " << p.sigma << "f, " << p.power << "f, " << p.maxLiftValue << "f };\n";
        }
        infile.close(); tempfile.close();
        std::remove("visemes_database.txt"); std::rename("visemes_database.tmp", "visemes_database.txt");
        Load("visemes_database.txt");
    }
};

// ---------------------------------------------------------
// EDITOR STATE & GUI
// ---------------------------------------------------------
struct EditorState {
    int mode = 0; // 0=Eye, 1=Mouth
    
    // Eye State
    Eyes::EyeParams eyeParams;
    int eyeRefIdx = 0;
    
    // Mouth State
    Mouth::MouthParams mouthParams;
    int mouthRefIdx = 0;

    // Shared State
    bool showReference = true;
    float refOpacity = 0.5f;
    bool usePhysics = false;
    bool enableGUI = true;
    bool debugBoxes = false;
    
    int ddEyeActive = 0, ddMouthActive = 0;
    bool ddEyeEdit = false, ddMouthEdit = false;
};

void DrawEyeControls(float& y, EditorState& s) {
    Eyes::EyeParams& p = s.eyeParams;
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 350}, "MAIN EYE SETTINGS"); y += 20.0f;
    int shapeInt = (int)p.eyeShapeID;
    UI::Slider("Shape ID", &p.eyeShapeID, 0.0f, 12.0f, y);
    const char* shapes[] = { "Dot", "Line", "Arc", "Cross", "Star", "Heart", "Spiral", "Chevron", "Shuriken", "Kawaii", "Shocked", "Teary", "Colon Eyes" };
    if(shapeInt >= 0 && shapeInt <= 12) GuiLabel({UI::START_X+10+UI::LABEL_WIDTH, y-25, UI::GetSliderWidth(), 20}, shapes[shapeInt]);
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

    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 480}, "ELEMENTS & FX"); y += 20.0f;
    UI::Checkbox("Brows", &p.showBrow, 10, y); UI::Checkbox("Tears", &p.showTears, 90, y); UI::Checkbox("Blush", &p.showBlush, 170, y); y += 30.0f;
    if (p.showBrow) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- BROW SETTINGS -"); y+= 20;
        UI::Slider("Thick", &p.eyebrowThickness, 1, 20, y); UI::Slider("Len", &p.eyebrowLength, 0.5f, 2.0f, y);
        UI::Slider("Spacing", &p.eyebrowSpacing, -100.0f, 100.0f, y); UI::Slider("Pos X", &p.eyebrowX, -10, 10, y);
        UI::Slider("Pos Y", &p.eyebrowY, -10, 10, y); UI::Slider("Scale", &p.browScale, 0.5f, 2.0f, y);
        UI::Slider("Angle", &p.browAngle, -45.0f, 45.0f, y); UI::Slider("Bend", &p.browBend, -2.0f, 2.0f, y);
        UI::Slider("Bend Off", &p.browBendOffset, 0.0f, 0.99f, y); UI::Checkbox("Use Lower Brow", &p.useLowerBrow, 10, y); y += 35;
    }
    if (p.showTears) {
        GuiLabel({UI::START_X+10, y, 200, 20}, "- TEAR SETTINGS -"); y+= 20; UI::Slider("Level", &p.tearsLevel, 0, 1, y);
    }
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
    if (UI::Checkbox("Squash/Stretch Distortion", &distMode, 10, y)) p.distortMode = distMode ? 1 : 0;
}

void DrawMouthControls(float& y, EditorState& s) {
    Mouth::MouthParams& p = s.mouthParams;
    GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 450}, "MOUTH PARAMETERS"); y += 20.0f;
    UI::Slider("Scale", &p.scale, 0.5f, 4.0f, y);
    UI::Slider("Outline", &p.outlineThickness, 1.f, 4.0f, y); y += 10;
    UI::Slider("Open", &p.open, 0.0f, 1.2f, y); UI::Slider("Width", &p.width, 0.1f, 1.5f, y);
    UI::Slider("Curve", &p.curve, -2.0f, 2.0f, y); y += 10;
    UI::Slider("Sqze Top", &p.squeezeTop, 0.0f, 1.0f, y); UI::Slider("Sqze Bot", &p.squeezeBottom, 0.0f, 1.0f, y); y += 10;
    UI::Slider("Sqze Sigma", &p.sigma, 0.0f, 1.0f, y); UI::Slider("Sqze Pow", &p.power, 0.0f, 10.0f, y);
    UI::Slider("Sqze Lift", &p.maxLiftValue, 0.0f, 1.0f, y); y += 10;
    UI::Slider("Teeth Y", &p.teethY, -1.0f, 1.0f, y); UI::Slider("Teeth W", &p.teethWidth, 0.1f, 1.0f, y);
    UI::Slider("Teeth Gap",&p.teethGap, 0.0f, 100.0f, y); y += 10;
    UI::Slider("Tongue Up", &p.tongueUp, 0.0f, 1.0f, y); UI::Slider("Tongue W", &p.tongueWidth, 0.3f, 1.0f, y);
    UI::Slider("Tongue X", &p.tongueX, -1.0f, 1.0f, y); y += 10;
    UI::Slider("Asymmetry", &p.asymmetry, -1.0f, 1.0f, y); UI::Slider("Squareness", &p.squareness, 0.0f, 1.0f, y);
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 900, "Face Engine: Unified");
    SetTargetFPS(60);

    // Common Style
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt(BLACK));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
    GuiSetStyle(SLIDER, BASE_COLOR_NORMAL, ColorToInt({ 60, 60, 60, 255 }));
    GuiSetStyle(SLIDER, BASE_COLOR_FOCUSED, ColorToInt({ 120, 180, 255, 255 }));
    GuiSetStyle(DROPDOWNBOX, BASE_COLOR_NORMAL, ColorToInt({ 40, 40, 40, 255 }));
    GuiSetStyle(DROPDOWNBOX, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
    
    // Systems
    Eyes::ParametricEyes eyeEng; eyeEng.Init();
    Mouth::ParametricMouth mouthEng; mouthEng.Init({800, 360});
    
    ReferenceAtlas atlas;
    atlas.Load("assets/BMO_Animation_LipSyncSprite.png", "assets/BMO_Animation_Lipsync.json");
    
    EyeDatabase dbEye; dbEye.Load("eyes_database.txt");
    MouthDatabase dbMouth; dbMouth.Load("visemes_database.txt");
    
    EditorState state;
    state.eyeParams.scaleX = 1.0f; state.eyeParams.scaleY = 1.0f;
    state.mouthParams = Mouth::MakeDefaultParams();

    // Toggle Panel Button state
    int activeTab = 0; 
    bool toggleEdit = false;

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        if(GlobalScaler.scale == 0) GlobalScaler.Update(); // Ensure scale
        else GlobalScaler.Update();

        Vector2 centerScreen = { (float)GetScreenWidth() * 0.5f, (float)GetScreenHeight() * 0.5f };

        // --- UPDATE ---
        eyeEng.usePhysics = state.usePhysics;
        eyeEng.debugBoxes = state.debugBoxes;
        eyeEng.Update(dt, state.eyeParams);

        mouthEng.usePhysics = state.usePhysics;
        mouthEng.centerPos = {centerScreen.x, centerScreen.y + 100.0f}; // Position mouth below center
        // Mouth sync
        if (state.usePhysics) {
             mouthEng.target = state.mouthParams;
             mouthEng.UpdatePhysics(dt);
             mouthEng.GenerateGeometry();
        } else {
             mouthEng.target = state.mouthParams;
             mouthEng.current = state.mouthParams; // Force set
             mouthEng.GenerateGeometry();
        }

        // --- DRAW ---
        BeginDrawing();
        ClearBackground({201, 228, 195, 255}); // BMO Face

        // Reference
        if (state.showReference) {
            if (activeTab == 0 && !atlas.eyeNames.empty()) 
                atlas.Draw(atlas.eyeNames[state.eyeRefIdx], centerScreen, 1.0f, state.refOpacity);
            else if (activeTab == 1 && !atlas.mouthNames.empty()) 
                atlas.Draw(atlas.mouthNames[state.mouthRefIdx], mouthEng.centerPos, 1.0f, state.refOpacity);
        }

        // Engine
        Eyes::EyeParams renderEyeP = state.eyeParams;
        if(state.usePhysics) {
            renderEyeP.scaleX = eyeEng.sScaleX.val; renderEyeP.scaleY = eyeEng.sScaleY.val;
            renderEyeP.lookX = eyeEng.sLookX.val; renderEyeP.lookY = eyeEng.sLookY.val;
        }
        eyeEng.Draw(centerScreen, renderEyeP, BLACK);
        mouthEng.Draw();

        // --- GUI ---
        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();
        
        // Mode Tabs
        GuiToggleGroup({UI::START_X, 10, 120, 30}, "EYES;MOUTH", &activeTab);
        state.mode = activeTab;

        if (state.enableGUI) {
            float y = UI::START_Y;
            
            // Reference Controls
            GuiGroupBox({UI::START_X, y, UI::PANEL_WIDTH, 110}, "SPRITE REFERENCE");
            if (GuiButton({UI::START_X + 10, y + 30, 40, 30}, "<")) {
                if(activeTab==0) { state.eyeRefIdx--; if(state.eyeRefIdx<0) state.eyeRefIdx=(int)atlas.eyeNames.size()-1; }
                else { state.mouthRefIdx--; if(state.mouthRefIdx<0) state.mouthRefIdx=(int)atlas.mouthNames.size()-1; }
            }
            if (GuiButton({UI::START_X + 270, y + 30, 40, 30}, ">")) {
                if(activeTab==0) { state.eyeRefIdx++; if(state.eyeRefIdx>=(int)atlas.eyeNames.size()) state.eyeRefIdx=0; }
                else { state.mouthRefIdx++; if(state.mouthRefIdx>=(int)atlas.mouthNames.size()) state.mouthRefIdx=0; }
            }
            std::string lbl;
            if (activeTab==0) lbl = atlas.eyeNames.empty() ? "NONE" : atlas.eyeNames[state.eyeRefIdx];
            else lbl = atlas.mouthNames.empty() ? "NONE" : atlas.mouthNames[state.mouthRefIdx];
            GuiLabel({UI::START_X + 60, y + 30, 200, 30}, lbl.c_str());
            
            if (GuiButton({UI::START_X + 10, y + 70, UI::PANEL_WIDTH - 20, 30}, "SAVE PRESET (Enter)")) {
                 if (activeTab==0) dbEye.Save("eyes_database.txt", lbl, state.eyeParams);
                 else dbMouth.Save(lbl, state.mouthParams);
            }
            y += 120.0f;

            // Param Controls
            if (activeTab == 0) DrawEyeControls(y, state);
            else DrawMouthControls(y, state);

            // Database & Viewport
            float screenW = (float)GetScreenWidth();
            float panelW = 250.0f;
            float vx = screenW - panelW - 16; float vy = 16;
            
            GuiGroupBox({ vx, vy, panelW, 120 }, "DATABASE LOAD");
            if (activeTab == 0) {
                if (GuiDropdownBox({ vx + 10, vy + 30, panelW - 20, 25 }, dbEye.dropdownStr.c_str(), &state.ddEyeActive, state.ddEyeEdit)) state.ddEyeEdit = !state.ddEyeEdit;
                if (GuiButton({ vx + 10, vy + 65, 160, 30 }, "LOAD SELECTED")) {
                    if (state.ddEyeActive < (int)dbEye.entries.size()) state.eyeParams = dbEye.entries[state.ddEyeActive].params;
                }
                if (GuiButton({ vx + panelW - 60, vy + 65, 50, 30 }, "RLD")) dbEye.Load("eyes_database.txt");
            } else {
                if (GuiDropdownBox({ vx + 10, vy + 30, panelW - 20, 25 }, dbMouth.dropdownStr.c_str(), &state.ddMouthActive, state.ddMouthEdit)) state.ddMouthEdit = !state.ddMouthEdit;
                if (GuiButton({ vx + 10, vy + 65, 160, 30 }, "LOAD SELECTED")) {
                    if (state.ddMouthActive < (int)dbMouth.entries.size()) state.mouthParams = dbMouth.entries[state.ddMouthActive].params;
                }
                if (GuiButton({ vx + panelW - 60, vy + 65, 50, 30 }, "RLD")) dbMouth.Load("visemes_database.txt");
            }

            // Viewport Box
            float vyView = vy + 120 + 12.0f;
            GuiGroupBox({ vx, vyView, panelW, 110 }, "VIEWPORT");
            GuiCheckBox({ vx + 10, vyView + 25, 20, 20 }, "Show Ref", &state.showReference);
            GuiSliderBar({ vx + 80, vyView + 55, 100, 20 }, "Opac", nullptr, &state.refOpacity, 0.0f, 1.0f);
            GuiCheckBox({ vx + 10, vyView + 80, 20, 20 }, "Test Physics", &state.usePhysics);
            
            // Toggles bottom right
            GuiCheckBox({screenW-180, (float)GetScreenHeight()-40, 20, 20}, "Debug Boxes", &state.debugBoxes);
            GuiCheckBox({screenW-180, (float)GetScreenHeight()-70, 20, 20}, "Enable GUI", &state.enableGUI);
        } else {
             GuiCheckBox({(float)GetScreenWidth()-180, (float)GetScreenHeight()-70, 20, 20}, "Enable GUI", &state.enableGUI);
        }

        // Key shortcuts
        if (IsKeyPressed(KEY_LEFT)) {
             if(activeTab==0) { state.eyeRefIdx--; if(state.eyeRefIdx<0) state.eyeRefIdx=(int)atlas.eyeNames.size()-1; }
             else { state.mouthRefIdx--; if(state.mouthRefIdx<0) state.mouthRefIdx=(int)atlas.mouthNames.size()-1; }
        }
        if (IsKeyPressed(KEY_RIGHT)) {
             if(activeTab==0) { state.eyeRefIdx++; if(state.eyeRefIdx>=(int)atlas.eyeNames.size()) state.eyeRefIdx=0; }
             else { state.mouthRefIdx++; if(state.mouthRefIdx>=(int)atlas.mouthNames.size()) state.mouthRefIdx=0; }
        }
        if (IsKeyPressed(KEY_ENTER)) {
             std::string lbl = (activeTab==0) ? (atlas.eyeNames.empty()?"":atlas.eyeNames[state.eyeRefIdx]) : (atlas.mouthNames.empty()?"":atlas.mouthNames[state.mouthRefIdx]);
             if(!lbl.empty()) {
                if (activeTab==0) dbEye.Save("eyes_database.txt", lbl, state.eyeParams);
                else dbMouth.Save(lbl, state.mouthParams);
             }
        }

        EndDrawing();
    }
    eyeEng.Unload(); mouthEng.Unload();
    CloseWindow();
    return 0;
}