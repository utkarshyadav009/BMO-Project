// FaceData.h
#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm> // For std::sort
// --------------------------------------------------------
// PARAMETERS
// --------------------------------------------------------
struct EyeParams {
    // Eye Shape
    float eyeShapeID = 0.0f; float bend = 0.0f; float eyeThickness = 4.0f;
    float eyeSide = 0.0f; float scaleX = 1.0f; float scaleY = 1.0f;
    float angle = 0.0f; float spacing = 612.0f; float squareness = 0.0f;

    // FX
    float stressLevel = 0.0f; float gloomLevel = 0.0f;
    float spiralSpeed = -1.2f;

    // Look
    float lookX = 0.0f; float lookY = 62.50f;

    // Brow
    bool showBrow = false; bool useLowerBrow = false;
    float eyebrowThickness = 4.0f; float eyebrowLength = 1.0f;
    float eyebrowSpacing = 0.0f; float eyebrowX = 0.0f; float eyebrowY = 0.0f;
    float browScale = 1.0f; float browSide = 1.0f; float browAngle = 0.0f;
    float browBend = 0.0f; float browBendOffset = 0.85f;

    // Tears/Blush
    bool showTears = false; bool showBlush = false;
    float tearsLevel = 0.0f; int blushMode = 0; float blushScale = 1.0f;
    float blushX = 0.0f; float blushY = 0.0f; float blushSpacing = 0.0f;

    // Global
    float pixelation = 1.0f; 
};

struct MouthParams {
    float open = 0.05f; float width = 0.5f; float curve = 0.0f; float mouthAngle = 0.0f; 
    float squeezeTop = 0.0f; float squeezeBottom = 0.0f; 
    float teethY = 0.0f; float tongueUp = 0.0f; float tongueX = 0.0f;
    float tongueWidth = 0.65f; float asymmetry = 0.0f; float squareness = 0.0f;
    float teethWidth = 0.50f; float teethGap = 45.0f; float scale = 1.0f; float outlineThickness = 1.5f;
    float sigma = 0.45f; float power = 6.0f; float maxLiftValue = 0.55f;
    float lookX = 0.0f; float lookY = 0.0f; float stressLines = 0.0f; bool showInnerMouth =true; bool isThreeShape; bool isDShape; bool isSlashShape;
};

struct FaceState {
    EyeParams eyes;
    MouthParams mouth;
    
    FaceState() {
        eyes = EyeParams(); 
        eyes.scaleX = 1.0f; eyes.scaleY = 1.0f; eyes.spacing = 200.0f;
        mouth = MouthParams(); 
    }

    void reset() {
        eyes = EyeParams(); 
        eyes.scaleX = 1.0f; eyes.scaleY = 1.0f; eyes.spacing = 200.0f;
        mouth = MouthParams(); 
    }
};

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

    inline void Load(const char* filename) {
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
                    // if (v.size() >= 62) {
                    //     if (v.size() > idx) mp.mouthAngle = v[idx++]; 
                    // } else {
                    //     mp.mouthAngle = 0.0f; // Default for old database entries
                    // }
                    if (v.size() > idx) mp.mouthAngle = v[idx++];
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
                    if (v.size() > idx) mp.isSlashShape = (bool)v[idx++];


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

    inline void Save(const char* filename, std::string name, const FaceState& s) {
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
           << p.stressLevel << "f, " << p.gloomLevel << "f, " << p.spiralSpeed << "f, "
           << p.lookX << "f, " << p.lookY << "f, "
           << (int)p.showBrow << ", " << (int)p.useLowerBrow << ", " << p.eyebrowThickness << "f, " << p.eyebrowLength << "f, "
           << p.eyebrowSpacing << "f, " << p.eyebrowX << "f, " << p.eyebrowY << "f, " << p.browScale << "f, " << p.browSide << "f, " 
           << p.browAngle << "f, " << p.browBend << "f, " << p.browBendOffset << "f, "
           << (int)p.showTears << ", " << (int)p.showBlush << ", " << p.tearsLevel << "f, " << p.blushMode << ", "
           << p.blushScale << "f, " << p.blushX << "f, " << p.blushY << "f, " << p.blushSpacing << "f, " 
           << p.pixelation << "f, ";

        // Mouth
        const MouthParams& m = s.mouth;
        ss << m.open << "f, " << m.width << "f, " << m.curve << "f, " << m.mouthAngle << "f, " << m.squeezeTop << "f, " << m.squeezeBottom << "f, "
           << m.teethY << "f, " << m.tongueUp << "f, " << m.tongueX << "f, " << m.tongueWidth << "f, "
           << m.asymmetry << "f, " << m.squareness << "f, " << m.teethWidth << "f, " << m.teethGap << "f, "
           << m.scale << "f, " << m.outlineThickness << "f, " << m.sigma << "f, " << m.power << "f, " << m.maxLiftValue << "f, " 
           << m.lookX << "f, " << m.lookY << "f, "<< m.stressLines << "f, " << (int)m.showInnerMouth << "f, "<< (int)m.isThreeShape << "f, "<< (int)m.isDShape << "f, "
           << (int)m.isSlashShape <<" };";

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
