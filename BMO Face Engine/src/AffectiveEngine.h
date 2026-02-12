// AffectiveEngine.h
#pragma once
#include "raylib.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring> // For memset
#include <fstream>
#include <iomanip> // For std::fixed, std::setprecision

// INCLUDE THE SHARED DATA
#include "FaceData.h"

// 1. THE 5D APPRAISAL VECTOR
struct AppraisalVector {
    float valence;      
    float arousal;      
    float control;      
    float novelty;      
    float obstruct;     

    float DistSq(const AppraisalVector& other) const {
        float dV = valence - other.valence;
        float dA = arousal - other.arousal;
        float dC = control - other.control;
        float dN = novelty - other.novelty;
        float dO = obstruct - other.obstruct;
        return dV*dV + dA*dA + dC*dC + dN*dN + dO*dO;
    }
};

// ---------------------------------------------------------
// TEMPORAL DYNAMICS (The Physics of Mood)
// ---------------------------------------------------------
struct AffectiveState {
    AppraisalVector current;  // Where we are now (The physical face)
    AppraisalVector target;   // Where the LLM wants us to be
    AppraisalVector velocity; // For spring physics
    
    // Per-dimension physical properties
    // 0 = Instant, 1 = Heavy/Slow
    struct Mass {
        float valence = 0.60f;   // Moods change slowly 
        float arousal = 0.50f;   // Energy shifts moderately
        float control = 0.60f;   // Confidence is sticky
        float novelty = 0.10f;   // Surprise is fast! 
        float obstruct = 0.60f;  // Frustration lingers
    } mass;

    void Reset(AppraisalVector initial) {
        current = initial;
        target = initial;
        velocity = {0,0,0,0,0};
    }

    // The Physics Integrator (Call 60 times/sec)
    void Update(float dt) {
        // 1. Novelty Decay (The "Reflex" Layer)
        // If the agent was surprised, it shouldn't STAY surprised forever.
        // It automatically decays back to 0.0 over time.
        target.novelty = Lerp(target.novelty, 0.0f, dt * 1.5f); 

        // 2. Spring-Damper Physics for Organic Movement
        // This replaces robotic linear interpolation.
        const float omega = 25.0f;  // Stiffness
        const float zeta = 0.75f;    // Damping (No jitter)

        auto UpdateChannel = [&](float& curr, float& vel, float tgt, float inertia) {
            // Heavier mass = Slower response
            float effectiveOmega = omega * (1.0f - inertia);
            
            float f = effectiveOmega * effectiveOmega * (tgt - curr) - 2.0f * zeta * effectiveOmega * vel;
            vel += f * dt;
            curr += vel * dt;
        };

        UpdateChannel(current.valence,  velocity.valence,  target.valence,  mass.valence);
        UpdateChannel(current.arousal,  velocity.arousal,  target.arousal,  mass.arousal);
        UpdateChannel(current.control,  velocity.control,  target.control,  mass.control);
        UpdateChannel(current.novelty,  velocity.novelty,  target.novelty,  mass.novelty);
        UpdateChannel(current.obstruct, velocity.obstruct, target.obstruct, mass.obstruct);
    }
    
    // Helper: Random Wandering (1/f Noise) [cite: 239-241]
    // Call this rarely (e.g., every frame, but with small influence)
    void ApplyNoise(float amount) {
        // Slight organic drift so the face is never 100% frozen
        target.arousal += ((float)GetRandomValue(-100,100)/100.0f) * amount;
        target.valence += ((float)GetRandomValue(-100,100)/100.0f) * amount;
        
        // Clamp to valid ranges
        target.arousal = Clamp(target.arousal, 0.0f, 1.0f);
        target.valence = Clamp(target.valence, -1.0f, 1.0f);
    }
};

// 2. THE MANIFOLD ENGINE
class AffectiveEngine {
private:
    struct DataPoint {
        std::string name;
        AppraisalVector input;
        FaceState output;
    };
    std::vector<DataPoint> trainingData;
    float kernelWidth = 0.45f; 

    //Logger Variables 
    std::ofstream logFile;
    float logTimer = 0.0f;

public:

    void InitLogger()
    {
        logFile.open("brain_debug.log", std::ios::out | std::ios::trunc);
        if (logFile.is_open()) {
            logFile << "TIMESTAMP | MOOD QUERY (V,A,C,N,O) | NEIGHBOR 1 (Dist) | NEIGHBOR 2 (Dist) | WEIGHTS | CRITICAL PARAMS\n";
            logFile << "---------------------------------------------------------------------------------------------------\n";
        }
    }


    void RegisterFace(std::string name, AppraisalVector tags, FaceState state) {
        trainingData.push_back({name, tags, state});
    }

    // FIXED: Now a proper member function taking the DB as const ref
    void LoadFromDB(const FaceDatabase& db) {
        // Tagging helper
        auto Tag = [&](std::string targetName, float v, float a, float c, float n, float o) {
            for (const auto& entry : db.entries) {
                if (entry.name == targetName) {
                    RegisterFace(targetName, {v, a, c, n, o}, entry.state);
                    return;
                }
            }
            // Optional: std::cout << "Warning: " << targetName << " missing." << std::endl;
        };

        // --- TAGGING MAPPING ---
        Tag("face_happy_standard",      0.8f,  0.4f,  0.8f,  0.0f,  0.0f);
        Tag("face_happy_talking",       0.7f,  0.5f,  0.8f,  0.0f,  0.0f);
        Tag("face_happy_closed_eyes",   0.9f,  0.1f,  0.9f,  0.0f,  0.0f);
        Tag("face_silly_tongue",        0.6f,  0.6f,  0.7f,  0.3f,  0.0f);
        Tag("face_kissy_face",          0.8f,  0.3f,  0.6f,  0.0f,  0.0f);
        Tag("face_smug",                0.3f,  0.3f,  1.0f,  0.0f,  0.0f);
        Tag("face_text_happy_D",        0.8f,  0.6f,  0.8f,  0.0f,  0.0f);
        Tag("face_kawaii_sparkle",      0.9f,  0.7f,  0.5f,  0.6f,  0.0f);
        Tag("face_sparkle_stars",       1.0f,  0.8f,  0.6f,  0.7f,  0.0f);
        Tag("face_excited_stars",       1.0f,  0.9f,  0.7f,  0.8f,  0.0f);
        Tag("face_love_hearts",         1.0f,  0.7f,  0.4f,  0.5f,  0.0f);
        Tag("face_shocked_pale",       -0.4f,  1.0f,  0.2f,  1.0f,  0.2f);
        Tag("face_hypnotized",          0.0f,  0.3f,  0.0f,  0.5f,  0.0f);
        Tag("face_sad_standard",       -0.6f,  0.2f,  0.2f,  0.0f,  0.5f);
        Tag("face_sad_frown",          -0.7f,  0.3f,  0.3f,  0.0f,  0.8f);
        Tag("face_text_sad_D",         -0.6f,  0.5f,  0.2f,  0.4f,  0.5f);
        Tag("face_worried_teary",      -0.5f,  0.6f,  0.3f,  0.4f,  0.6f);
        Tag("face_crying_tears",       -0.9f,  0.6f,  0.1f,  0.0f,  0.9f);
        Tag("face_crying_wail",        -1.0f,  0.9f,  0.0f,  0.2f,  1.0f);
        Tag("face_wincing",            -0.6f,  0.8f,  0.4f,  0.6f,  0.7f);
        Tag("face_skeptical",          -0.2f,  0.4f,  0.8f,  0.2f,  0.4f);
        Tag("face_look_side",           0.0f,  0.2f,  0.6f,  0.1f,  0.0f);
        Tag("face_angry_growl",        -0.7f,  0.7f,  0.9f,  0.0f,  0.8f);
        Tag("face_angry_shout",        -0.8f,  1.0f,  1.0f,  0.1f,  1.0f);
        Tag("face_blush",               0.6f,  0.5f,  0.3f,  0.2f,  0.0f);
        Tag("face_tired_droopy",       -0.2f,  0.0f,  0.3f,  0.0f,  0.0f);
        Tag("face_dead_x",             -1.0f,  0.0f,  0.0f,  1.0f,  1.0f);
        Tag("face_pixel_smile",         0.81f,  0.41f,  0.8f,  0.0f,  0.0f);
        Tag("face_pixel_annoyed",      -0.4f,  0.2f,  0.6f,  0.0f,  0.5f);
        Tag("face_pixel_blank",         0.0f,  0.0f,  0.5f,  0.0f,  0.0f);
    }

    // -----------------------------------------------------
    // STEP 2: PREDICT (SELECTOR MODE)
    // -----------------------------------------------------
    FaceState SolveDualLogger(AppraisalVector query) {
        int idx1 = -1, idx2 = -1;
        float dist1 = 100000.0f, dist2 = 100000.0f;

        // 1. Find Neighbors
        for (size_t i = 0; i < trainingData.size(); i++) {
            float d = query.DistSq(trainingData[i].input);
            if (d < dist1) {
                dist2 = dist1; idx2 = idx1;
                dist1 = d; idx1 = (int)i;
            } else if (d < dist2) {
                dist2 = d; idx2 = (int)i;
            }
        }

        if (idx1 == -1) return FaceState();

        // 2. Calculate Weights
        // Avoid divide by zero if faces are identical
        float totalDist = dist1 + dist2;
        if (totalDist < 0.0001f) totalDist = 0.0001f; 

        float w1 = dist2 / totalDist; 
        float w2 = dist1 / totalDist;
        
        // Normalize (Crucial check: do they sum to 1.0?)
        float sum = w1 + w2;
        w1 /= sum; w2 /= sum;

        // 3. LOGGING (Runs 4 times per second to save disk IO)
        // We use a static counter or time check
        static int logFrame = 0;
        logFrame++;
        if (logFile.is_open() && logFrame % 15 == 0) {
            logFile << std::fixed << std::setprecision(2);
            logFile << "[" << logFrame << "] Query: " 
                    << query.valence << "," << query.arousal << " | ";
            
            logFile << "N1: " << trainingData[idx1].name << "(" << dist1 << ") | ";
            
            if (idx2 != -1) {
                logFile << "N2: " << trainingData[idx2].name << "(" << dist2 << ") | ";
                logFile << "W: " << w1 << "/" << w2 << " | ";
                
                // Debug critical conflicts
                float s1_sq = trainingData[idx1].output.eyes.squareness;
                float s2_sq = trainingData[idx2].output.eyes.squareness;
                float s1_shp = trainingData[idx1].output.eyes.eyeShapeID;
                float s2_shp = trainingData[idx2].output.eyes.eyeShapeID;
                
                logFile << "ShapeID: " << s1_shp << " vs " << s2_shp;
                if (std::abs(s1_sq - s2_sq) > 0.5f) logFile << " [SQUARENESS CONFLICT]";
                if (std::abs(s1_shp - s2_shp) > 0.1f) logFile << " [SHAPE CONFLICT]";
            } else {
                logFile << "N2: NONE";
            }
            logFile << "\n";
            logFile.flush(); // Ensure it writes to disk immediately
        }

        // 4. Snap Logic (The "Anti-Funk" Safety Valve)
        // If the two faces are fundamentally incompatible (different topology), 
        // DO NOT BLEND. Just snap to the closest one.
        bool incompatible = false;
        
        // Check 1: Shape ID difference
        if (idx2 != -1) {
            float id1 = trainingData[idx1].output.eyes.eyeShapeID;
            float id2 = trainingData[idx2].output.eyes.eyeShapeID;
            if (std::abs(id1 - id2) > 0.1f) incompatible = true;
        }

        // Check 2: Distance dominance
        // If we are 90% closer to A than B, just use A. 
        // Blending the last 10% usually looks like ghosting.
        if (w1 > 0.85f) incompatible = true; 

        if (incompatible || idx2 == -1) {
            return trainingData[idx1].output;
        }

        // 5. Blending (Only runs if shapes are compatible)
        FaceState result;
        std::memset(&result, 0, sizeof(FaceState));

        const auto& s1 = trainingData[idx1].output;
        const auto& s2 = trainingData[idx2].output;

        #define BLEND_EYE(PARAM) result.eyes.PARAM = (s1.eyes.PARAM * w1) + (s2.eyes.PARAM * w2)
        #define BLEND_MOUTH(PARAM) result.mouth.PARAM = (s1.mouth.PARAM * w1) + (s2.mouth.PARAM * w2)

        BLEND_EYE(bend); BLEND_EYE(eyeThickness); BLEND_EYE(eyeSide);
        BLEND_EYE(scaleX); BLEND_EYE(scaleY); BLEND_EYE(angle);
        BLEND_EYE(spacing); BLEND_EYE(squareness); 
        BLEND_EYE(stressLevel); BLEND_EYE(gloomLevel); BLEND_EYE(spiralSpeed);
        BLEND_EYE(lookX); BLEND_EYE(lookY);
        BLEND_EYE(pixelation);
        
        BLEND_EYE(eyebrowThickness); BLEND_EYE(eyebrowLength); BLEND_EYE(eyebrowSpacing);
        BLEND_EYE(eyebrowX); BLEND_EYE(eyebrowY); BLEND_EYE(browScale);
        BLEND_EYE(browSide); BLEND_EYE(browAngle); BLEND_EYE(browBend); BLEND_EYE(browBendOffset);
        
        BLEND_EYE(tearsLevel); BLEND_EYE(blushScale); BLEND_EYE(blushX); 
        BLEND_EYE(blushY); BLEND_EYE(blushSpacing);

        BLEND_MOUTH(open); BLEND_MOUTH(width); BLEND_MOUTH(curve);
        BLEND_MOUTH(mouthAngle); BLEND_MOUTH(squeezeTop); BLEND_MOUTH(squeezeBottom);
        BLEND_MOUTH(teethY); BLEND_MOUTH(tongueUp); BLEND_MOUTH(tongueX);
        BLEND_MOUTH(tongueWidth); BLEND_MOUTH(asymmetry); BLEND_MOUTH(squareness);
        BLEND_MOUTH(teethWidth); BLEND_MOUTH(teethGap); BLEND_MOUTH(scale);
        BLEND_MOUTH(outlineThickness); BLEND_MOUTH(sigma); BLEND_MOUTH(power);
        BLEND_MOUTH(maxLiftValue); BLEND_MOUTH(lookX); BLEND_MOUTH(lookY);
        BLEND_MOUTH(stressLines);

        // Discrete Snap
        result.eyes.eyeShapeID = s1.eyes.eyeShapeID;
        result.eyes.showBrow = s1.eyes.showBrow;
        result.eyes.useLowerBrow = s1.eyes.useLowerBrow;
        result.eyes.showTears = s1.eyes.showTears;
        result.eyes.showBlush = s1.eyes.showBlush;
        result.eyes.blushMode = s1.eyes.blushMode;
        result.mouth.showInnerMouth = s1.mouth.showInnerMouth;
        result.mouth.isThreeShape = s1.mouth.isThreeShape;
        result.mouth.isDShape = s1.mouth.isDShape;
        result.mouth.isSlashShape = s1.mouth.isSlashShape;

        return result;
    }
    FaceState SolveDual(AppraisalVector query) {
        int idx1 = -1, idx2 = -1;
        float dist1 = 100000.0f, dist2 = 100000.0f;

        // 1. Find the Top 2 Closest Faces
        for (size_t i = 0; i < trainingData.size(); i++) {
            float d = query.DistSq(trainingData[i].input);
            if (d < dist1) {
                dist2 = dist1; idx2 = idx1;
                dist1 = d; idx1 = (int)i;
            } else if (d < dist2) {
                dist2 = d; idx2 = (int)i;
            }
        }

        // Safety check for empty DB
        if (idx1 == -1) return FaceState();

        // --- CRITICAL FIX 1: THE BLACK HOLE PATCH ---
        // If we are ON TOP of a face (distance near 0), returns immediately.
        // This prevents the 0/0 = NaN division error.
        if (dist1 < 0.0001f) return trainingData[idx1].output;

        // If no second neighbor exists, return the first
        if (idx2 == -1) return trainingData[idx1].output;

        

        // --- CRITICAL FIX 2: THE FRANKENSTEIN GUARD ---
        bool incompatible = false;
        const auto& s1 = trainingData[idx1].output;
        const auto& s2 = trainingData[idx2].output;

        // A. Topology Conflict (Star vs Circle)
        if (std::abs(s1.eyes.eyeShapeID - s2.eyes.eyeShapeID) > 0.1f) incompatible = true;
        
        // B. Style Conflict (Pixel Art vs HD Vector)
        // Checks 'squareness' and 'pixelation' to ensure we don't mix styles.
        if (std::abs(s1.eyes.squareness - s2.eyes.squareness) > 0.2f) incompatible = true;
        if (std::abs(s1.eyes.pixelation - s2.eyes.pixelation) > 1.0f) incompatible = true;

        // C. Calculate Weights
        float totalDist = dist1 + dist2;
        float w1 = dist2 / totalDist; // Inverse distance weighting
        
        // D. Dominance Snap
        // If we are 90% closer to A than B, just snap to A. Blending the last 10% looks like ghosting.
        if (w1 > 0.90f) incompatible = true; 

        // EXECUTE SNAP IF INCOMPATIBLE
        if (incompatible) {
            return s1; 
        }

        // 3. Blend (Only runs if safe)
        FaceState result;
        std::memset(&result, 0, sizeof(FaceState));

        float w2 = 1.0f - w1;

        // BLENDING MACROS
        #define BLEND_EYE(PARAM) result.eyes.PARAM = (s1.eyes.PARAM * w1) + (s2.eyes.PARAM * w2)
        #define BLEND_MOUTH(PARAM) result.mouth.PARAM = (s1.mouth.PARAM * w1) + (s2.mouth.PARAM * w2)

        BLEND_EYE(bend); BLEND_EYE(eyeThickness); BLEND_EYE(eyeSide);
        BLEND_EYE(scaleX); BLEND_EYE(scaleY); BLEND_EYE(angle);
        BLEND_EYE(spacing); BLEND_EYE(squareness); 
        BLEND_EYE(stressLevel); BLEND_EYE(gloomLevel); BLEND_EYE(spiralSpeed);
        BLEND_EYE(lookX); BLEND_EYE(lookY);
        BLEND_EYE(pixelation);
        
        BLEND_EYE(eyebrowThickness); BLEND_EYE(eyebrowLength); BLEND_EYE(eyebrowSpacing);
        BLEND_EYE(eyebrowX); BLEND_EYE(eyebrowY); BLEND_EYE(browScale);
        BLEND_EYE(browSide); BLEND_EYE(browAngle); BLEND_EYE(browBend); BLEND_EYE(browBendOffset);
        
        BLEND_EYE(tearsLevel); BLEND_EYE(blushScale); BLEND_EYE(blushX); 
        BLEND_EYE(blushY); BLEND_EYE(blushSpacing);

        BLEND_MOUTH(open); BLEND_MOUTH(width); BLEND_MOUTH(curve);
        BLEND_MOUTH(mouthAngle); BLEND_MOUTH(squeezeTop); BLEND_MOUTH(squeezeBottom);
        BLEND_MOUTH(teethY); BLEND_MOUTH(tongueUp); BLEND_MOUTH(tongueX);
        BLEND_MOUTH(tongueWidth); BLEND_MOUTH(asymmetry); BLEND_MOUTH(squareness);
        BLEND_MOUTH(teethWidth); BLEND_MOUTH(teethGap); BLEND_MOUTH(scale);
        BLEND_MOUTH(outlineThickness); BLEND_MOUTH(sigma); BLEND_MOUTH(power);
        BLEND_MOUTH(maxLiftValue); BLEND_MOUTH(lookX); BLEND_MOUTH(lookY);
        BLEND_MOUTH(stressLines);

        // Discrete Snap (Winner Takes All)
        result.eyes.eyeShapeID = s1.eyes.eyeShapeID;
        result.eyes.showBrow = s1.eyes.showBrow;
        result.eyes.useLowerBrow = s1.eyes.useLowerBrow;
        result.eyes.showTears = s1.eyes.showTears;
        result.eyes.showBlush = s1.eyes.showBlush;
        result.eyes.blushMode = s1.eyes.blushMode;
        result.mouth.showInnerMouth = s1.mouth.showInnerMouth;
        result.mouth.isThreeShape = s1.mouth.isThreeShape;
        result.mouth.isDShape = s1.mouth.isDShape;
        result.mouth.isSlashShape = s1.mouth.isSlashShape;

        return result;
    }

    FaceState Solve(AppraisalVector query) {
        int bestIdx = -1;
        float minDistance = 100000.0f; 

        // 1. Scan the whole database for the closest match
        for (size_t i = 0; i < trainingData.size(); i++) {
            float d = query.DistSq(trainingData[i].input);
            
            // If this face is closer, it becomes the new winner
            if (d < minDistance) {
                minDistance = d;
                bestIdx = (int)i;
            }
        }

        // 2. Return the exact handcrafted pose
        if (bestIdx != -1) {
             return trainingData[bestIdx].output;
        }

        // Fallback (Should never happen if DB is loaded)
        FaceState empty;
        return empty;
    }
    // FIXED: Corrected signature and undefined variables
    void ApplySpeech(FaceState& state, float visemeOpen, float visemeWidth, float speechIntensity) {
        if (speechIntensity <= 0.01f) return;

        float dampen = 1.0f - (speechIntensity * 0.7f); 
        state.mouth.curve *= dampen;
        
        // Manual lerp since Raylib's Lerp might collide if using <algorithm>
        float start = state.mouth.width;
        state.mouth.width = start + speechIntensity * (visemeWidth - start);

        state.mouth.open += visemeOpen;
    }
};