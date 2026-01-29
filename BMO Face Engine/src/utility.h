#ifndef UTILITY_H
#define UTILITY_H

#include "raylib.h"
#include <algorithm> // for std::min

// --- DESIGN RESOLUTION ---
// The resolution you "imagine" you are working in. 
// All your spacing=612, thickness=4, etc. are tuned for this.
const float REF_WIDTH  = 1920.0f;
const float REF_HEIGHT = 1080.0f;

struct Scaler {
    float scale = 1.0f;

    // Call this at the start of every frame
    void Update() {
        float screenW = (float)GetScreenWidth();
        float screenH = (float)GetScreenHeight();

        float ratioX = screenW / REF_WIDTH;
        float ratioY = screenH / REF_HEIGHT;

        // "Contain" mode: Use the smaller ratio.
        // This ensures the face fits entirely on screen, even on portrait phones.
        scale = std::min(ratioX, ratioY);
    }

    // Convert Reference Value -> Screen Value
    float S(float val) const {
        return val * scale;
    }
    
    // Convert Reference Point -> Screen Point (relative to center)
    // Useful if you have specific hardcoded offsets
    Vector2 S(Vector2 v) const {
        return { v.x * scale, v.y * scale };
    }
};

// Make a global instance available so both main.cpp and the class can see it
// (Alternatively, pass 'Scaler&' to your Draw functions)
inline Scaler GlobalScaler;

#endif // UTILITY_H