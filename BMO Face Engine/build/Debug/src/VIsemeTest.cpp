#include "raylib.h"
#include "json.hpp"
#include <zmq.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstring>
#include <cmath>
#include <fstream>

using json = nlohmann::json;

// ---------------------------------------------------------
// 1. ASSET LOADER (Unchanged)
// ---------------------------------------------------------
struct SpriteAtlas {
    Texture2D texture{};
    std::unordered_map<std::string, Rectangle> frames;

    void Load(const char* imagePath, const char* jsonPath) {
        texture = LoadTexture(imagePath);
        std::ifstream f(jsonPath);
        if (!f.is_open()) return;
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
// 2. STREAMING AUDIO SYSTEM (NEW)
// ---------------------------------------------------------
struct AudioPacket {
    std::vector<float> samples;
    std::string viseme;
};

// Thread-safe buffer between ZMQ (Network) and Raylib (Audio/Render)
class AudioStreamer {
private:
    std::deque<float> audioBuffer;

    struct VisemeEvent {
        size_t sampleIndex;   // timeline index in "played samples"
        std::string shape;
    };
    std::deque<VisemeEvent> visemeQueue;

    std::mutex mtx;
    AudioStream stream{};
    std::string currentViseme = "mouth_phoneme_X";
    bool isTalking = false;

    // Timeline tracking (in samples at 48kHz)
    size_t totalSamplesPlayed = 0;

    // ZMQ
    zmq::context_t ctx;
    zmq::socket_t receiver;
    std::thread netThread;
    std::atomic<bool> running{true};

    // Config
    static constexpr int SAMPLE_RATE = 48000;
    static constexpr int FRAMES_PER_UPDATE = 4096; // match SetAudioStreamBufferSizeDefault
    static constexpr size_t STARTUP_BUFFER = SAMPLE_RATE / 4; // ~250ms

public:
    AudioStreamer() : ctx(1), receiver(ctx, ZMQ_PULL) {}

    void Init() {
        SetAudioStreamBufferSizeDefault(FRAMES_PER_UPDATE);
        stream = LoadAudioStream(SAMPLE_RATE, 32, 1);
        PlayAudioStream(stream);

        // Avoid hanging forever on recv during shutdown
        receiver.set(zmq::sockopt::rcvtimeo, 100);   // 100ms timeout
        receiver.set(zmq::sockopt::linger, 0);

        receiver.connect("tcp://localhost:5555");

        netThread = std::thread(&AudioStreamer::NetworkLoop, this);
        TraceLog(LOG_INFO, "AUDIO: Connected via ZMQ PULL.");
    }

    void NetworkLoop() {
        while (running.load()) {
            zmq::message_t msg;
            try {
                auto ok = receiver.recv(msg, zmq::recv_flags::none);
                if (!ok) continue; // timeout

                auto* ptr = static_cast<uint8_t*>(msg.data());
                size_t size = msg.size();
                if (size < 4) continue;

                uint32_t visemeLen = 0;
                std::memcpy(&visemeLen, ptr, 4);
                ptr += 4;

                // sanity checks
                if (visemeLen > 256) continue;
                if (size < 4 + visemeLen) continue;

                std::string newViseme((char*)ptr, visemeLen);
                ptr += visemeLen;

                size_t audioBytes = size - (4 + visemeLen);
                if (audioBytes % sizeof(float) != 0) continue;
                int sampleCount = (int)(audioBytes / sizeof(float));
                if (sampleCount <= 0) continue;

                float* samples = reinterpret_cast<float*>(ptr);

                std::lock_guard<std::mutex> lock(mtx);

                // Schedule viseme at the point this chunk will actually START playing
                // (current playhead + how much audio is already queued)
                size_t eventAt = totalSamplesPlayed + audioBuffer.size();
                visemeQueue.push_back({ eventAt, std::move(newViseme) });

                // Append audio
                audioBuffer.insert(audioBuffer.end(), samples, samples + sampleCount);

                // Add this static counter
                static int pktDebugCount = 0;
                pktDebugCount++;

                if (pktDebugCount % 50 == 0) {
                    std::cout << "[CPP][RECV] pkt=" << pktDebugCount 
                              << " viseme=" << newViseme 
                              << " bufferSize=" << audioBuffer.size() << "\n";
                }
            } catch (...) {
                // swallow for now, but ideally log errors
            }
        }
    }

    void Update() {

        // Outside the loop/class or static inside the method
        static std::string lastPrintedViseme = "";
            
        if (currentViseme != lastPrintedViseme) {
            std::cout << "[CPP][PLAY] Switching to: " << currentViseme << "\n";
            lastPrintedViseme = currentViseme;
        }
        if (!IsAudioStreamProcessed(stream)) return;

        std::vector<float> feed(FRAMES_PER_UPDATE, 0.0f);
        bool hadAudio = false;

        {
            std::lock_guard<std::mutex> lock(mtx);

            // Optional: avoid starting playback until some buffer has accumulated
            // This makes the system resilient to jitter.
            static bool started = false;
            if (!started) {
                if (audioBuffer.size() < STARTUP_BUFFER) {
                    isTalking = false;
                    return; // don't feed yet; let buffer fill
                }
                started = true;
            }

            int available = (int)audioBuffer.size();
            int n = std::min(available, FRAMES_PER_UPDATE);

            for (int i = 0; i < n; i++) {
                feed[i] = audioBuffer.front();
                audioBuffer.pop_front();
            }

            hadAudio = (n > 0);
            totalSamplesPlayed += (size_t)FRAMES_PER_UPDATE; // timeline advances regardless (we padded silence)

            // Apply viseme events whose time has arrived
            while (!visemeQueue.empty() && visemeQueue.front().sampleIndex <= totalSamplesPlayed) {
                currentViseme = visemeQueue.front().shape;
                visemeQueue.pop_front();
            }

            // Talking state: if we had real audio or still have queued audio, treat as talking
            isTalking = hadAudio || !audioBuffer.empty();
            if (!isTalking) {
                currentViseme = "mouth_phoneme_X";
            }
        }

        // Always feed the device; never starve
        UpdateAudioStream(stream, feed.data(), FRAMES_PER_UPDATE);
    }

    std::string GetCurrentMouth() {
        std::lock_guard<std::mutex> lock(mtx);
        return currentViseme;
    }

    bool IsTalking() {
        std::lock_guard<std::mutex> lock(mtx);
        return isTalking;
    }

    void Cleanup() {
        running.store(false);
        if (netThread.joinable()) netThread.join();
        StopAudioStream(stream);
        UnloadAudioStream(stream);
    }
};


// ---------------------------------------------------------
// 3. MAIN (Modified for Split-Brain)
// ---------------------------------------------------------
int main() {
    const int W = 1280, H = 720;
    InitWindow(W, H, "BMO: Embodied AI");
    InitAudioDevice();
    SetTargetFPS(60);

    SpriteAtlas atlas;
    atlas.Load("assets/BMO_Animation_LipSyncSprite.png", "assets/BMO_Animation_Lipsync.json");

    AudioStreamer voiceSystem;
    voiceSystem.Init();

    // Physics vars
    float scaleY = 1.0f, scaleX = 1.0f;
    std::string currentMood = "face_happy_standard";

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        
        // 1. Process Network & Audio
        voiceSystem.Update();
        std::string mouthTex = voiceSystem.GetCurrentMouth();
        
        // 2. Physics & Procedural Animation
        // Squash/Stretch based on talking
        float targetSY = 1.0f;
        if (voiceSystem.IsTalking()) targetSY = 1.05f; // Stretch when talking
        
        scaleY += (targetSY - scaleY) * 10.0f * dt;
        scaleX = 2.0f - scaleY; // Volume preservation

        // 3. Draw
        BeginDrawing();
        ClearBackground({131, 220, 169, 255}); 

        // Draw Eyes (Mood)
        atlas.DrawFrame(currentMood + "_eyes", 0, 0, W, H, {0,0}, {scaleX, scaleY}, WHITE);
        
        // Draw Mouth (Synced Viseme)
        atlas.DrawFrame(mouthTex, 0, 0, W, H, {0, -10}, {scaleX, scaleY}, WHITE);

        DrawText("WAITING FOR ZMQ STREAM...", 20, 20, 20, DARKGRAY);
        EndDrawing();
    }

    voiceSystem.Cleanup();
    CloseAudioDevice();
    CloseWindow();
    return 0;
}