// Lightweight motion tracker for BMO's eyes.
//
// Deliberately minimal: raw GStreamer C API only (no OpenCV, no Python
// runtime). Captures tiny 80x60 grayscale frames, does manual byte-level
// consecutive-frame differencing, and writes the motion centroid to a
// tmpfs file that the face engine polls each frame. No accumulating
// history/background model -- just two small frame buffers swapped each
// cycle, so memory stays flat over time and RSS should sit in the tens of
// MB (mostly GStreamer's own runtime/plugin loading), not the ~190MB the
// Python+OpenCV prototype used.
//
// Build: g++ -O2 motion_tracker.cpp -o motion_tracker \
//          $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0)

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Small on purpose: 80x60 = 4800 bytes/frame. Plenty of resolution for a
// coarse "where is the person" centroid, not for anything detailed.
static const int FRAME_W = 80;
static const int FRAME_H = 60;
static const int DIFF_THRESHOLD = 25;      // per-pixel brightness delta to count as "moved"
static const int MIN_MOTION_PIXELS = 40;   // ignore sensor noise / tiny flicker

static const char* OUTPUT_PATH = "/dev/shm/bmo_motion.txt";

int main(int argc, char** argv) {
    gst_init(&argc, &argv);

    const char* pipelineDesc =
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw,width=80,height=60,format=GRAY8 ! "
        "appsink name=sink drop=true max-buffers=1 sync=false";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipelineDesc, &error);
    if (!pipeline) {
        fprintf(stderr, "Failed to build pipeline: %s\n", error ? error->message : "unknown error");
        return 1;
    }

    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    std::vector<unsigned char> prevFrame(FRAME_W * FRAME_H, 0);
    bool havePrev = false;

    while (true) {
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
        if (!sample) break; // pipeline EOS'd or errored

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            if (map.size >= (gsize)(FRAME_W * FRAME_H)) {
                const unsigned char* cur = map.data;

                if (havePrev) {
                    long sumX = 0, sumY = 0, weight = 0;
                    for (int y = 0; y < FRAME_H; y++) {
                        for (int x = 0; x < FRAME_W; x++) {
                            int idx = y * FRAME_W + x;
                            int d = (int)cur[idx] - (int)prevFrame[idx];
                            if (d < 0) d = -d;
                            if (d > DIFF_THRESHOLD) {
                                sumX += x;
                                sumY += y;
                                weight++;
                            }
                        }
                    }

                    FILE* f = fopen(OUTPUT_PATH, "w");
                    if (f) {
                        if (weight > MIN_MOTION_PIXELS) {
                            // Normalize centroid to -1..1 (0,0 = frame center)
                            float cx = ((float)sumX / weight) / FRAME_W * 2.0f - 1.0f;
                            float cy = ((float)sumY / weight) / FRAME_H * 2.0f - 1.0f;
                            fprintf(f, "1 %.4f %.4f\n", cx, cy);
                        } else {
                            fprintf(f, "0 0 0\n");
                        }
                        fclose(f);
                    }
                }

                memcpy(prevFrame.data(), cur, FRAME_W * FRAME_H);
                havePrev = true;
            }
            gst_buffer_unmap(buffer, &map);
        }
        gst_sample_unref(sample);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    return 0;
}
