import sys
import time
import zmq
import threading
import math
from raylib import *
from pyray import *

# --- CONFIGURATION ---
SCREEN_WIDTH = 800  # Raylib DRM usually grabs native res, but good to define
SCREEN_HEIGHT = 480
IPC_ADDRESS = "tcp://127.0.0.1:5555"

# --- COLORS ---
OUTSIDE_COLOR = GetColor(0xA3D9C1FF) 
FACE_BG_COLOR = GetColor(0xC7E1C4FF)
SCANLINE      = GetColor(0x00000022)
VIGNETTE      = GetColor(0x00000055)

class BMOFace:
    def __init__(self):
        self.running = True
        
        # 1. Setup ZeroMQ Listener (Subscribes to Brain)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(IPC_ADDRESS)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all
        
        # 2. Initialize Raylib in DRM Mode
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"BMO Face")
        SetTargetFPS(60)
        HideCursor()
        
        # 3. State Variables
        self.emotion_target = 0.0 # 0.0 = Neutral, 1.0 = Happy
        self.emotion_value = 0.0  # Current blended value
        self.is_talking = False
        self.blink_timer = 0.0
        self.is_blinking = False
        
        # Threading lock for state updates
        self.lock = threading.Lock()

    # --- HELPER MATH ---
    def lerp(self, a, b, t):
        return a + (b - a) * t

    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def draw_quadratic_bezier(self, p0, p1, p2, color, segments=20):
        prev_x, prev_y = p0
        for i in range(1, segments + 1):
            t = i / segments
            x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
            y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
            DrawLine(int(prev_x), int(prev_y), int(x), int(y), color)
            prev_x, prev_y = x, y

    # --- DRAWING LOGIC (Adapted from your code) ---
    def draw_eyes(self, box_x, box_y, box_w, box_h):
        left_eye_x  = box_x + box_w * 0.35
        right_eye_x = box_x + box_w * 0.65
        eye_y       = box_y + box_h * 0.35

        EYE_RADIUS = max(5, int(box_w * 0.06))
        PUPIL_RADIUS = max(2, int(box_w * 0.025))
        MAX_PUPIL_OFFSET = max(3, int(box_w * 0.035))

        # We don't have mouse input in headless mode usually, 
        # so we default to looking center or slowly drifting
        # Later we can hook this to 'Attention' variables
        t = GetTime()
        mx = left_eye_x + math.sin(t * 0.5) * 20
        my = eye_y + math.cos(t * 0.3) * 10

        for eye_x in [left_eye_x, right_eye_x]:
            if self.is_blinking:
                DrawRectangle(int(eye_x - EYE_RADIUS - 4), int(eye_y), EYE_RADIUS * 2 + 8, int(EYE_RADIUS * 0.6), BLACK)
                continue

            # Eye white
            DrawCircle(int(eye_x), int(eye_y), EYE_RADIUS, BLACK)

            # Pupil tracking
            dx = self.clamp(mx - eye_x, -30, 30)
            dy = self.clamp(my - eye_y, -30, 30)
            px = eye_x + (dx / 30) * MAX_PUPIL_OFFSET
            py = eye_y + (dy / 30) * MAX_PUPIL_OFFSET

            DrawCircle(int(px), int(py), PUPIL_RADIUS, WHITE)

    def draw_mouth(self, width, height):
        cx = width * 0.5
        cy = height * 0.55
        mouth_w = width * 0.15
        
        # Blend mouth height based on emotion (Smile factor)
        base_h = self.lerp(height * 0.01, height * 0.05, self.emotion_value)

        # Talk wobble
        if self.is_talking:
            # High speed wobble for talking
            base_h += math.sin(GetTime() * 15) * height * 0.03

        # Bezier Control Points
        top_left  = (cx - mouth_w/2, cy)
        top_mid   = (cx, cy - base_h)
        top_right = (cx + mouth_w/2, cy)

        bottom_left  = (cx - mouth_w/2, cy)
        bottom_mid   = (cx, cy + base_h/2)
        bottom_right = (cx + mouth_w/2, cy)

        self.draw_quadratic_bezier(top_left, top_mid, top_right, BLACK)
        self.draw_quadratic_bezier(bottom_left, bottom_mid, bottom_right, BLACK)

        # Fill mouth (simple lines)
        steps = 8
        for i in range(steps + 1):
            t = i / steps
            x_start = self.lerp(top_left[0], bottom_left[0], t)
            y_start = self.lerp(top_left[1], bottom_left[1], t)
            x_end   = self.lerp(top_right[0], bottom_right[0], t)
            y_end   = self.lerp(top_right[1], bottom_right[1], t)
            DrawLine(int(x_start), int(y_start), int(x_end), int(y_end), WHITE)

    def draw_crt_overlay(self, box_x, box_y, box_w, box_h):
        # Scanlines
        line_spacing = max(3, int(box_h * 0.015))
        for y in range(int(box_y), int(box_y + box_h), line_spacing):
            DrawRectangle(int(box_x), y, int(box_w), 1, SCANLINE)
        
        # Vignette
        thick = max(20, int(min(box_w, box_h) * 0.1))
        DrawRectangle(int(box_x), int(box_y), int(box_w), thick, VIGNETTE) # Top
        DrawRectangle(int(box_x), int(box_y + box_h - thick), int(box_w), thick, VIGNETTE) # Bottom
        DrawRectangle(int(box_x), int(box_y), thick, int(box_h), VIGNETTE) # Left
        DrawRectangle(int(box_x + box_w - thick), int(box_y), thick, int(box_h), VIGNETTE) # Right

    def check_messages(self):
        """Non-blocking check for ZMQ commands"""
        try:
            msg = self.socket.recv_json(flags=zmq.NOBLOCK)
            with self.lock:
                # Map commands to state
                if "emotion_target" in msg:
                    self.emotion_target = float(msg["emotion_target"])
                if "talking" in msg:
                    self.is_talking = bool(msg["talking"])
                print(f"Received: {msg}")
        except zmq.Again:
            pass

    def run(self):
        while not WindowShouldClose():
            # 1. Update State
            self.check_messages()
            dt = GetFrameTime()
            
            # Smoothly blend emotion
            self.emotion_value = self.lerp(self.emotion_value, self.emotion_target, 5 * dt)
            
            # Blink Logic
            self.blink_timer += dt
            if self.blink_timer > 3.0:
                self.is_blinking = True
                if self.blink_timer > 3.15:
                    self.is_blinking = False
                    self.blink_timer = 0.0
            
            # 2. Draw
            width = GetScreenWidth()
            height = GetScreenHeight()
            
            box_width = width * 0.8 # Slightly larger for fullscreen feel
            box_height = height * 0.8
            box_x = (width - box_width) / 2
            box_y = (height - box_height) / 2

            BeginDrawing()
            ClearBackground(OUTSIDE_COLOR)
            
            # Face Box
            DrawRectangle(int(box_x), int(box_y), int(box_width), int(box_height), FACE_BG_COLOR)
            DrawRectangleLines(int(box_x), int(box_y), int(box_width), int(box_height), BLACK)
            
            # Features
            self.draw_eyes(box_x, box_y, box_width, box_height)
            self.draw_mouth(width, height)
            self.draw_crt_overlay(box_x, box_y, box_width, box_height)
            
            # Debug info (remove in production)
            DrawFPS(10, 10)
            EndDrawing()

        CloseWindow()

if __name__ == "__main__":
    face = BMOFace()
    face.run()