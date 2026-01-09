import sys
import time
import zmq
import threading
import math
from raylib import *
from pyray import *

# --- 1. CONFIGURATION & CONSTANTS ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480
IPC_ADDRESS = "tcp://127.0.0.1:5555"

# --- COLORS (Restored) ---
# We define these globally so the class can see them
OUTSIDE_COLOR = GetColor(0xA3D9C1FF) 
FACE_BG_COLOR = GetColor(0xC7E1C4FF)
SCANLINE      = GetColor(0x00000022)
VIGNETTE      = GetColor(0x00000055)

# --- SHAPE DEFINITIONS ---
EYE_DOT     = 0
EYE_OPEN    = 1  # Hollow circle (Surprised)
EYE_LINE    = 2  # Closed/Sleeping
EYE_CHEVRON = 3  # > < (Excited/Wincing)

MOUTH_CURVE = 0  # Standard Smile/Frown
MOUTH_OPEN  = 1  # :D or :O
MOUTH_FLAT  = 2  # Straight line (Annoyed)

class BMOFace:
    def __init__(self):
        self.running = True
        
        # Setup ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(IPC_ADDRESS)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "") 
        
        # Initialize Raylib
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, b"BMO Face")
        SetTargetFPS(60)
        
        # State Variables
        self.eye_shape = EYE_DOT
        self.mouth_shape = MOUTH_CURVE
        
        self.curvature = 0.5   # -1.0 to 1.0
        self.is_talking = False
        self.blink_timer = 0.0
        self.is_blinking = False
        
        # Visual Style
        self.face_color = BLACK
        self.lock = threading.Lock()

    # --- HELPER MATH ---
    def lerp(self, a, b, t):
        return a + (b - a) * t

    # --- NEW: MANUAL BEZIER DRAWING ---
    # This replaces the missing Raylib function
    def draw_manual_bezier(self, start, end, control, thick, color, segments=20):
        previous = start
        t_step = 1.0 / segments
        
        for i in range(1, segments + 1):
            t = i * t_step
            u = 1 - t
            
            # Quadratic Bezier Formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
            # Vector2 attributes are .x and .y
            x = (u*u * start.x) + (2*u*t * control.x) + (t*t * end.x)
            y = (u*u * start.y) + (2*u*t * control.y) + (t*t * end.y)
            
            current = Vector2(x, y)
            DrawLineEx(previous, current, thick, color)
            previous = current

    # --- DRAWING PRIMITIVES ---
    def draw_chevron_eye(self, x, y, radius, is_left_eye):
        thickness = 6.0
        size = radius * 1.5
        direction = 1 if is_left_eye else -1 
        tip_offset = size * direction
        
        DrawLineEx(Vector2(x + tip_offset, y - size), Vector2(x, y), thickness, self.face_color)
        DrawLineEx(Vector2(x, y), Vector2(x + tip_offset, y + size), thickness, self.face_color)

    def draw_open_eye(self, x, y, radius):
        DrawRing(Vector2(x, y), radius * 0.7, radius, 0, 360, 0, self.face_color)

    def draw_line_eye(self, x, y, radius):
        w = radius * 2.5
        DrawRectangle(int(x - w/2), int(y - 4), int(w), 8, self.face_color)

    # --- MAIN DRAW FUNCTIONS ---
    def draw_eyes(self, box_x, box_y, box_w, box_h):
        left_eye_x  = box_x + box_w * 0.35
        right_eye_x = box_x + box_w * 0.65
        eye_y       = box_y + box_h * 0.35
        radius = max(5.0, box_w * 0.06)

        if self.is_blinking:
            self.draw_line_eye(left_eye_x, eye_y, radius)
            self.draw_line_eye(right_eye_x, eye_y, radius)
            return

        # Left Eye
        if self.eye_shape == EYE_DOT:
            DrawCircle(int(left_eye_x), int(eye_y), radius, self.face_color)
            DrawCircle(int(left_eye_x - radius*0.3), int(eye_y - radius*0.3), radius*0.3, WHITE)
        elif self.eye_shape == EYE_OPEN:
            self.draw_open_eye(left_eye_x, eye_y, radius)
        elif self.eye_shape == EYE_CHEVRON:
            self.draw_chevron_eye(left_eye_x, eye_y, radius, True)
        elif self.eye_shape == EYE_LINE:
            self.draw_line_eye(left_eye_x, eye_y, radius)

        # Right Eye
        if self.eye_shape == EYE_DOT:
            DrawCircle(int(right_eye_x), int(eye_y), radius, self.face_color)
            DrawCircle(int(right_eye_x - radius*0.3), int(eye_y - radius*0.3), radius*0.3, WHITE)
        elif self.eye_shape == EYE_OPEN:
            self.draw_open_eye(right_eye_x, eye_y, radius)
        elif self.eye_shape == EYE_CHEVRON:
            self.draw_chevron_eye(right_eye_x, eye_y, radius, False)
        elif self.eye_shape == EYE_LINE:
            self.draw_line_eye(right_eye_x, eye_y, radius)

    def draw_mouth(self, width, height):
        cx = width * 0.5
        cy = height * 0.55
        mouth_w = width * 0.15
        mouth_h = height * 0.05
        
        if self.is_talking:
            mouth_h += abs(math.sin(GetTime() * 15) * 10)

        if self.mouth_shape == MOUTH_FLAT:
            DrawRectangle(int(cx - mouth_w/2), int(cy), int(mouth_w), 8, self.face_color)

        elif self.mouth_shape == MOUTH_CURVE:
            start = Vector2(cx - mouth_w/2, cy)
            end   = Vector2(cx + mouth_w/2, cy)
            
            # Control point determines curvature (Smile vs Frown)
            control_y = cy + (self.curvature * 60) 
            control = Vector2(cx, control_y)
            
            # --- FIX: Use Manual Bezier ---
            self.draw_manual_bezier(start, end, control, 8.0, self.face_color)

        elif self.mouth_shape == MOUTH_OPEN:
            rect_h = mouth_h * 4
            DrawEllipse(int(cx), int(cy + rect_h/2), int(mouth_w/2), int(rect_h/2), self.face_color)
            DrawRectangle(int(cx - mouth_w/2), int(cy - rect_h), int(mouth_w), int(rect_h), FACE_BG_COLOR)
            DrawRectangle(int(cx - mouth_w/2), int(cy), int(mouth_w), 6, self.face_color)

    def draw_crt_overlay(self, box_x, box_y, box_w, box_h):
        line_spacing = 4
        for y in range(int(box_y), int(box_y + box_h), line_spacing):
            DrawRectangle(int(box_x), y, int(box_w), 1, SCANLINE)
        
        thick = 20
        DrawRectangle(int(box_x), int(box_y), int(box_w), thick, VIGNETTE)
        DrawRectangle(int(box_x), int(box_y + box_h - thick), int(box_w), thick, VIGNETTE)
        DrawRectangle(int(box_x), int(box_y), thick, int(box_h), VIGNETTE)
        DrawRectangle(int(box_x + box_w - thick), int(box_y), thick, int(box_h), VIGNETTE)

    def check_messages(self):
        try:
            msg = self.socket.recv_json(flags=zmq.NOBLOCK)
            with self.lock:
                if "expression" in msg:
                    expr = msg["expression"]
                    if expr == "HAPPY":
                        self.eye_shape, self.mouth_shape, self.curvature = EYE_DOT, MOUTH_CURVE, 1.0
                    elif expr == "SAD":
                        self.eye_shape, self.mouth_shape, self.curvature = EYE_DOT, MOUTH_CURVE, -1.0
                    elif expr == "EXCITED":
                        self.eye_shape, self.mouth_shape = EYE_CHEVRON, MOUTH_OPEN
                    elif expr == "SLEEP":
                        self.eye_shape, self.mouth_shape = EYE_LINE, MOUTH_FLAT
                    elif expr == "SURPRISED":
                        self.eye_shape, self.mouth_shape = EYE_OPEN, MOUTH_OPEN
                
                if "talking" in msg:
                    self.is_talking = bool(msg["talking"])
                    
                print(f"Update: {msg}")
        except zmq.Again:
            pass

    def run(self):
        while not WindowShouldClose():
            self.check_messages()
            dt = GetFrameTime()
            
            # Blink logic
            self.blink_timer += dt
            if self.blink_timer > 3.0:
                self.is_blinking = True
                if self.blink_timer > 3.15:
                    self.is_blinking = False
                    self.blink_timer = 0.0

            # Drawing
            width = GetScreenWidth()
            height = GetScreenHeight()
            box_w = width * 0.85
            box_h = height * 0.85
            box_x = (width - box_w) / 2
            box_y = (height - box_h) / 2

            BeginDrawing()
            ClearBackground(OUTSIDE_COLOR)
            
            DrawRectangle(int(box_x), int(box_y), int(box_w), int(box_h), FACE_BG_COLOR)
            DrawRectangleLines(int(box_x), int(box_y), int(box_w), int(box_h), BLACK)
            
            self.draw_eyes(box_x, box_y, box_w, box_h)
            self.draw_mouth(width, height)
            self.draw_crt_overlay(box_x, box_y, box_w, box_h)
            
            DrawFPS(10, 10)
            EndDrawing()

        CloseWindow()
        
if __name__ == "__main__":
    face = BMOFace()
    face.run()