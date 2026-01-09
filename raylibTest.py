import raylib as rl
import math

# --------------------------------------------------
# Initial Window size (can be resized)
# --------------------------------------------------
rl.InitWindow(800, 450, b"BMO Advanced Face (Resizable)")
rl.SetTargetFPS(60)

# Enable window resizing (if supported)
rl.SetWindowState(rl.FLAG_WINDOW_RESIZABLE)

# --------------------------------------------------
# Colors
# --------------------------------------------------
OUTSIDE_COLOR = rl.GetColor(0xA3D9C1FF)  # outside face box
FACE_BG_COLOR = rl.GetColor(0xC7E1C4FF)  # inside face box
BLACK        = rl.BLACK
WHITE        = rl.WHITE
SCANLINE     = rl.GetColor(0x00000022)
VIGNETTE     = rl.GetColor(0x00000055)

# --------------------------------------------------
# Emotion values (0.0 - 1.0)
# --------------------------------------------------
emotion_target = 0.0
emotion_value  = 0.0   # blended value

# --------------------------------------------------
# Animation state
# --------------------------------------------------
blink_timer = 0.0
is_blinking = False
talk_time   = 0.0

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def lerp(a, b, t):
    return a + (b - a) * t

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def lerp_point(p1, p2, t):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))

def draw_quadratic_bezier(p0, p1, p2, color, segments=20):
    """Draws a quadratic Bezier curve by connecting multiple lines."""
    prev_x, prev_y = p0
    for i in range(1, segments + 1):
        t = i / segments
        # Quadratic Bezier formula
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        rl.DrawLine(int(prev_x), int(prev_y), int(x), int(y), color)
        prev_x, prev_y = x, y


# Mouth curve points relative to window size
def get_mouth_points(width, height, emotion, talking):
    # Base position
    cx = width * 0.5
    cy = height * 0.55

    # Base mouth width and height
    mouth_w = width * 0.15
    base_h = lerp(height * 0.01, height * 0.05, emotion)

    # Talking movement
    if talking:
        base_h += math.sin(rl.GetTime() * 12) * height * 0.02

    # Control points for top curve
    top_left  = (cx - mouth_w/2, cy)
    top_mid   = (cx, cy - base_h)
    top_right = (cx + mouth_w/2, cy)

    # Control points for bottom curve
    bottom_left  = (cx - mouth_w/2, cy)
    bottom_mid   = (cx, cy + base_h/2)
    bottom_right = (cx + mouth_w/2, cy)

    return top_left, top_mid, top_right, bottom_left, bottom_mid, bottom_right


# --------------------------------------------------
# Drawing functions
# --------------------------------------------------
def draw_eyes(blinking, box_x, box_y, box_w, box_h):
    left_eye_x  = box_x + box_w * 0.35
    right_eye_x = box_x + box_w * 0.65
    eye_y       = box_y + box_h * 0.35

    EYE_RADIUS = max(5, int(box_w * 0.06))
    PUPIL_RADIUS = max(2, int(box_w * 0.025))
    MAX_PUPIL_OFFSET = max(3, int(box_w * 0.035))

    mx = rl.GetMouseX()
    my = rl.GetMouseY()

    for eye_x in [left_eye_x, right_eye_x]:
        if blinking:
            rl.DrawRectangle(int(eye_x - EYE_RADIUS - 4), int(eye_y), EYE_RADIUS * 2 + 8, int(EYE_RADIUS * 0.6), BLACK)
            continue

        # Eye white (outline)
        rl.DrawCircle(int(eye_x), int(eye_y), EYE_RADIUS, BLACK)

        # Pupil tracking
        dx = clamp(mx - eye_x, -30, 30)
        dy = clamp(my - eye_y, -30, 30)

        px = eye_x + (dx / 30) * MAX_PUPIL_OFFSET
        py = eye_y + (dy / 30) * MAX_PUPIL_OFFSET

        rl.DrawCircle(int(px), int(py), PUPIL_RADIUS, WHITE)

def draw_mouth_curve(emotion, talking, width, height):
    cx = width * 0.5
    cy = height * 0.55

    mouth_w = width * 0.15
    base_h = lerp(height * 0.01, height * 0.05, emotion)

    if talking:
        base_h += math.sin(rl.GetTime() * 12) * height * 0.02

    # Quadratic Bezier points
    top_left  = (cx - mouth_w/2, cy)
    top_mid   = (cx, cy - base_h)
    top_right = (cx + mouth_w/2, cy)

    bottom_left  = (cx - mouth_w/2, cy)
    bottom_mid   = (cx, cy + base_h/2)
    bottom_right = (cx + mouth_w/2, cy)

    # Draw curves
    draw_quadratic_bezier(top_left, top_mid, top_right, BLACK)
    draw_quadratic_bezier(bottom_left, bottom_mid, bottom_right, BLACK)

    # Fill mouth interior (simple horizontal lines)
    steps = 8
    for i in range(steps + 1):
        t = i / steps
        x_start = lerp(top_left[0], bottom_left[0], t)
        y_start = lerp(top_left[1], bottom_left[1], t)
        x_end   = lerp(top_right[0], bottom_right[0], t)
        y_end   = lerp(top_right[1], bottom_right[1], t)
        rl.DrawLine(int(x_start), int(y_start), int(x_end), int(y_end), WHITE)


def draw_crt(width, height, box_x, box_y, box_w, box_h):
    # Draw scanlines only inside the face box
    line_spacing = max(3, int(box_h * 0.015))
    for y in range(int(box_y), int(box_y + box_h), line_spacing):
        rl.DrawRectangle(int(box_x), y, int(box_w), 1, SCANLINE)

    # Vignette thickness inside the box
    vignette_thickness = max(20, int(min(box_w, box_h) * 0.1))

    # Draw vignette rectangles carefully inside the box
    rl.DrawRectangle(int(box_x), int(box_y), int(box_w), vignette_thickness, VIGNETTE)  # top
    rl.DrawRectangle(int(box_x), int(box_y + box_h - vignette_thickness), int(box_w), vignette_thickness, VIGNETTE)  # bottom
    rl.DrawRectangle(int(box_x), int(box_y), vignette_thickness, int(box_h), VIGNETTE)  # left
    rl.DrawRectangle(int(box_x + box_w - vignette_thickness), int(box_y), vignette_thickness, int(box_h), VIGNETTE)  # right

# --------------------------------------------------
# Main loop
# --------------------------------------------------
while not rl.WindowShouldClose():

    dt = rl.GetFrameTime()
    talk_time += dt

    width = rl.GetScreenWidth()
    height = rl.GetScreenHeight()

    # Define face box (centered, aspect ratio ~16:9)
    box_width = width * 0.7
    box_height = height * 0.7
    box_x = (width - box_width) / 2
    box_y = (height - box_height) / 2

    # ---- Blink ----
    blink_timer += dt
    if blink_timer > 3.0:
        is_blinking = True
        if blink_timer > 3.15:
            is_blinking = False
            blink_timer = 0.0

    # ---- Input (emotion blending) ----
    if rl.IsKeyPressed(rl.KEY_ONE):
        emotion_target = 0.0
    elif rl.IsKeyPressed(rl.KEY_TWO):
        emotion_target = 1.0

    # ---- Smooth emotion blending ----
    emotion_value = lerp(emotion_value, emotion_target, 5 * dt)

    # ---- Talking ----
    talking = rl.IsKeyDown(rl.KEY_SPACE)

    # ---- Draw ----
    rl.BeginDrawing()

    # Draw outside background
    rl.ClearBackground(OUTSIDE_COLOR)

    # Draw face box background
    rl.DrawRectangle(int(box_x), int(box_y), int(box_width), int(box_height), FACE_BG_COLOR)

    # Draw black border around the face box
    border_thickness = max(3, int(min(box_width, box_height) * 0.015))
    rl.DrawRectangleLines(int(box_x), int(box_y), int(box_width), int(box_height), BLACK)

    # Draw face elements inside the box
    draw_eyes(is_blinking, box_x, box_y, box_width, box_height)
    draw_mouth_curve(emotion_value, talking, width, height)


    # Draw CRT effect inside the face box only
    draw_crt(width, height, box_x, box_y, box_width, box_height)

    # Instructions text
    rl.DrawText(b"Resize window, 1=Neutral, 2=Happy, SPACE=Talk", 10, 10, max(12, int(height*0.04)), BLACK)

    rl.EndDrawing()

rl.CloseWindow()
