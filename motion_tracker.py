#!/usr/bin/env python3
"""
Basic motion tracking on the CSI camera (IMX219, sensor-id=0 / CAM0).

Deliberately simple and light on RAM: no ML model, just consecutive-frame
differencing at a small capture resolution.
  1. Grab a small (320x240) grayscale frame.
  2. Blur it slightly to ignore sensor noise.
  3. Diff against the previous frame, threshold, and find contours.
  4. Draw a box around any contour big enough to count as real motion.

Total working memory is a handful of 320x240 buffers (~75KB each) -- this
is not a background-model tracker (no accumulating history), so RAM use
stays flat over time.
"""
import cv2

# Same flip as stream_camera.sh: the panel is physically mounted upside
# down, and xorg.conf carries a blanket 180-degree rotation on the whole
# display to compensate. That's fine for drawn content but flips the
# camera's already-correct view of the real world, so pre-rotate it back.
GST_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
    "nvvidconv flip-method=2 ! "
    "video/x-raw,width=320,height=240,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1"
)

MIN_MOTION_AREA = 250  # pixels, at 320x240 -- filters out sensor-noise specks


def main():
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open camera pipeline")
        return

    cv2.namedWindow("Motion Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion Tracker", 800, 480)

    prev_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < MIN_MOTION_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        prev_gray = gray

        cv2.imshow("Motion Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
