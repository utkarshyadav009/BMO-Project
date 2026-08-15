#!/bin/bash
# Streams the CSI (IMX219) camera live to the attached display.
# Usage: ./stream_camera.sh [width] [height]
#   Defaults to 1280x720 capture, scaled to fit the 800x480 panel.

set -e

DISPLAY="${DISPLAY:-:0}"
CAP_W="${1:-1280}"
CAP_H="${2:-720}"
OUT_W=800
OUT_H=480

export DISPLAY

# nvvidconv preserves source aspect ratio rather than stretching to the
# output caps, so a straight 1280x720 (16:9) -> 800x480 (5:3) scale leaves a
# black letterbox bar. Crop the sides down to a 5:3 frame first (matching the
# panel's aspect) so the scale below fills 800x480 exactly with no distortion.
CROP_W=$(( CAP_H * OUT_W / OUT_H ))
CROP_SIDE=$(( (CAP_W - CROP_W) / 2 ))
# nvvidconv's left/top/right/bottom are absolute bounding-box edges (not
# trim amounts), so "right" is the x-coordinate of the crop's right edge.
CROP_RIGHT=$(( CAP_W - CROP_SIDE ))

# flip-method=2 (180 deg): the panel itself is physically mounted upside
# down, so xorg.conf carries a blanket 180-deg rotation on the whole DP-1
# output to compensate. That's correct for drawn content like the face
# engine (no inherent "up"), but it also flips the camera feed, which
# already shows the real world right-side-up -- so we pre-rotate it 180
# here to cancel the display's compensation back out.
exec gst-launch-1.0 -e nvarguscamerasrc sensor-id=0 ! \
  "video/x-raw(memory:NVMM),width=${CAP_W},height=${CAP_H},framerate=30/1" ! \
  nvvidconv left="${CROP_SIDE}" right="${CROP_RIGHT}" top=0 bottom="${CAP_H}" flip-method=2 ! \
  "video/x-raw,width=${OUT_W},height=${OUT_H},pixel-aspect-ratio=1/1" ! \
  xvimagesink sync=false force-aspect-ratio=false
