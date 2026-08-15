#!/bin/bash
# Waits for the bmo_xorg service's X server to be ready, disables DPMS
# blanking (otherwise the panel sleeps after 10 min idle), then launches
# BMO_Engine. Meant to run as the bmo_face_engine systemd service.

export DISPLAY=:0

for i in $(seq 1 30); do
    [ -S /tmp/.X11-unix/X0 ] && break
    sleep 0.5
done

xset -dpms
xset s off
xset s noblank

cd "/home/bmo/bmo_production/face_engine/BMO Face Engine/build"
exec ./BMO_Engine
