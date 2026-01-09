import zmq
import time
import random

# --- SETUP ---
context = zmq.Context()
socket = context.socket(zmq.PUB)
# The Brain "Binds" (broadcasts), the Face "Connects" (listens)
socket.bind("tcp://127.0.0.1:5555") 

print("BMO Brain Simulator Started... (Press Ctrl+C to stop)")
print("Ensure Test3.py is running!")

try:
    while True:
        # 1. EXCITED (Chevrons > < and Talking)
        print("State: EXCITED (Talking)")
        socket.send_json({
            "expression": "EXCITED", 
            "talking": True
        })
        time.sleep(3)
        
        # 2. HAPPY (Standard Dot Eyes, Smiling, Not talking)
        print("State: HAPPY (Silent)")
        socket.send_json({
            "expression": "HAPPY", 
            "talking": False
        })
        time.sleep(3)
        
        # 3. SAD (Standard Dot Eyes, Frowning)
        # We use "SAD" instead of -0.5 because our Face script 
        # maps "SAD" directly to curvature -1.0
        print("State: SAD")
        socket.send_json({
            "expression": "SAD", 
            "talking": False
        })
        time.sleep(3)

        # 4. SURPRISED (Hollow Eyes O O)
        print("State: SURPRISED")
        socket.send_json({
            "expression": "SURPRISED", 
            "talking": False
        })
        time.sleep(2)

        # 5. SLEEP (Line Eyes - -)
        print("State: SLEEP")
        socket.send_json({
            "expression": "SLEEP", 
            "talking": False
        })
        time.sleep(3)

except KeyboardInterrupt:
    print("\nDisconnecting Brain...")
    socket.close()
    context.term()