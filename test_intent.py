import cv2
import numpy as np
import subprocess
import time
import os

os.environ["ALSA_LOG_LEVEL"] = "none"

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 360)
cap.set(cv2.CAP_PROP_FPS, 20)

prev_center = None
last_action = 0
cooldown = 0.8

def can_trigger():
    global last_action
    now = time.time()
    if now - last_action > cooldown:
        last_action = now
        return True
    return False

def run_command(keys):
    subprocess.run(["xdotool", "key", keys])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range (adjustable)
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 3000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)

                cv2.circle(frame, center, 5, (0,255,0), -1)

                # Motion tracking
                if prev_center:
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]

                    threshold = 25

                    if abs(dx) > threshold and can_trigger():
                        if dx > 0:
                            run_command("Alt+Shift+Tab")
                        else:
                            run_command("Alt+Tab")

                    if abs(dy) > threshold and can_trigger():
                        if dy > 0:
                            run_command("Super_L+Down")
                        else:
                            run_command("Ctrl+w")

                prev_center = center

                # Closed fist detection (compact contour)
                perimeter = cv2.arcLength(cnt, True)
                compactness = (perimeter * perimeter) / (4 * np.pi * area)

                if compactness < 25 and can_trigger():
                    run_command("Alt+F4")

    cv2.imshow("Ultra-Light Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
