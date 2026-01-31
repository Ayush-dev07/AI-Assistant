import cv2
import mediapipe as mp
import subprocess
import time
import os

os.environ["ALSA_LOG_LEVEL"] = "none"

# ----------------------
# LOW-END CAMERA SETTINGS
# ----------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 20)

# ----------------------
# MEDIAPIPE HANDS SETUP (0.10.32 SAFE)
# ----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # FASTEST MODEL
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ----------------------
# PERFORMANCE VARIABLES
# ----------------------
prev_x = None
prev_y = None
last_action = 0
cooldown = 0.8  # seconds

# ----------------------
# COMMAND EXECUTOR (LINUX)
# ----------------------
def run_command(keys):
    subprocess.run(["xdotool", "key", keys])

# ----------------------
# ACTION RATE LIMITER
# ----------------------
def can_trigger():
    global last_action
    now = time.time()
    if now - last_action > cooldown:
        last_action = now
        return True
    return False

# ----------------------
# CLOSED FIST DETECTOR
# ----------------------
def is_fist(lm):
    folded = 0
    tips = [8, 12, 16, 20]
    for tip in tips:
        if lm[tip].y > lm[tip - 2].y:
            folded += 1
    return folded >= 3

# ----------------------
# SWIPE DETECTOR
# ----------------------
def detect_swipe(wrist):
    global prev_x, prev_y

    if prev_x is None:
        prev_x = wrist.x
        prev_y = wrist.y
        return None

    dx = wrist.x - prev_x
    dy = wrist.y - prev_y

    prev_x = wrist.x
    prev_y = wrist.y

    threshold = 0.05

    if dx > threshold:
        return "RIGHT"
    if dx < -threshold:
        return "LEFT"
    if dy > threshold:
        return "DOWN"
    if dy < -threshold:
        return "UP"

    return None

# ----------------------
# MAIN LOOP
# ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        wrist = lm[0]

        # FIST → CLOSE WINDOW
        if is_fist(lm) and can_trigger():
            run_command("Alt+F4")

        # SWIPES
        swipe = detect_swipe(wrist)
        if swipe and can_trigger():
            if swipe == "LEFT":
                run_command("Alt+Tab")

            elif swipe == "RIGHT":
                run_command("Alt+Shift+Tab")

            elif swipe == "DOWN":
                run_command("Super_L+Down")

            elif swipe == "UP":
                run_command("Ctrl+w")

    cv2.imshow("Gesture Control (Low-End)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
