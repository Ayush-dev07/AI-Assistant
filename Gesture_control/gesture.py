#!/usr/bin/env python3
"""
Hand Gesture Control System for Linux Mint
FIXED for latest MediaPipe API
Works with MediaPipe 0.10.x versions

Gestures:
- Fist: Close current window/tab
- Swipe Left: Previous application (Alt+Tab)
- Swipe Right: Next application
- Swipe Down: Show desktop (minimize all)
- Swipe Up: Close all windows
"""

import cv2
import numpy as np
import time
import subprocess
import os
import sys
from collections import deque
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import MediaPipe with proper error handling
try:
    import mediapipe as mp
    print(f"✓ MediaPipe {mp.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ Error importing MediaPipe: {e}")
    print("\nPlease install MediaPipe:")
    print("  pip3 install --break-system-packages mediapipe")
    sys.exit(1)


class GestureRecognizer:
    """Hand gesture recognition using MediaPipe"""
    
    def __init__(self):
        """Initialize MediaPipe Hands with latest API"""
        try:
            # Initialize MediaPipe Hands using the correct API
            self.mp_hands = mp.tasks.vision.HandLandmarker
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Create HandLandmarker options
            base_options = mp.tasks.BaseOptions(
                model_asset_path=self._download_hand_model()
            )
            
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            # Create the hand landmarker
            self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
            
        except Exception as e:
            print(f"Warning: Could not initialize new MediaPipe API: {e}")
            print("Falling back to legacy API...")
            self._init_legacy_api()
        
        # Gesture tracking
        self.gesture_history = deque(maxlen=10)
        self.last_gesture = None
        self.last_action_time = 0
        self.cooldown = 1.5
        
        # Movement tracking for swipes
        self.hand_positions = deque(maxlen=15)
        
        self.use_legacy = False
    
    def _download_hand_model(self):
        """Download hand landmarker model if needed"""
        model_path = os.path.expanduser("~/.mediapipe/hand_landmarker.task")
        
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download model
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            
            try:
                print("Downloading hand detection model...")
                urllib.request.urlretrieve(model_url, model_path)
                print("✓ Model downloaded")
            except Exception as e:
                print(f"Warning: Could not download model: {e}")
                raise
        
        return model_path
    
    def _init_legacy_api(self):
        """Initialize using legacy MediaPipe API (for older versions)"""
        try:
            self.mp_hands_legacy = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.hands = self.mp_hands_legacy.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            
            self.use_legacy = True
            print("✓ Using legacy MediaPipe API")
            
        except Exception as e:
            print(f"❌ Error: Could not initialize MediaPipe: {e}")
            print("\nPlease reinstall MediaPipe:")
            print("  pip3 uninstall mediapipe")
            print("  pip3 install --break-system-packages mediapipe")
            sys.exit(1)
    
    def detect_fist(self, hand_landmarks):
        """Detect closed fist gesture"""
        # Get landmark positions (works with both APIs)
        if hasattr(hand_landmarks, 'landmark'):
            landmarks = hand_landmarks.landmark
        else:
            landmarks = hand_landmarks
        
        # Get fingertip and base positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Get palm base
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        
        # Calculate distances from palm
        def distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        # All fingertips should be close to palm for fist
        thumb_closed = distance(thumb_tip, index_mcp) < 0.1
        index_closed = distance(index_tip, wrist) < 0.15
        middle_closed = distance(middle_tip, wrist) < 0.15
        ring_closed = distance(ring_tip, wrist) < 0.15
        pinky_closed = distance(pinky_tip, wrist) < 0.15
        
        # At least 4 fingers should be closed
        closed_count = sum([thumb_closed, index_closed, middle_closed, ring_closed, pinky_closed])
        
        return closed_count >= 4
    
    def detect_open_hand(self, hand_landmarks):
        """Detect open hand (all fingers extended)"""
        if hasattr(hand_landmarks, 'landmark'):
            landmarks = hand_landmarks.landmark
        else:
            landmarks = hand_landmarks
        
        # Check if all fingers are extended
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        wrist = landmarks[0]
        
        def distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        # All fingertips should be far from palm
        index_extended = distance(index_tip, wrist) > 0.2
        middle_extended = distance(middle_tip, wrist) > 0.2
        ring_extended = distance(ring_tip, wrist) > 0.2
        pinky_extended = distance(pinky_tip, wrist) > 0.2
        
        extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
        
        return extended_count >= 3
    
    def detect_swipe(self):
        """Detect swipe direction from hand movement"""
        if len(self.hand_positions) < 10:
            return None
        
        # Get start and end positions
        start_pos = self.hand_positions[0]
        end_pos = self.hand_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Minimum movement threshold
        min_movement = 0.15
        
        # Check for horizontal swipe
        if abs(dx) > abs(dy) and abs(dx) > min_movement:
            if dx > 0:
                return "swipe_right"
            else:
                return "swipe_left"
        
        # Check for vertical swipe
        elif abs(dy) > abs(dx) and abs(dy) > min_movement:
            if dy > 0:
                return "swipe_down"
            else:
                return "swipe_up"
        
        return None
    
    def process_frame_legacy(self, frame):
        """Process frame with legacy API"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands_legacy.HAND_CONNECTIONS
                )
                
                # Get palm center for swipe detection
                palm_center = hand_landmarks.landmark[9]
                self.hand_positions.append((palm_center.x, palm_center.y))
                
                # Detect fist
                if self.detect_fist(hand_landmarks):
                    gesture = "fist"
                    self.hand_positions.clear()
                
                # Detect open hand for swipes
                elif self.detect_open_hand(hand_landmarks):
                    swipe = self.detect_swipe()
                    if swipe:
                        gesture = swipe
                        self.hand_positions.clear()
        else:
            self.hand_positions.clear()
        
        return frame, gesture
    
    def process_frame_new(self, frame):
        """Process frame with new API"""
        # Convert frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect hands
        detection_result = self.detector.detect(mp_image)
        
        gesture = None
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw landmarks manually
                for idx, landmark in enumerate(hand_landmarks):
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Get palm center for swipe detection
                palm_center = hand_landmarks[9]
                self.hand_positions.append((palm_center.x, palm_center.y))
                
                # Detect fist
                if self.detect_fist(hand_landmarks):
                    gesture = "fist"
                    self.hand_positions.clear()
                
                # Detect open hand for swipes
                elif self.detect_open_hand(hand_landmarks):
                    swipe = self.detect_swipe()
                    if swipe:
                        gesture = swipe
                        self.hand_positions.clear()
        else:
            self.hand_positions.clear()
        
        return frame, gesture
    
    def process_frame(self, frame):
        """Process frame and detect gestures"""
        if self.use_legacy:
            return self.process_frame_legacy(frame)
        else:
            try:
                return self.process_frame_new(frame)
            except:
                # Fallback to legacy if new API fails
                if not self.use_legacy:
                    self._init_legacy_api()
                return self.process_frame_legacy(frame)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.use_legacy and hasattr(self, 'hands'):
            self.hands.close()
        elif hasattr(self, 'detector'):
            self.detector.close()


class SystemController:
    """Control Linux Mint system with gestures"""
    
    def __init__(self):
        self.desktop_environment = self.detect_desktop_environment()
        print(f"✓ Detected desktop environment: {self.desktop_environment}")
    
    def detect_desktop_environment(self):
        """Detect which desktop environment is running"""
        desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
        
        if 'cinnamon' in desktop:
            return 'cinnamon'
        elif 'mate' in desktop:
            return 'mate'
        elif 'xfce' in desktop:
            return 'xfce'
        elif 'gnome' in desktop:
            return 'gnome'
        else:
            return 'cinnamon'
    
    def execute_command(self, command):
        """Execute shell command safely"""
        try:
            subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            return True
        except:
            return False
    
    def close_window(self):
        """Close current window/tab"""
        print("→ Gesture: FIST - Closing current window")
        
        commands = [
            "xdotool getactivewindow windowkill",
            "wmctrl -c :ACTIVE:",
            "xdotool key alt+F4",
        ]
        
        for cmd in commands:
            if self.execute_command(cmd):
                return True
        return False
    
    def previous_application(self):
        """Switch to previous application"""
        print("→ Gesture: SWIPE LEFT - Previous application")
        self.execute_command("xdotool key alt+Tab")
        return True
    
    def open_terminal(self):
        """Open Terminal with open palm gesture."""
        print("→ Gesture: OPEN PALM - Open Terminal")
        self.execute_command("xdotool key ctrl+alt+t")
        return True
    
    def next_application(self):
        """Switch to next application"""
        print("→ Gesture: SWIPE RIGHT - Next application")
        self.execute_command("xdotool key alt+shift+Tab")
        return True
    
    def show_desktop(self):
        """Show desktop (minimize all windows)"""
        print("→ Gesture: SWIPE DOWN - Show desktop")
        
        commands = {
            'cinnamon': "xdotool key super+d",
            'mate': "xdotool key super+d",
            'xfce': "xdotool key ctrl+alt+d",
            'gnome': "xdotool key super+d",
        }
        
        cmd = commands.get(self.desktop_environment, "xdotool key super+d")
        self.execute_command(cmd)
        return True
    


class GestureControlSystem:
    """Main gesture control system"""
    
    def __init__(self):
        print("="*60)
        print("Hand Gesture Control System")
        print("="*60)
        print()
        
        self.recognizer = GestureRecognizer()
        self.controller = SystemController()
        self.running = False
        self.last_gesture_time = 0
        self.cooldown = 1.5
    
    def execute_gesture(self, gesture):
        """Execute action for recognized gesture"""
        current_time = time.time()
        
        if current_time - self.last_gesture_time < self.cooldown:
            return
        
        if gesture == "fist":
            self.controller.close_window()
            self.last_gesture_time = current_time
        
        elif gesture == "swipe_left":
            self.controller.previous_application()
            self.last_gesture_time = current_time
        
        elif gesture == "swipe_right":
            self.controller.next_application()
            self.last_gesture_time = current_time
        
        elif gesture == "swipe_down":
            self.controller.show_desktop()
            self.last_gesture_time = current_time
        
        elif gesture == "swipe_up":
            self.controller.open_terminal()
            self.last_gesture_time = current_time
    
    def run(self):
        """Run the gesture control system"""
        print("🎥 Initializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera initialized")
        print()
        print("="*60)
        print("Gesture Controls:")
        print("="*60)
        print("  🤜 FIST           → Close current window/tab")
        print("  👈 SWIPE LEFT     → Previous application")
        print("  👉 SWIPE RIGHT    → Next application")
        print("  👇 SWIPE DOWN     → Show desktop")
        print("  👆 SWIPE UP       → Open terminal")
        print()
        print("Press 'Q' to quit")
        print("="*60)
        print()
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                processed_frame, gesture = self.recognizer.process_frame(frame)
                
                if gesture:
                    self.execute_gesture(gesture)
                
                cv2.putText(processed_frame, "Hand Gesture Control", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Press 'Q' to quit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if gesture:
                    gesture_text = gesture.replace("_", " ").upper()
                    cv2.putText(processed_frame, f"Gesture: {gesture_text}", (10, 450),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Hand Gesture Control', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nExiting...")
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.recognizer.cleanup()
            print("✓ Cleanup complete")


def main():
    """Main function"""
    system = GestureControlSystem()
    system.run()


if __name__ == "__main__":
    main()