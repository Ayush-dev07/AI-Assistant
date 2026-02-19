"""Remove ALSA spam from my device."""
import ctypes
from ctypes.util import find_library

# Load ALSA library
asound = ctypes.CDLL(find_library("asound"))

# Define no-op error handler
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p
)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

# Set error handler
asound.snd_lib_error_set_handler(c_error_handler)

"""Main code starts here."""
import speech_recognition as sr
import os
import re
import datetime
from Speaker import EmotionalTTS
import subprocess
from Context import Context
from Entities.Entity_extractor import APPS

recognizer =sr.Recognizer()
recognizer.pause_threshold = 0.8
recognizer.energy_threshold = 400

mic = sr.Microphone(device_index=13)
tts = EmotionalTTS()

def listen_once():
    with mic as source:
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio).lower()
        print("Heard :", text)
        return text
    except:
        return ""
    
def wait_for_wake():
    from Intents.Intent_predictor import predict_intent
    from Entities.Entity_extractor import extract_entities
    from Intents.Intent_predictor import handled_command
    print("Zero Two is idle. Say the command Darling...")
    while True:
        text = listen_once()
        intent, conf = predict_intent(text)
        entities  = extract_entities(text, intent)
        if intent == "wake_words":
            if conf >= 0.20:
                tts.speak("Yes sir, I'm here to help.")
                return

def volume_up():
    os.system("pactl set-sink-volume @DEFAULT_SINK@ +10%")
    tts.speak("Volume increased, Darling. I hope you can hear better now.")

def volume_down():
    os.system("pactl set-sink-volume @DEFAULT_SINK@ -10%")
    tts.speak("Volume decreased, Darling. I hope you're comfortable now.")

def mute_volume():
    os.system("pactl set-sink-volume @DEFAULT_SINK@ toggle")
    tts.speak("Silence.")

def set_volume(percent):
    os.system(f"pactl set-sink-volume @DEFAULT_SINK@ {percent}%")
    tts.speak(f"Volume set to {percent} percent.")

def brightness_up():
    os.system("brightnessctl set +10%")
    tts.speak("Brightness increased, darling.")

def brightness_down():
    os.system("brightnessctl set 10%-")
    tts.speak("Brightness decreased, darling.")

def set_brightness(percent: int):
    percent = max(1, min(percent, 100))
    subprocess.run(
        ["brightnessctl", "set", f"{percent}%"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    tts.speak(f"Brightness set to {percent} percent.")

def execute_command(intent, entities):

    if intent == "get_time":
        now= datetime.datetime.now().strftime("%I:%M %p")
        tts.speak(f"The time is {now}")
        return True

    if intent == "volume_up":
        volume_up()
        return True
    
    if intent == "decrease_volume":
        volume_down()
        return True
    
    if intent == "mute_volume":
        mute_volume()
        return True

    if intent == "gesture_call":
        from Gesture_control.gesture import GestureRecognizer, SystemController, GestureControlSystem
        system = GestureControlSystem()
        system.run()
        return True
    
    if intent == "set_volume":
        value = entities.get("value")
        if value is not None:
            set_volume(value)
            return True

    if intent == "brightness_up":
        brightness_up()
        return True

    if intent == "brightness_down":
        brightness_down()
        return True
    
    if intent == "set_brightness":
        value = entities.get("value")
        if value is not None:
            set_brightness(value)
            return True
    
    if intent == "get_date":
        today = datetime.date.today().strftime("%B %d, %Y")
        tts.speak(f"Today's date is {today}")
        return True
    if intent == "open_app":
        from auto_app_detection import resolve_and_launch
        app_name = entities.get("app")
        resolve_and_launch(app_name)
        tts.speak(f"Opening {app_name} for you, Darling.")
        return True
    
    if intent == "shutdown_system":
        tts.speak("Are you sure you want to shut down, darling? Please confirm.")
        response = listen_once()
        if "confirm" in response:
            tts.speak("Shutting down, darling. See you again.")
            os.system("shutdown now")
        else:
            tts.speak("Shutdown cancelled.")
        return True
    
    if intent == "restart_system":
        tts.speak("Are you sure you want to restart, darling? Please confirm.")
        response = listen_once()
        if "confirm" in response:
            tts.speak("Restarting the system, darling.")
            os.system("reboot")
        else:
            tts.speak("Restart cancelled.")
        return True
    
    if intent == "sleep_words":
            tts.speak("Okay darling. We'll meet again soon.")
            return False
    
    tts.speak("I don't recognize this command yet, Darling. Please keep teaching me.")
    return True


def command_loop():
    from Intents.Intent_predictor import predict_intent
    from Entities.Entity_extractor import extract_entities
    from Memory import set_memory, learn
    while True:
        command = listen_once()
        intent, conf = predict_intent(command)
        entities  = extract_entities(intent, command)

        handled = execute_command(intent, entities)
        learn(intent, entities)
        if not handled:
            break

def activate():
    while True:
        wait_for_wake()
        command_loop()      
        