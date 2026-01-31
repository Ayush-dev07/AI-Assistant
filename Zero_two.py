import speech_recognition as sr
import os
import re
import datetime
from Speaker import speak
import subprocess
from Context import Context
from Entities.Entity_extractor import APPS

recognizer =sr.Recognizer()
recognizer.pause_threshold = 0.8
recognizer.energy_threshold = 400

mic = sr.Microphone()

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
                speak("Yes sir, I'm here to help.")
                return

def volume_up():
    os.system("pactl set-sink-volume @DEFAULT_SINK@ +10%")
    speak("Volume increased, Darling. I hope you can hear better now.")

def volume_down():
    os.system("pactl set-sink-volume @DEFAULT_SINK@ -10%")
    speak("Volume decreased, Darling. I hope you're comfortable now.")

def mute_volume():
    os.system("pactl set-sink-volume @DEFAULT_SINK@ toggle")
    speak("Silence.")

def set_volume(percent):
    os.system(f"pactl set-sink-volume @DEFAULT_SINK@ {percent}%")
    speak(f"Volume set to {percent} percent.")

def brightness_up():
    os.system("brightnessctl set +10%")
    speak("Brightness increased, darling.")

def brightness_down():
    os.system("brightnessctl set 10%-")
    speak("Brightness decreased, darling.")

def set_brightness(percent: int):
    percent = max(1, min(percent, 100))
    subprocess.run(
        ["brightnessctl", "set", f"{percent}%"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    speak(f"Brightness set to {percent} percent.")

def execute_command(intent, entities):

    if intent == "get_time":
        now= datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The time is {now}")
        return True

    if intent == "volume_up":
        volume_up()
        return True
    
    if intent == "decrease volume":
        volume_down()
        return True
    
    if intent == "mute volume":
        mute_volume()
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
        speak(f"Today's date is {today}")
        return True
    if intent == "open_app":
        from auto_app_detection import resolve_and_launch
        resolve_and_launch()
        app_name = entities.get("app")
        app = APPS.get(app_name)
        if not app:
            speak(f"I can't find {app_name} on your system, Darling.")
            return True
        os.system(f"{app['exec']} &")
        speak(f"Opening {app_name} for you, Darling.")
        return True
    
    if intent == "shutdown_system":
        speak("Are you sure you want to shut down, darling? Please confirm.")
        response = listen_once()
        if "confirm" in response:
            speak("Shutting down, darling. See you again.")
            os.system("shutdown now")
        else:
            speak("Shutdown cancelled.")
        return True
    
    if intent == "restart_system":
        speak("Are you sure you want to restart, darling? Please confirm.")
        response = listen_once()
        if "confirm" in response:
            speak("Restarting the system, darling.")
            os.system("reboot")
        else:
            speak("Restart cancelled.")
        return True
    
    if intent == "sleep_words":
            speak("Okay darling. We'll meet again soon.")
            return False
    
    speak("I don't recognize this command yet, Darling. Please keep teaching me.")
    return True


def command_loop():
    from Intents.Intent_predictor import predict_intent
    from Entities.Entity_extractor import extract_entities
    from Memory import set_memory
    while True:
        command = listen_once()
        intent, conf = predict_intent(command)
        entities  = extract_entities(command)

        handled = execute_command(intent, entities)
        if not handled:
            break

def activate():
    while True:
        wait_for_wake()
        command_loop()      
        