import speech_recognition as sr
import os
import re
import datetime
from Speaker import speak
import subprocess
from Intents.Intent_predictor import predict_intent
from Entities.Entity_extractor import extract_entities
from Intents.Intent_predictor import handled_command

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
    print("Zero Two is idle. Say the command Darling...")
    while True:
        text = listen_once()
        intent, conf = predict_intent(text)
        entities  = extract_entities(text, intent)
        if intent == "wake_words":
            if conf >= 0.23:
                speak("Yes sir, I'm here to help.")
                return
        
wait_for_wake()
        
