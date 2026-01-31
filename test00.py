import asyncio
import edge_tts
import subprocess
import queue
import os 
import threading

os.environ["ALSA_LOG_LEVEL"] = "none"


VOICE = "en-US-JennyNeural"

RATE = "-8%"     # slower = natural
PITCH = "+4Hz"   # subtle warmth

audio_queue = queue.Queue()

import edge_tts
import subprocess
import queue
import threading

VOICE = "en-US-JennyNeural"

EMOTION_PROFILES = {
    "neutral": {"rate": "+0%", "pitch": "+0Hz"},
    "calm": {"rate": "-8%", "pitch": "-1Hz"},
    "friendly": {"rate": "-3%", "pitch": "+2Hz"},
    "cheerful": {"rate": "+6%", "pitch": "+3Hz"},
    "serious": {"rate": "-6%", "pitch": "-2Hz"},
    "urgent": {"rate": "+10%", "pitch": "+4Hz"},
    "sad": {"rate": "-10%", "pitch": "-3Hz"},
    "angry": {"rate": "+8%", "pitch": "+5Hz"}
}

audio_queue = queue.Queue()

def audio_player():
    while True:
        audio_bytes = audio_queue.get()
        if audio_bytes is None:
            break

        player = subprocess.Popen(
            ["ffplay", "-f", "mp3", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
            stdin=subprocess.PIPE
        )

        try:
            player.stdin.write(audio_bytes)
            player.stdin.close()
            player.wait()
        except:
            pass


threading.Thread(target=audio_player, daemon=True).start()


async def generate_tts(text, emotion="neutral"):
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])

    rate = profile["rate"]
    pitch = profile["pitch"]

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=rate,
        pitch=pitch
    )

    buffer = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.extend(chunk["data"])

    audio_queue.put(bytes(buffer))


def speak(text, emotion="neutral"):
    asyncio.run(generate_tts(text, emotion))


def audio_player():
    while True:
        audio_bytes = audio_queue.get()
        if audio_bytes is None:
            break

        player = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        )

        try:
            player.stdin.write(audio_bytes)
            player.stdin.close()
            player.wait()
        except:
            pass


threading.Thread(target=audio_player, daemon=True).start()


async def generate_tts(text):
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        pitch=PITCH
    )

    buffer = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.extend(chunk["data"])

    audio_queue.put(bytes(buffer))


def speak(text):
    asyncio.run(generate_tts(text))
