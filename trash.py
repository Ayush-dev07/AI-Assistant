import os
import sys
import warnings
import subprocess
import tempfile
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Callable, List
import numpy as np

# Suppress ALL warnings and spam
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect stderr to suppress ALSA/JACK/PortAudio spam
class SuppressStream:
    def __init__(self, stream=sys.stderr):
        self.stream = stream
        self.fd = stream.fileno()
        
    def __enter__(self):
        self.old_fd = os.dup(self.fd)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, self.fd)
        return self
        
    def __exit__(self, *args):
        os.dup2(self.old_fd, self.fd)
        os.close(self.old_fd)
        os.close(self.devnull)


class AudioDeviceManager:
    
    def __init__(self):
        self.device_index = None
        self.device_name = None
        self.sample_rate = 16000  # Optimal for Whisper
        self.channels = 1
        
    def detect_microphone(self) -> bool:
        
        with SuppressStream():
            try:
                import pyaudio
                
                audio = pyaudio.PyAudio()
                
                # Get all input devices
                devices = []
                for i in range(audio.get_device_count()):
                    try:
                        info = audio.get_device_info_by_index(i)
                        if info['maxInputChannels'] > 0:
                            devices.append({
                                'index': i,
                                'name': info['name'].lower(),
                                'channels': info['maxInputChannels'],
                                'sample_rate': int(info['defaultSampleRate'])
                            })
                    except:
                        continue
                
                audio.terminate()
                
                if not devices:
                    return False
                
                # Priority selection
                # 1. Soundcard (ALSA, PCI, USB)
                for dev in devices:
                    if any(x in dev['name'] for x in ['alsa', 'pci', 'usb', 'hw:']):
                        self.device_index = dev['index']
                        self.device_name = dev['name']
                        return True
                
                # 2. Bluetooth
                for dev in devices:
                    if any(x in dev['name'] for x in ['blue', 'bt', 'headset']):
                        self.device_index = dev['index']
                        self.device_name = dev['name']
                        return True
                
                # 3. Default (first available)
                self.device_index = devices[0]['index']
                self.device_name = devices[0]['name']
                return True
                
            except Exception as e:
                return False
    
    def get_device_info(self) -> dict:
       
        return {
            'index': self.device_index,
            'name': self.device_name,
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }


class VoiceActivityDetector:
    
    def __init__(self, threshold: float = 0.015):
        self.threshold = threshold
        
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        
        if len(audio_chunk) == 0:
            return False
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        return energy > self.threshold


class SpeechToText:
   
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(SpeechToText, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_size: str = "base"):
        
        if self._initialized:
            return
        
        self.model_size = model_size
        self.model = None
        self.audio_manager = AudioDeviceManager()
        self.vad = VoiceActivityDetector()
        self.temp_dir = tempfile.mkdtemp(prefix="stt_")
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        
        # Interruption callback
        self.on_speech_start = None  # Called when user starts speaking
        
        # Initialize
        self._load_model()
        self._setup_microphone()
        
        self._initialized = True
    
    def _load_model(self):
       
        print(f"Loading Whisper '{self.model_size}' model...")
        
        with SuppressStream():
            try:
                import whisper
                self.model = whisper.load_model(self.model_size)
                print(f"✓ Model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                raise
    
    def _setup_microphone(self):
       
        print("Detecting microphone...")
        
        if self.audio_manager.detect_microphone():
            info = self.audio_manager.get_device_info()
            print(f"✓ Microphone detected: {info['name']}")
            print(f"  Device index: {info['index']}")
            print(f"  Sample rate: {info['sample_rate']} Hz")
        else:
            print("✗ No microphone detected")
            print("  Please check your audio input devices")
    
    def transcribe_file(self, audio_file: str, language: str = None) -> dict:
       
        if not os.path.exists(audio_file):
            return {"error": "File not found", "text": ""}
        
        with SuppressStream():
            try:
                result = self.model.transcribe(
                    audio_file,
                    language=language,
                    fp16=False  # CPU compatibility
                )
                
                return {
                    "text": result["text"].strip(),
                    "language": result.get("language", "unknown"),
                    "segments": result.get("segments", []),
                    "success": True
                }
                
            except Exception as e:
                return {
                    "error": str(e),
                    "text": "",
                    "success": False
                }
    
    def _record_audio(self, duration: float):
       
        with SuppressStream():
            try:
                import pyaudio
                import wave
                
                audio = pyaudio.PyAudio()
                
                # Open stream
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=self.audio_manager.channels,
                    rate=self.audio_manager.sample_rate,
                    input=True,
                    input_device_index=self.audio_manager.device_index,
                    frames_per_buffer=1024
                )
                
                print("Recording...")
                frames = []
                
                for _ in range(0, int(self.audio_manager.sample_rate / 1024 * duration)):
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(data)
                
                print("Recording complete")
                
                # Stop stream
                stream.stop_stream()
                stream.close()
                audio.terminate()
                
                # Save to file
                output_file = os.path.join(self.temp_dir, f"recording_{int(time.time())}.wav")
                
                wf = wave.open(output_file, 'wb')
                wf.setnchannels(self.audio_manager.channels)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.audio_manager.sample_rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                return output_file
                
            except Exception as e:
                print(f"Recording error: {e}")
                return None
    
    def listen_once(self, duration: float = 5.0, language: str = None) -> str:
    
        audio_file = self._record_audio(duration)
        
        if not audio_file:
            return ""
        
        result = self.transcribe_file(audio_file, language)
        
        # Cleanup
        try:
            os.remove(audio_file)
        except:
            pass
        
        return result.get("text", "")
    
    def _continuous_recording_worker(self, callback: Callable[[str], None], 
                                     language: str = None):
       
        with SuppressStream():
            try:
                import pyaudio
                import wave
                
                audio = pyaudio.PyAudio()
                
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=self.audio_manager.channels,
                    rate=self.audio_manager.sample_rate,
                    input=True,
                    input_device_index=self.audio_manager.device_index,
                    frames_per_buffer=1024
                )
                
                frames = []
                silence_chunks = 0
                # 1 second silence = sample_rate / chunk_size chunks
                max_silence_chunks = int(1.0 * self.audio_manager.sample_rate / 1024)
                speech_started = False
                user_interrupted = False
                
                while self.is_recording:
                    data = stream.read(1024, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Voice activity detection
                    if self.vad.is_speech(audio_data):
                        # Speech detected
                        if not speech_started:
                            speech_started = True
                            user_interrupted = True
                            # Trigger interruption callback
                            if self.on_speech_start:
                                self.on_speech_start()
                        
                        frames.append(data)
                        silence_chunks = 0
                    else:
                        # Silence detected
                        if speech_started:
                            silence_chunks += 1
                            frames.append(data)
                            
                            # 1 second of silence reached - process audio
                            if silence_chunks >= max_silence_chunks and len(frames) > 0:
                                # Save and transcribe
                                output_file = os.path.join(
                                    self.temp_dir, 
                                    f"recording_{int(time.time()*1000)}.wav"
                                )
                                
                                wf = wave.open(output_file, 'wb')
                                wf.setnchannels(self.audio_manager.channels)
                                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                                wf.setframerate(self.audio_manager.sample_rate)
                                wf.writeframes(b''.join(frames))
                                wf.close()
                                
                                # Transcribe
                                result = self.transcribe_file(output_file, language)
                                text = result.get("text", "").strip()
                                
                                if text and callback:
                                    callback(text)
                                
                                # Cleanup
                                try:
                                    os.remove(output_file)
                                except:
                                    pass
                                
                                # Reset for next utterance
                                frames = []
                                silence_chunks = 0
                                speech_started = False
                                user_interrupted = False
                
                stream.stop_stream()
                stream.close()
                audio.terminate()
                
            except Exception as e:
                print(f"Continuous recording error: {e}")
    
    def start_continuous_listening(self, 
                                   callback: Callable[[str], None],
                                   language: str = None,
                                   on_speech_start: Callable[[], None] = None):
       
        if self.is_recording:
            print("Already recording")
            return
        
        self.on_speech_start = on_speech_start
        self.is_recording = True
        self.recording_thread = threading.Thread(
            target=self._continuous_recording_worker,
            args=(callback, language),
            daemon=True
        )
        self.recording_thread.start()
        print("✓ Continuous listening started (1-second silence trigger)")
    
    def stop_continuous_listening(self):
       
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        self.on_speech_start = None
        print("✓ Continuous listening stopped")
    
    def is_listening(self) -> bool:
        
        return self.is_recording
    
    def get_model_info(self) -> dict:
        
        return {
            "model_size": self.model_size,
            "microphone": self.audio_manager.get_device_info()
        }
    
    def change_model(self, model_size: str):
       
        if model_size == self.model_size:
            return
        
        self.model_size = model_size
        self._load_model()
    
    def cleanup(self):
        
        self.stop_continuous_listening()
        
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass


# Convenience functions
def transcribe_file(audio_file: str, language: str = None) -> str:
   
    stt = SpeechToText()
    result = stt.transcribe_file(audio_file, language)
    return result.get("text", "")


def listen(duration: float = 5.0, language: str = None) -> str:
   
    stt = SpeechToText()
    return stt.listen_once(duration, language)


if __name__ == "__main__":
   
    print("="*60)
    print("Speech-to-Text - 1 Second Silence Trigger Test")
    print("="*60)
    print()
    
    # Initialize
    stt = SpeechToText(model_size="base")
    
    print("\n" + "="*60)
    print("System Information:")
    print("-" * 60)
    info = stt.get_model_info()
    print(f"Model: {info['model_size']}")
    print(f"Microphone: {info['microphone']['name']}")
    print(f"Silence trigger: 1.0 second")
    print("="*60)
    print()
    
    print("Starting continuous listening...")
    print("Speak naturally, then pause for 1 second.")
    print("Press Ctrl+C to stop.\n")
    
    def on_speech(text):
        print(f"\n> Transcription: {text}\n")
    
    def on_interrupt():
        print("[Speech detected - would interrupt TTS here]")
    
    stt.start_continuous_listening(on_speech, language="en", on_speech_start=on_interrupt)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        stt.stop_continuous_listening()
        stt.cleanup()
        print("Done!")