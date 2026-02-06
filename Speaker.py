import asyncio
import edge_tts
import os
import subprocess
import tempfile
import time
from pathlib import Path
import warnings
import sys
import re
from typing import Dict, Tuple, Optional
from queue import Queue
from threading import Thread, Lock

warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'

class SuppressStream:
    def __init__(self, stream=sys.stderr):
        self.stream = stream
        self.fd = stream.fileno()
        
    def __enter__(self):
        self.old_fd = os.dup(self.fd)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, self.fd)
        
    def __exit__(self, *args):
        os.dup2(self.old_fd, self.fd)
        os.close(self.old_fd)
        os.close(self.devnull)


class EmotionDetector:

    def __init__(self):
        self.emotion_patterns = {
            'excited': {
                'keywords': ['amazing', 'awesome', 'fantastic', 'wonderful', 'excellent', 
                           'great', 'yay', 'wow', 'brilliant', 'perfect', 'incredible'],
                'punctuation': ['!', '!!'],
                'pitch': '+20Hz',
                'rate': '+15%',
                'volume': '+10%'
            },
            'happy': {
                'keywords': ['good', 'nice', 'pleasant', 'lovely', 'delightful', 
                           'glad', 'happy', 'cheerful', 'enjoy', 'fun'],
                'punctuation': ['!'],
                'pitch': '+10Hz',
                'rate': '+5%',
                'volume': '+5%'
            },
            'sad': {
                'keywords': ['sad', 'sorry', 'unfortunately', 'disappointed', 'upset',
                           'unhappy', 'regret', 'fail', 'failed', 'problem'],
                'punctuation': [],
                'pitch': '-15Hz',
                'rate': '-10%',
                'volume': '-5%'
            },
            'calm': {
                'keywords': ['okay', 'alright', 'fine', 'understood', 'noted', 
                           'acknowledged', 'sure', 'yes'],
                'punctuation': ['.'],
                'pitch': '+0Hz',
                'rate': '+0%',
                'volume': '+0%'
            },
            'urgent': {
                'keywords': ['urgent', 'emergency', 'immediately', 'now', 'quick',
                           'hurry', 'warning', 'alert', 'critical', 'important'],
                'punctuation': ['!', '!!'],
                'pitch': '+25Hz',
                'rate': '+20%',
                'volume': '+15%'
            },
            'questioning': {
                'keywords': ['what', 'why', 'how', 'when', 'where', 'who', 'which'],
                'punctuation': ['?'],
                'pitch': '+15Hz',
                'rate': '+5%',
                'volume': '+0%'
            },
            'apologetic': {
                'keywords': ['sorry', 'apologize', 'pardon', 'excuse', 'forgive',
                           'my bad', 'oops', 'mistake'],
                'punctuation': [],
                'pitch': '-5Hz',
                'rate': '-5%',
                'volume': '-3%'
            },
            'confident': {
                'keywords': ['definitely', 'certainly', 'absolutely', 'sure', 'confirm',
                           'guaranteed', 'promise', 'will do', 'done', 'completed'],
                'punctuation': [],
                'pitch': '+5Hz',
                'rate': '+0%',
                'volume': '+8%'
            }
        }
    
    def detect_emotion(self, text: str) -> Tuple[str, Dict[str, str]]:
    
        text_lower = text.lower()
        scores = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
        
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    score += 2
            
            for punct in patterns['punctuation']:
                if punct in text:
                    score += 1
            
            scores[emotion] = score
        
        if max(scores.values()) > 0:
            emotion = max(scores, key=scores.get)
        else:
            emotion = 'calm'
        
        return emotion, self.emotion_patterns[emotion]


class SentenceSplitter:
   
    @staticmethod
    def split_sentences(text: str) -> list:
        
        text = re.sub(r'([.!?])\s+', r'\1|||', text)
        sentences = text.split('|||')
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


class EmotionalTTS:

    _instance = None
    _initialized = False
    
    def __new__(cls):

        if cls._instance is None:
            cls._instance = super(EmotionalTTS, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
       
        if self._initialized:
            return
        
        self.temp_dir = tempfile.mkdtemp(prefix="etts_")
        self.audio_device = None
        self.player_command = None
      
        self.voice = 'en-US-JennyNeural'
        
        self.emotion_detector = EmotionDetector()
        self.sentence_splitter = SentenceSplitter()
        
        self.speaking = False
        self.current_emotion = 'calm'
        
        self.streaming_queue = Queue()
        self.streaming_active = False
        self.streaming_lock = Lock()
        self.stream_worker = None
        
        self._setup_audio()
        self._initialized = True
    
    def _setup_audio(self):
       
        try:
            result = subprocess.run(
                ['pactl', 'list', 'short', 'sinks'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            sink_name = parts[1].lower()
                            
                            if 'alsa' in sink_name or 'pci' in sink_name:
                                self.audio_device = parts[1]
                                break
                            elif 'blue' in sink_name or 'bt' in sink_name:
                                if not self.audio_device:
                                    self.audio_device = parts[1]
        except:
            pass
        
        players = [
            ('ffplay', ['-nodisp', '-autoexit', '-loglevel', 'quiet']),
            ('paplay', []),
            ('mpv', ['--really-quiet', '--no-video']),
        ]
        
        for player, args in players:
            try:
                result = subprocess.run(['which', player], capture_output=True, timeout=1)
                if result.returncode == 0:
                    self.player_command = [player] + args
                    break
            except:
                continue
    
    def _parse_prosody_value(self, value: str, param_type: str) -> str:
        
        if not value or value in ['0Hz', '0%', '+0%', '+0Hz']:
            return None
        
        value = value.replace('+', '')
        
        return value
    
    async def _generate_speech(self, text: str, output_file: str, use_emotion: bool = True) -> bool:
       
        try:
            if use_emotion:
          
                emotion, emotion_params = self.emotion_detector.detect_emotion(text)
                self.current_emotion = emotion
                
                pitch = self._parse_prosody_value(emotion_params.get('pitch', '0Hz'), 'pitch')
                rate = self._parse_prosody_value(emotion_params.get('rate', '0%'), 'rate')
                volume = self._parse_prosody_value(emotion_params.get('volume', '0%'), 'volume')
                
                communicate = edge_tts.Communicate(
                    text, 
                    self.voice,
                    rate=rate,
                    pitch=pitch,
                    volume=volume
                )
            else:

                communicate = edge_tts.Communicate(text, self.voice)
            
            await communicate.save(output_file)
            return True
            
        except Exception as e:

            try:
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(output_file)
                return True
            except:
                return False
    
    def _play_audio(self, audio_file: str) -> bool:
        if not os.path.exists(audio_file):
            return False
        
        try:
            if self.player_command[0] == 'paplay' and self.audio_device:
                cmd = ['paplay', '-d', self.audio_device, audio_file]
            else:
                cmd = self.player_command + [audio_file]
            
            with SuppressStream():
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30
                )
            
            return result.returncode == 0
        except:
            return False
    
    async def _speak_async(self, text: str, emotion: str = None):
       
        if not text.strip():
            return False
        
        audio_file = os.path.join(self.temp_dir, f"speech_{int(time.time()*1000)}.mp3")
        
        try:
            self.speaking = True
            if emotion:
          
                emotion_params = self.emotion_detector.emotion_patterns.get(
                    emotion, 
                    self.emotion_detector.emotion_patterns['calm']
                )
                self.current_emotion = emotion
                
                pitch = self._parse_prosody_value(emotion_params.get('pitch', '0Hz'), 'pitch')
                rate = self._parse_prosody_value(emotion_params.get('rate', '0%'), 'rate')
                volume = self._parse_prosody_value(emotion_params.get('volume', '0%'), 'volume')
                
                communicate = edge_tts.Communicate(
                    text, 
                    self.voice,
                    rate=rate,
                    pitch=pitch,
                    volume=volume
                )
                await communicate.save(audio_file)
            else:
                success = await self._generate_speech(text, audio_file, use_emotion=True)
                if not success:
                    return False
            
            with SuppressStream():
                success = self._play_audio(audio_file)
            
            return success
            
        finally:
            self.speaking = False

            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except:
                pass
    
    def _stream_worker(self):
       
        while self.streaming_active:
            try:

                chunk = self.streaming_queue.get(timeout=0.1)
                
                if chunk is None:  
                    break
                
                self.speak(chunk)
                
                self.streaming_queue.task_done()
                
            except:
                continue
    
    def speak(self, text: str, emotion: str = None) -> bool:
        
        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            
            if loop and loop.is_running():
                future = asyncio.ensure_future(self._speak_async(text, emotion))
                while not future.done():
                    time.sleep(0.01)
                return future.result()
            else:
                return asyncio.run(self._speak_async(text, emotion))
        except Exception as e:
            return False
    
    async def speak_async(self, text: str, emotion: str = None):
       
        return await self._speak_async(text, emotion)
    
    def start_stream(self):
       
        with self.streaming_lock:
            if not self.streaming_active:
                self.streaming_active = True
                self.stream_worker = Thread(target=self._stream_worker, daemon=True)
                self.stream_worker.start()
    
    def stream_text(self, text_chunk: str):
        
        if not self.streaming_active:
            self.start_stream()
        
        sentences = self.sentence_splitter.split_sentences(text_chunk)
        
        for sentence in sentences:
            if sentence.strip():
                self.streaming_queue.put(sentence)
    
    def end_stream(self):
       
        with self.streaming_lock:
            if self.streaming_active:
    
                self.streaming_queue.join()
                
                self.streaming_queue.put(None)  
                if self.stream_worker:
                    self.stream_worker.join(timeout=2)
                
                self.streaming_active = False
    
    def stop_stream(self):
        
        with self.streaming_lock:
            if self.streaming_active:
                while not self.streaming_queue.empty():
                    try:
                        self.streaming_queue.get_nowait()
                        self.streaming_queue.task_done()
                    except:
                        break
                
                self.streaming_queue.put(None)
                if self.stream_worker:
                    self.stream_worker.join(timeout=2)
                
                self.streaming_active = False
    
    def get_current_emotion(self) -> str:
        return self.current_emotion
    
    def is_speaking(self) -> bool:
        return self.speaking
    
    def is_streaming(self) -> bool:
        return self.streaming_active
    
    def set_voice(self, voice: str):
        
        self.voice = voice
    
    def get_available_emotions(self) -> list:
        return list(self.emotion_detector.emotion_patterns.keys())
    
    def cleanup(self):
        self.stop_stream()
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

def speak(text: str, emotion: str = None) -> bool:

    tts = EmotionalTTS()
    return tts.speak(text, emotion)


async def speak_async(text: str, emotion: str = None):
   
    tts = EmotionalTTS()
    return await tts.speak_async(text, emotion)


if __name__ == "__main__":
    print("="*60)
    print("Emotional TTS with Streaming Support")
    print("="*60)
    print()
    
    tts = EmotionalTTS()
    
    print("TEST 1: Instant Mode (Confirmations)")
    print("-" * 40)
    tts.speak("Task completed successfully!", emotion='confident')
    time.sleep(1)
    
    print("\nTEST 2: Streaming Mode (LLM Simulation)")
    print("-" * 40)
    
    llm_response = [
        "Hello! I'm your AI assistant.",
        "I can help you with various tasks.",
        "Just let me know what you need!",
        "I'm here to make your life easier."
    ]
    
    tts.start_stream()
    for chunk in llm_response:
        print(f"Streaming: {chunk}")
        tts.stream_text(chunk)
        time.sleep(0.5)  
    tts.end_stream()
    
    print("\n" + "="*60)
    print("Available emotions:", ", ".join(tts.get_available_emotions()))
    print("="*60)
    
    tts.cleanup()