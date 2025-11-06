from io import BytesIO
import os
from typing import Optional

from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
import speech_recognition as sr


audio_folder = "src/audio"
os.makedirs(audio_folder, exist_ok=True)

mp3_path = os.path.join(audio_folder, "output.mp3")
wav_path = os.path.join(audio_folder, "output.wav")


def text_to_speech_bytes(text: str) -> bytes:
    """Return an MP3 byte stream generated with gTTS."""
    if not text.strip():
        raise ValueError("Cannot synthesize empty text.")
    tts = gTTS(text)
    mp3_buffer = BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    return mp3_buffer.read()


def _ensure_wav_bytes(audio_bytes: bytes, fmt: str) -> bytes:
    """Convert arbitrary audio bytes into mono WAV for recognition."""
    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format=fmt)
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

    wav_buffer = BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer.read()


def speech_to_text_from_audio_bytes(audio_bytes: bytes, fmt: str = "wav") -> Optional[str]:
    """Transcribe uploaded audio bytes using Google's recognizer."""
    fmt = (fmt or "wav").lower()
    safe_formats = {"wav", "x-wav", "aiff", "aifc", "flac"}
    wav_bytes = audio_bytes

    if fmt not in safe_formats:
        try:
            wav_bytes = _ensure_wav_bytes(audio_bytes, fmt)
            fmt = "wav"
        except Exception:
            return None

    recognizer = sr.Recognizer()
    audio_file = BytesIO(wav_bytes)
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None


# --- TEXT TO SPEECH (LOCAL PLAYBACK) ---
def text_to_speech(text: str):
    mp3_bytes = text_to_speech_bytes(text)
    with open(mp3_path, "wb") as mp3_file:
        mp3_file.write(mp3_bytes)

    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

    wave_obj = sa.WaveObject.from_wave_file(wav_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()


# --- SPEECH TO TEXT (LOCAL MICROPHONE) ---
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("STT request failed:", e)
        return None


if __name__ == "__main__":
    user_text = speech_to_text()
    if user_text:
        print("Repeating what you said...")
        text_to_speech(user_text)
