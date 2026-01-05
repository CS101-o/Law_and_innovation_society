from io import BytesIO
import os
from typing import Optional
import tempfile

from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
import speech_recognition as sr


audio_folder = "src/audio"
os.makedirs(audio_folder, exist_ok=True)

mp3_path = os.path.join(audio_folder, "output.mp3")
wav_path = os.path.join(audio_folder, "output.wav")

# --- TEXT TO SPEECH ---
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
    # For WebM and other streaming formats, try multiple approaches
    if fmt.lower() in {"webm", "opus", "ogg"}:
        # First try: temp file (more stable for some codecs)
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt.lower()}", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                audio_segment = AudioSegment.from_file(tmp_path, format=fmt)
                os.unlink(tmp_path)
            except Exception as e1:
                os.unlink(tmp_path)
                # Second try: BytesIO with format auto-detection
                try:
                    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format=fmt)
                except Exception as e2:
                    # Third try: Try raw audio interpretation if both fail
                    print(f"⚠ WebM decoding failed: {e1}")
                    raise e2
        except Exception as e:
            raise e
    else:
        audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format=fmt)

    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)

    wav_buffer = BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer.read()


def _detect_audio_format(audio_bytes: bytes) -> str:
    """Detect actual audio format from magic bytes."""
    if not audio_bytes or len(audio_bytes) < 4:
        return "unknown"

    magic = audio_bytes[:4]

    # RIFF (WAV, AVI, etc)
    if magic == b'RIFF':
        if len(audio_bytes) >= 12 and audio_bytes[8:12] == b'WAVE':
            return 'wav'
        return 'riff'

    # EBML (WebM, Matroska)
    if magic[:1] == b'\x1a' and len(magic) > 1 and magic[1:2] == b'E':
        return 'webm'

    # MP3 (ID3 or FFSync)
    if magic[:2] == b'ID' or (magic[0] == 0xFF and (magic[1] & 0xE0) == 0xE0):
        return 'mp3'

    # MP4/MOV
    if len(audio_bytes) >= 8 and audio_bytes[4:8] == b'ftyp':
        return 'mp4'

    # OGG Vorbis
    if magic == b'OggS':
        return 'ogg'

    # FLAC
    if magic == b'fLaC':
        return 'flac'

    return "unknown"


def speech_to_text_from_audio_bytes(audio_bytes: bytes, fmt: str = "wav") -> Optional[str]:
    """Transcribe uploaded audio bytes using Google's recognizer."""
    if not audio_bytes or len(audio_bytes) == 0:
        return None

    # Auto-detect format from magic bytes if not explicitly WAV
    detected_fmt = _detect_audio_format(audio_bytes)
    if detected_fmt != "unknown":
        fmt = detected_fmt
    else:
        fmt = (fmt or "wav").lower()

    print(f"📋 Detected audio format: {fmt}")
    recognizer = sr.Recognizer()

    # Try direct recognition first if it's already WAV
    if fmt in {"wav", "x-wav"}:
        try:
            audio_file = BytesIO(audio_bytes)
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print(f"✓ Transcribed: {text}")
            return text
        except Exception as e:
            print(f"Direct WAV transcription failed: {e}")
            return None

    # For other formats, try conversion
    try:
        converted_bytes = _ensure_wav_bytes(audio_bytes, fmt)
    except Exception as e:
        print(f"Format conversion error ({fmt}): {e}")

        # If format is unknown, it might be raw PCM - try to convert as raw audio
        if fmt == "unknown":
            try:
                print("Treating as raw PCM audio, attempting conversion...")
                # Assume raw PCM: mono, 16-bit, 16kHz (common browser recorder default)
                audio_segment = AudioSegment(
                    audio_bytes,
                    sample_width=2,  # 16-bit = 2 bytes
                    frame_rate=16000,
                    channels=1
                )
                wav_buffer = BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                converted_bytes = wav_buffer.read()
                print("✓ Raw PCM conversion successful")
            except Exception as e3:
                print(f"Raw PCM conversion failed: {e3}")
                # Try direct transcription as last resort
                try:
                    print("Attempting direct transcription without conversion...")
                    audio_file = BytesIO(audio_bytes)
                    with sr.AudioFile(audio_file) as source:
                        audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio)
                    print(f"✓ Transcribed (direct): {text}")
                    return text
                except Exception as e4:
                    print(f"Direct transcription also failed: {e4}")
                    return None
        else:
            # Not raw PCM, try direct transcription
            try:
                print("Attempting direct transcription without conversion...")
                audio_file = BytesIO(audio_bytes)
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                print(f"✓ Transcribed (direct): {text}")
                return text
            except Exception as e2:
                print(f"Direct transcription also failed: {e2}")
                return None

    # Transcribe the converted audio
    try:
        audio_file = BytesIO(converted_bytes)
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        print(f"✓ Transcribed: {text}")
        return text

    except sr.UnknownValueError:
        print("⚠ Audio could not be understood by Google Speech Recognition")
        return None
    except sr.RequestError as e:
        print(f"⚠ Google Speech Recognition API error: {e}")
        return None
    except Exception as e:
        print(f"⚠ Unexpected error during transcription: {e}")
        import traceback
        traceback.print_exc()
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
