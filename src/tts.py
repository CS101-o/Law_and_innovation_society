from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
import speech_recognition as sr
import os


audio_folder = "src/audio"
os.makedirs(audio_folder, exist_ok=True)

mp3_path = os.path.join(audio_folder, "output.mp3")
wav_path = os.path.join(audio_folder, "output.wav")

# --- TEXT TO SPEECH ---
def text_to_speech(text: str):
    tts = gTTS(text)
    tts.save(mp3_path)

    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

    wave_obj = sa.WaveObject.from_wave_file(wav_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()


# --- SPEECH TO TEXT ---
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
    # Speech-to-Text
    user_text = speech_to_text()
    if user_text:
        # Text-to-Speech (repeat what user said)
        print("Repeating what you said...")
        text_to_speech(user_text)