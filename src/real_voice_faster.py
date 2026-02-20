import os
import sys
import time
import queue
import getpass
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3

from faster_whisper import WhisperModel
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

MAX_RECORD_SECONDS = 12
SILENCE_HOLD_SECONDS = 1.0

# Tuning: increase threshold if it stops too late (room noise),
# decrease if it stops too early.
SILENCE_RMS_THRESHOLD = 0.012

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium/large-v3
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Reply concisely (2-6 sentences) unless asked otherwise."
)

TEMP_WAV_PATH = "utterance.wav"


# -----------------------------
# TTS helpers
# -----------------------------
def pick_female_voice(engine: pyttsx3.Engine) -> None:
    voices = engine.getProperty("voices") or []
    preferred_keywords = ["zira", "female", "susan", "kate", "eva", "joanna"]
    for kw in preferred_keywords:
        for v in voices:
            name = (getattr(v, "name", "") or "").lower()
            vid = (getattr(v, "id", "") or "").lower()
            if kw in name or kw in vid:
                engine.setProperty("voice", v.id)
                return


def speak(engine: pyttsx3.Engine, text: str) -> None:
    engine.say(text)
    engine.runAndWait()


# -----------------------------
# Audio capture
# -----------------------------
def record_until_silence() -> np.ndarray:
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        q.put(indata.copy())

    print("🎙️ Recording... (speak now)")
    chunks = []
    silence_start = None
    start_time = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
        blocksize=int(SAMPLE_RATE * 0.05),  # 50ms blocks
    ):
        while True:
            if time.time() - start_time > MAX_RECORD_SECONDS:
                break

            try:
                chunk = q.get(timeout=0.2)
            except queue.Empty:
                continue

            chunks.append(chunk)

            rms = float(np.sqrt(np.mean(np.square(chunk))))
            now = time.time()

            # stop after continuous silence
            if rms < SILENCE_RMS_THRESHOLD:
                if silence_start is None:
                    silence_start = now
                elif (now - silence_start) >= SILENCE_HOLD_SECONDS and (now - start_time) > 0.8:
                    break
            else:
                silence_start = None

    audio = np.concatenate(chunks, axis=0).reshape(-1)
    return audio


def save_wav(path: str, audio_f32: np.ndarray) -> None:
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    write(path, SAMPLE_RATE, audio_i16)


# -----------------------------
# faster-whisper transcription
# -----------------------------
def build_whisper_model() -> WhisperModel:
    """
    Try GPU if available, otherwise CPU.
    compute_type int8 gives best CPU performance.
    """
    prefer_gpu = os.getenv("WHISPER_DEVICE", "").lower()  # set to "cuda" to force
    if prefer_gpu == "cuda":
        return WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    # default: CPU int8
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe_whisper(model: WhisperModel, wav_path: str) -> str:
    segments, info = model.transcribe(
        wav_path,
        vad_filter=True,          # helps in noisy rooms
        beam_size=5,
    )
    text = "".join(seg.text for seg in segments).strip()
    return text


# -----------------------------
# OpenAI call
# -----------------------------
def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    return getpass.getpass("Enter OPENAI_API_KEY (input hidden): ").strip()


def ask_openai(client: OpenAI, user_text: str) -> str:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )
    return (getattr(resp, "output_text", "") or "").strip()


# -----------------------------
# Main loop
# -----------------------------
def main():
    api_key = get_api_key()
    if not api_key:
        print("No API key provided.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Whisper model: {WHISPER_MODEL}")
    print(f"OpenAI model:  {OPENAI_MODEL}")
    print("Ready. Press Enter to talk, Ctrl+C to quit.\n")

    whisper = build_whisper_model()

    tts = pyttsx3.init()
    tts.setProperty("rate", 165)
    pick_female_voice(tts)

    while True:
        try:
            input("▶ Press Enter to talk...")
            audio = record_until_silence()
            save_wav(TEMP_WAV_PATH, audio)

            text = transcribe_whisper(whisper, TEMP_WAV_PATH)
            if not text:
                print("…I didn't catch that. Try again.\n")
                continue

            print(f"📝 You said: {text}")

            answer = ask_openai(client, text)
            if not answer:
                print("⚠️ Empty response from API.\n")
                continue

            print(f"🤖 Assistant: {answer}\n")
            speak(tts, answer)

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
