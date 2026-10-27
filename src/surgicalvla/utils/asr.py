import difflib
import queue
import re
import threading

import noisereduce as nr
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

INSTRUMENTS = ["forceps", "scissors", "scalpel", "clamp", "retractor", "hemostat", "tweezers", "needle"]
COMMAND_PATTERN = re.compile(
    r"\b(hand\s?over|hanover|handover|hand me|pass|get me|give me|bring me|i need|i want|fetch|grab)\b"
)


class ASR:
    def __init__(self, model_size="small", language="en", sample_rate=16000,
                 chunk_ms=32, silence_ms=608, min_speech_ms=288, on_command=None, input_device=None):
        self.sr = sample_rate
        self.chunk = int(sample_rate * chunk_ms / 1000)
        self.silence_chunks = silence_ms // chunk_ms
        self.min_chunks = min_speech_ms // chunk_ms
        self.language = language
        self.on_command = on_command or (lambda instrument, text: print(f"COMMAND -> {instrument}"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute)
        self.vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self.prompt = "Pass the scalpel. Hand over the forceps. Get me the scissors."
        self.input_device = input_device
        self.capture_sr = self._pick_capture_sr(sample_rate)
        self.audio_q = queue.Queue()
        self.running = False
        self.active = threading.Event()
        self.active.set()
        self.noise_profile = None

    def _pick_capture_sr(self, target_sr):
        """Return a sample rate the input device actually supports.

        Devices like USB mics often reject 16 kHz; in that case fall back to
        the device default (e.g. 48 kHz) and resample to target_sr in the callback.
        """
        if self.input_device is None:
            return target_sr
        try:
            sd.check_input_settings(device=self.input_device, samplerate=target_sr, channels=1)
            return target_sr
        except Exception:
            default_sr = int(sd.query_devices(self.input_device)["default_samplerate"])
            print(f"Input device does not support {target_sr} Hz; "
                  f"capturing at {default_sr} Hz and resampling to {target_sr} Hz.")
            return default_sr

    def _callback(self, indata, frames, time, status):
        audio = indata[:, 0].copy()
        if self.capture_sr != self.sr:
            audio = resample_poly(audio, self.sr, self.capture_sr).astype(np.float32)
        self.audio_q.put(audio)

    def _calibrate(self, seconds=1.5):
        print("Calibrating noise profile, stay quiet...")
        buf = sd.rec(int(seconds * self.capture_sr), samplerate=self.capture_sr,
                     channels=1, dtype="float32", device=self.input_device)
        sd.wait()
        noise = buf[:, 0]
        if self.capture_sr != self.sr:
            noise = resample_poly(noise, self.sr, self.capture_sr).astype(np.float32)
        self.noise_profile = noise

    def _denoise(self, audio):
        return nr.reduce_noise(y=audio, sr=self.sr, y_noise=self.noise_profile,
                               stationary=True, prop_decrease=0.9)

    def _is_speech(self, chunk):
        # Silero VAD requires >= 512 samples at 16 kHz (sr/len must be <= 31.25)
        if chunk.shape[0] < 512:
            chunk = np.pad(chunk, (0, 512 - chunk.shape[0]))
        return self.vad(torch.from_numpy(chunk), self.sr).item() > 0.5

    def _find_instrument(self, text):
        text = text.lower()
        if not COMMAND_PATTERN.search(text):
            return None
        for word in re.findall(r"[a-z]+", text):
            match = difflib.get_close_matches(word, INSTRUMENTS, n=1, cutoff=0.8)
            if match:
                return match[0]
        return None

    def _transcribe(self, frames):
        audio = self._denoise(np.concatenate(frames)).astype(np.float32)
        segments, _ = self.model.transcribe(audio, language=self.language, beam_size=5,
                                            vad_filter=True, initial_prompt=self.prompt)
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            return
        print(text)
        instrument = self._find_instrument(text)
        if instrument:
            self.active.clear()
            self.on_command(instrument, text)
            print("Transcription paused. Press Enter or say a new command via resume().")

    def resume(self):
        while not self.audio_q.empty():
            self.audio_q.get_nowait()
        self.active.set()

    def _worker(self):
        frames, silent, speaking = [], 0, False
        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self.active.is_set():
                frames, silent, speaking = [], 0, False
                continue
            if self._is_speech(chunk):
                speaking, silent = True, 0
                frames.append(chunk)
            elif speaking:
                silent += 1
                frames.append(chunk)
                if silent >= self.silence_chunks:
                    if len(frames) - silent >= self.min_chunks:
                        self._transcribe(frames)
                    frames, silent, speaking = [], 0, False

    def _resume_listener(self):
        while self.running:
            input()
            self.resume()

    def run(self):
        self._calibrate()
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._resume_listener, daemon=True).start()
        print("Listening... Ctrl+C to stop.")
        try:
            # blocksize must be in capture-rate samples so each callback chunk ≈ chunk_ms
            capture_blocksize = int(self.chunk * self.capture_sr / self.sr)
            with sd.InputStream(samplerate=self.capture_sr, channels=1, dtype="float32",
                                blocksize=capture_blocksize, callback=self._callback,
                                device=self.input_device):
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            self.running = False


if __name__ == "__main__":
    ASR(input_device="Amazon Basics BT-21: USB Audio (hw:1,0)").run()