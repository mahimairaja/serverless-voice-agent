"""Chatterbox Turbo TTS on Modal, exposed as an HTTP synthesis endpoint."""

import modal

APP_NAME = "voice-agent-tts"
MINUTES = 60

# Chatterbox synthesizes at 24 kHz mono. LiveKit resamples on its side.
SAMPLE_RATE = 24000

# Voice cloning is optional. Drop a ~10s wav into the volume below and name it here
# to clone it, or leave the volume empty to use the checkpoint's built-in voice.
#   modal volume create chatterbox-tts-voices
#   modal volume put chatterbox-tts-voices <unzipped-voice-prompts-dir>
# Prompt pack: https://modal-cdn.com/blog/audio/chatterbox-tts-voices.zip
VOICE_PROMPTS_DIR = "/chatterbox-tts/prompts"
VOICE_FILE = "Lucy.wav"

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "chatterbox-tts==0.1.6",
    "fastapi[standard]==0.124.4",
    "peft==0.18.0",
)

app = modal.App(APP_NAME, image=image)

# create_if_missing so a fresh clone deploys without any volume setup at all.
voices_vol = modal.Volume.from_name("chatterbox-tts-voices", create_if_missing=True)

# Weights are gated: modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret")

with image.imports():
    import io
    import wave
    from pathlib import Path

    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from fastapi import Response


@app.cls(
    gpu="a10g",
    secrets=[hf_secret],
    volumes={VOICE_PROMPTS_DIR: voices_vol},
    scaledown_window=5 * MINUTES,  # stay warm between turns of a conversation
)
# Four generations per A10G. Higher packs more callers per GPU but stretches
# time-to-first-audio, which is the number that decides whether a call feels alive.
@modal.concurrent(max_inputs=4)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")

        # Prepare the voice once here rather than per request. prepare_conditionals
        # is not cheap, and re-running it on every sentence would show up as latency
        # on every single turn.
        voice = self._find_voice()
        if voice:
            print(f"cloning voice from {voice}")
            self.model.prepare_conditionals(voice)
        elif self.model.conds is None:
            raise RuntimeError(
                f"no {VOICE_FILE} in the chatterbox-tts-voices volume and this "
                "checkpoint ships no default voice. Upload a voice prompt, see the "
                "comment at the top of this file."
            )
        else:
            print("using the checkpoint's built-in voice")

        self.model.generate("Warming up.")  # pay the first-call cost before a caller does

    def _find_voice(self) -> str | None:
        root = Path(VOICE_PROMPTS_DIR)
        if not root.exists():
            return None
        # rglob because `modal volume put` of a directory preserves its nesting
        matches = sorted(root.rglob(VOICE_FILE))
        return str(matches[0]) if matches else None

    @modal.method()
    def generate(self, text: str) -> bytes:
        """Synthesize text to raw int16 PCM at SAMPLE_RATE."""
        # audio_prompt_path is omitted on purpose: conditionals were prepared at load.
        wav = self.model.generate(text)
        pcm = (wav.squeeze(0).clamp(-1.0, 1.0) * 32767).to(torch.int16)
        return pcm.cpu().numpy().tobytes()

    @modal.fastapi_endpoint(docs=True, method="POST")
    def speak(self, text: str, format: str = "wav"):
        pcm = self.generate.local(text)

        if format == "pcm":
            return Response(
                content=pcm,
                media_type=f"audio/L16; rate={SAMPLE_RATE}; channels=1",
            )

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(pcm)
        return Response(content=buffer.getvalue(), media_type="audio/wav")


@app.local_entrypoint()
def test(
    text: str = "Chatterbox running on Modal [chuckle].",
    output_path: str = "/tmp/chatterbox-tts/output.wav",
):
    import pathlib
    import wave

    pcm = Chatterbox().generate.remote(text)

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)

    print(f"wrote {len(pcm) / 2 / SAMPLE_RATE:.1f}s of audio to {path}")
