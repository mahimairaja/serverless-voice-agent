"""Kyutai STT on Modal, exposed as a streaming WebSocket."""

import asyncio
import json
import time
from pathlib import Path

import modal

APP_NAME = "voice-agent-stt"
MODEL_NAME = "kyutai/stt-1b-en_fr"
GPU = "l40s"
MINUTES = 60

# Mimi, the audio codec this model runs on, is fixed at 24 kHz mono.
# Resampling room audio to this rate is the LiveKit plugin's job.
SAMPLE_RATE = 24000

app = modal.App(APP_NAME)

stt_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "moshi==0.2.9",
        "fastapi[standard]==0.116.1",
        "huggingface-hub==0.33.5",
        "julius==0.2.7",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# Weights live on a Volume rather than in the image, so rebuilds stay fast and the
# first container after a deploy loads from Modal's disk instead of Hugging Face.
hf_cache_vol = modal.Volume.from_name(f"{APP_NAME}-hf-cache", create_if_missing=True)
HF_CACHE_PATH = Path("/root/.cache/huggingface")


@app.cls(
    image=stt_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    scaledown_window=5 * MINUTES,
    timeout=30 * MINUTES,  # a WebSocket is one long-lived input, so this caps session length
)
class STT:
    # Deliberately no @modal.concurrent. The model holds per-stream state on the
    # instance, so one container serves exactly one WebSocket and Modal adds
    # containers as callers arrive. Raising concurrency here corrupts transcripts.

    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from moshi.models import LMGen, loaders

        start = time.monotonic()
        snapshot_download(MODEL_NAME)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        info = loaders.CheckpointInfo.from_hf_repo(MODEL_NAME)

        self.mimi = info.get_mimi(device=self.device)
        assert int(self.mimi.sample_rate) == SAMPLE_RATE, (
            f"checkpoint expects {self.mimi.sample_rate} Hz, SAMPLE_RATE is {SAMPLE_RATE}"
        )
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)  # 1920 samples = 80 ms
        self.lm_gen = LMGen(info.get_moshi(device=self.device), temp=0, temp_text=0)
        self.text_tokenizer = info.get_text_tokenizer()
        self.padding_token_id = info.raw_config.get("text_padding_token_id", 3)
        # The model expects roughly a second of silence before real speech begins.
        # Every new connection gets primed with it.
        self.silence_prefix_seconds = info.stt_config.get("audio_silence_prefix_seconds", 1.0)

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        for _ in range(4):  # warm the kernels so the first real caller is not the slow one
            codes = self.mimi.encode(torch.zeros(1, 1, self.frame_size, device=self.device))
            for c in range(codes.shape[-1]):
                self.lm_gen.step(codes[:, :, c : c + 1])
        if self.device == "cuda":
            torch.cuda.synchronize()

        self.reset()
        print(f"loaded {MODEL_NAME} in {time.monotonic() - start:.1f}s on {self.device}")

    def reset(self):
        """Clear streaming state. Call between connections, never mid-stream."""
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    def _step(self, frame):
        """Run one 80 ms frame through the model, yielding transcript events."""
        import torch

        with torch.no_grad():
            chunk = torch.from_numpy(frame).view(1, 1, -1).to(self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                text_tokens, extra_heads = self.lm_gen.step_with_extra_heads(codes[:, :, c : c + 1])
                if text_tokens is None:
                    continue  # model is mid-computation, nothing to emit yet
                if extra_heads and extra_heads[2][0, 0, 0].item() > 0.5:
                    yield {"type": "eot"}
                token = int(text_tokens[0, 0, 0].item())
                if token not in (0, self.padding_token_id):
                    piece = self.text_tokenizer.id_to_piece(token).replace("▁", " ")
                    yield {"type": "delta", "text": piece}

    @modal.asgi_app()
    def web(self):
        import numpy as np
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        api = FastAPI()

        @api.get("/health")
        def health():
            return {"status": "ok", "model": MODEL_NAME, "sample_rate": SAMPLE_RATE}

        @api.websocket("/ws")
        async def transcribe(ws: WebSocket):
            await ws.accept()
            self.reset()

            inbound: asyncio.Queue = asyncio.Queue()
            buffer = np.zeros(int(self.silence_prefix_seconds * SAMPLE_RATE), dtype=np.float32)

            async def receive():
                try:
                    while True:
                        await inbound.put(await ws.receive_bytes())
                finally:
                    await inbound.put(None)

            async def infer():
                nonlocal buffer
                while True:
                    data = await inbound.get()
                    if data is None:
                        return
                    if len(data) % 2:
                        continue  # not a whole number of int16 samples, drop it
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    buffer = np.concatenate((buffer, samples))
                    # Inference is synchronous and holds the event loop. That is fine
                    # at one stream per container: a frame is 80 ms of audio and takes
                    # single-digit milliseconds on an L40S. Receiving is decoupled via
                    # the queue so network jitter does not stall the GPU.
                    while buffer.shape[-1] >= self.frame_size:
                        frame, buffer = buffer[: self.frame_size], buffer[self.frame_size :]
                        for event in self._step(frame):
                            await ws.send_text(json.dumps(event))

            tasks = [asyncio.create_task(receive()), asyncio.create_task(infer())]
            try:
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print("websocket error:", e)
                raise
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        return api

    @modal.method()
    def transcribe_bytes(self, audio: bytes) -> str:
        """Batch path used by the smoke test below. The WebSocket is the real interface."""
        import tempfile

        import sphn

        self.reset()
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            tmp.write(audio)
            tmp.flush()
            pcm, _ = sphn.read(tmp.name, sample_rate=SAMPLE_RATE)

        buffer = pcm[0].astype("float32")
        pieces = []
        while buffer.shape[-1] >= self.frame_size:
            frame, buffer = buffer[: self.frame_size], buffer[self.frame_size :]
            for event in self._step(frame):
                if event["type"] == "delta":
                    pieces.append(event["text"])
        return "".join(pieces).strip()


@app.local_entrypoint()
def test(
    audio_url: str = "https://github.com/kyutai-labs/delayed-streams-modeling/raw/refs/heads/main/audio/bria.mp3",
):
    from urllib.request import urlopen

    audio = urlopen(audio_url).read()
    print(f"transcribing {len(audio)} bytes from {audio_url}")

    start = time.monotonic()
    print(STT().transcribe_bytes.remote(audio))
    print(f"done in {time.monotonic() - start:.1f}s")
