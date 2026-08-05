"""Nemotron 3 Nano on SGLang, served from Modal as an OpenAI-compatible endpoint."""

import json
import subprocess
import time

import modal

APP_NAME = "voice-agent-llm"
MINUTES = 60
PORT = 8000

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
GPU_TYPE, N_GPUS = "B200", 1
GPU = f"{GPU_TYPE}:{N_GPUS}"

# Latency is dominated by geography. Put the GPUs and the routing proxies in the
# same region as the LiveKit workers and the callers.
REGION = "us"
ROUTING_REGION = "us-west"

# 0 keeps the bill at zero when idle but makes the first request pay a multi-minute
# cold start. Set to 1 before a live demo.
MIN_CONTAINERS = 0

# How many requests one replica should carry before Modal scales up. Tune with a
# real benchmark, not intuition.
TARGET_INPUTS = 32

app = modal.App(APP_NAME)

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.11")
    .entrypoint([])  # silence the image's chatty startup logs
    .run_commands("rm -rf /root/.cache/huggingface")
    .env(
        {
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "NVIDIA_TF32_OVERRIDE": "1",
        }
    )
)

# Gated and large-file downloads need a token. Create it once with:
#   modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret")

HF_CACHE_PATH = "/root/.cache/huggingface"
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

with sglang_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES):
    """Block until SGLang answers /health, so Modal does not route to a cold replica."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (rc := process.poll()) is not None:
            raise subprocess.CalledProcessError(rc, cmd=process.args)
        try:
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
            time.sleep(5)
    raise TimeoutError(f"SGLang not ready within {timeout}s")


def warmup():
    payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}
    for _ in range(3):
        requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=60).raise_for_status()


@app.server(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    secrets=[hf_secret],
    compute_region=REGION,
    routing_region=ROUTING_REGION,
    min_containers=MIN_CONTAINERS,
    target_concurrency=TARGET_INPUTS,
    startup_timeout=20 * MINUTES,  # weights are large on a cold Volume
    exit_grace_period=15,  # seconds to finish in-flight turns before shutdown
    port=PORT,
    unauthenticated=True,  # the LiveKit plugin then only needs the base_url
)
class Server:
    @modal.enter()
    def startup(self):
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            f"{PORT}",
            "--tp",
            f"{N_GPUS}",
            # only capture CUDA graphs for batch sizes we will actually see
            "--cuda-graph-max-bs",
            f"{TARGET_INPUTS * 2}",
            # quantize the KV cache: small accuracy cost, large memory win
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--enable-metrics",
            "--decode-log-interval",
            "10",
            "--trust-remote-code",
            # keeps tool calls and reasoning out of `delta.content`, so the agent
            # never speaks its own scratchpad aloud
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "nemotron_3",
        ]
        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def test(prompt: str = "In one sentence, what is a serverless GPU?", timeout: int = 10 * MINUTES):
    import urllib.error
    import urllib.request

    url = Server.get_url()
    print(f"server: {url}")

    body = json.dumps(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
        }
    ).encode()

    deadline = time.time() + timeout
    while time.time() < deadline:
        request = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                # Requests sharing this value are routed to the same replica, which
                # keeps the conversation's KV cache warm. Use one value per call.
                "Modal-Session-ID": "smoke-test",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            print(payload["choices"][0]["message"]["content"])
            return
        except urllib.error.HTTPError as e:
            if e.code != 503:  # 503 means no replica is live yet
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)

    raise TimeoutError(f"no response from {url} within {timeout}s")
