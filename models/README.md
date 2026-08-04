# Modal-hosted models

Open-source replacements for the three cloud plugins in `../agent.py`. Each file is a
standalone Modal app: deploy it, take the URL, point the LiveKit plugin at it.

| file | model | replaces | interface |
| --- | --- | --- | --- |
| `stt.py` | Kyutai STT 1B | `deepgram.STT` | WebSocket `/ws`, int16 PCM in, JSON out |
| `llm.py` | Nemotron 3 Nano 30B-A3B | `openai.LLM` | OpenAI `/v1/chat/completions` |
| `tts.py` | Chatterbox Turbo | `deepgram.TTS` | `POST /?text=...&format=pcm` |

All three run at 24 kHz mono where audio is involved.

## One-time setup

```
uv add modal
uv run modal setup
uv run modal secret create huggingface-secret HF_TOKEN=hf_...
```

## Up

Run from the repo root:

```
uv run modal deploy models/stt.py
uv run modal deploy models/llm.py
uv run modal deploy models/tts.py
```

Each prints a URL. Put them in `.env.local`:

```
MODAL_STT_URL=wss://<workspace>--voice-agent-stt-stt-web.modal.run/ws
MODAL_LLM_URL=https://<workspace>--voice-agent-llm-server.us-west.modal.direct/v1
MODAL_TTS_URL=https://<workspace>--voice-agent-tts-chatterbox-speak.modal.run
```

Smoke test each one before wiring the agent. The first run of each is slow: it
downloads weights to a Volume. Later runs load from that Volume.

```
uv run modal run models/stt.py
uv run modal run models/llm.py
uv run modal run models/tts.py
```

## Down

```
uv run modal app stop voice-agent-stt
uv run modal app stop voice-agent-llm
uv run modal app stop voice-agent-tts
```

`modal app list` shows what is still running.

Stopping is what ends billing. A deployed app holds containers for its
`scaledown_window` after the last request, and `llm.py` runs a B200 continuously if
you set `MIN_CONTAINERS = 1` (which you want for a live demo, and do not want
afterwards). Volumes persist across stops, so redeploying is fast and cheap.
