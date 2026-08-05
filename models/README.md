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

## These endpoints are open

All three deploy with no authentication. That is deliberate: `modal deploy`, copy
the URL, point the agent at it, teach. It is not a configuration to leave running,
and it is not what you would ship.

A Modal URL is `https://<workspace>--<app>-<class>-<method>.modal.run`. That is a
naming convention, not a secret. Anyone who has it can:

- Run inference on your GPUs, billed to you.
- Hold one dedicated L40S per WebSocket on `stt.py` for up to 30 minutes, and
  Modal starts a container per caller. This is the expensive one.
- OOM an `stt.py` container by streaming audio faster than realtime. The inbound
  queue and the sample buffer are both uncapped.
- Read the Swagger UI `tts.py` serves at `/docs`.

`llm.py` also runs SGLang with `--trust-remote-code`, which executes Python from
whatever Hugging Face repo `MODEL_NAME` names, in a container that has your
`HF_TOKEN` mounted. Fine while that constant points at the pinned NVIDIA repo it
ships with. Read the repo before you point it somewhere else.

`modal app stop` is the control that actually matters. Run it when class ends.

### Closing them

Modal checks proxy auth at the edge, so an unauthenticated request never starts a
container. Three one-line changes:

```python
# stt.py
@modal.asgi_app(requires_proxy_auth=True)

# tts.py
@modal.fastapi_endpoint(docs=False, method="POST", requires_proxy_auth=True)

# llm.py, in the @app.server(...) block
unauthenticated=False,
```

Create a token under Proxy Auth Tokens in your Modal workspace settings, then send
it as `Modal-Key` and `Modal-Secret` headers. See
https://modal.com/docs/guide/webhook-proxy-auth

The LiveKit side absorbs this without a rewrite. `openai.LLM` takes a `client`
argument, so the LLM is one preconfigured `AsyncClient` with `default_headers`,
and the STT and TTS plugin clients are ours to configure. The lesson still holds:
the plugin interface is the contract, not the vendor.
