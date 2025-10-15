# Azure OpenAI Voice Live Demo

This workspace contains a Python service and a modern React web app that talk to the Azure OpenAI **Voice Live** API. The Zara persona instructions are baked in so every reply sounds like the character you described.

## Prerequisites

- Python 3.10 or later
- Node.js 18 or later
- An Azure OpenAI resource with:
  - A GPT-4o Realtime (Voice Live) deployment
  - The Voice Live preview enabled on the subscription
- Environment variables:
  - `AZURE_OPENAI_ENDPOINT` – e.g. `my-resource.openai.azure.com`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT` – name of the GPT-4o (voice) deployment
  - `AZURE_OPENAI_REALTIME_HOST` – regional realtime RTC endpoint, e.g. `https://swedencentral.realtimeapi-preview.ai.azure.com`
  - Optional overrides:
    - `AZURE_OPENAI_VOICE`
    - `AZURE_OPENAI_SAMPLE_RATE`
    - `AZURE_OPENAI_API_VERSION`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and edit the values:

```bash
cp .env.template .env
```

Install web dependencies:

```bash
cd web
npm install
```

Create the web env file (update the URL if the backend runs elsewhere):

```bash
cp .env.template .env
```

## Backend: Zara Voice API

Start the FastAPI server (it exposes `/api/respond`, `/api/realtime/session`, and `/api/realtime/handshake` as a thin proxy to Azure OpenAI):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Set `PYTHONPATH` to the repository root if your shell requires it, e.g.:

```bash
PYTHONPATH=. uvicorn app.main:app --reload
```

The service expects base64-encoded 16-bit PCM audio and returns a WAV-encoded reply along with the transcript. For realtime calls it creates an ephemeral session (`/api/realtime/session`) and relays the SDP offer/answer exchange (`/api/realtime/handshake`). CORS is open for local development.

## Frontend: React Web Client

The web UI is built with Vite + React + TypeScript. Configure the API URL (defaults to `http://localhost:8000`) by copying `web/.env.template` to `web/.env` and adjusting the value if needed.

Run the development server:

```bash
cd web
npm run dev
```

Open the printed URL (default `http://localhost:5173`). You can:

- Type some text and click **发送文字** for a quick audio reply.
- Record a short clip with **开始录音 / 发送语音** and hear Zara answer with fresh audio.
- Click **开始实时通话** to start a WebRTC session with the Azure realtime endpoint. The button flips to **结束实时通话** while connected so you can hang up at any time. Realtime mode negotiates SDP through the backend, streams microphone audio, and plays Zara’s voice as soon as she speaks. Make sure your environment variables include `AZURE_OPENAI_REALTIME_HOST`.

All captured audio is resampled to 24 kHz PCM16 before being forwarded to Azure.

## CLI: Quick Text/Mic Test

The `python src/run_zara_voice.py` script opens a WebSocket session to the Voice Live endpoint, pushes a starter utterance, and saves the streamed audio response to `out/zara_response.wav`.

```bash
export AZURE_OPENAI_ENDPOINT="my-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="gpt-realtime"
python src/run_zara_voice.py --text "Hey Zara, how are you today?"
```

Use the `--microphone` flag if you want to capture microphone audio instead of typed text (macOS/Linux only, requires `sounddevice`).

## Files

- `src/voice_client.py` – Azure Voice Live WebSocket client
- `src/run_zara_voice.py` – CLI entry point
- `app/main.py` – FastAPI bridge between browser and Azure OpenAI
- `requirements.txt` – Python dependencies
- `web/` – React + Vite application for audio chat

## Notes

- The script stores each generated reply as a timestamped WAV file under the `out/` directory.
- The persona instructions in `src/voice_client.py` follow the specification supplied in the task description.
- Network access is required to run the demo against Azure OpenAI; the SDK is not bundled here.
- Browser recording relies on the MediaRecorder API (Chrome, Edge, and Firefox desktop all support it).
