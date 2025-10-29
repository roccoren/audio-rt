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
  - Optional Azure VoiceLive (preview):
    - `AZURE_VOICELIVE_ENDPOINT`
    - `AZURE_VOICELIVE_MODEL`
    - `AZURE_VOICELIVE_API_VERSION` (defaults to `2025-10-01`)
    - `AZURE_VOICELIVE_API_KEY` (required when not using AAD credentials)
  - Optional Live Interpreter translation:
    - `AZURE_SPEECH_TRANSLATION_KEY` (defaults to `AZURE_SPEECH_KEY` when present)
    - `AZURE_SPEECH_TRANSLATION_ENDPOINT` or `AZURE_SPEECH_TRANSLATION_REGION`
    - Optional overrides:
      - `AZURE_SPEECH_TRANSLATION_VOICE` (set to `personal-voice` when using Personal Voice)
      - `AZURE_SPEECH_TRANSLATION_PERSONAL_VOICE_NAME` (override the personal voice identifier, default `personal-voice`)
      - `AZURE_SPEECH_TRANSLATION_TARGET_LANGUAGES` (comma separated, default `en`)
      - `AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGE` (default recognition language, default `en-US`)
      - `AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGES` (comma separated)
      - `AZURE_SPEECH_TRANSLATION_AUTO_DETECT` (defaults to `true`)
  - Optional personal voice support (reused by Live Interpreter when present):
    - `AZURE_SPEECH_KEY`
    - `AZURE_SPEECH_REGION`
    - `AZURE_SPEECH_SPEAKER_PROFILE_ID` (speaker profile created via Azure Personal Voice)
    - Optional: `AZURE_SPEECH_VOICE_NAME`, `AZURE_SPEECH_VOICE_STYLE`, `AZURE_SPEECH_LANGUAGE`

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

### Weather Lookup Helper (Optional)

To enable the async `lookup_current_weather` helper in `src/weather_lookup.py`, supply a WeatherAPI.com key:

```bash
export WEATHER_API_KEY="your-weatherapi-key"
```

Optional settings include `WEATHER_API_BASE_URL`, `WEATHER_API_TIMEOUT` (seconds), and `WEATHER_API_LANG` for localized condition text. The helper follows the official WeatherAPI.com Swagger definition for the `/current.json` endpoint.

When this key is present, the VoiceLive bridge advertises a `get_current_weather` tool so the assistant can call WeatherAPI.com and summarize live conditions during a session.

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

`/api/translate` accepts base64-encoded PCM audio and relays it to the Azure Speech translation (Live Interpreter) endpoint. It returns the recognized source text, a map of target language translations, and—when a voice is configured—the synthesized translation audio as base64-encoded WAV.

`/api/realtime/session` accepts a `provider` field so you can choose between the default GPT Realtime deployment (`"gpt-realtime"`) and Azure VoiceLive (`"voicelive"`). When `provider` is `"voicelive"` the backend simply mints the VoiceLive client secret and hands the WebSocket URL back to the browser without brokering the audio stream.

> VoiceLive support currently expects a valid `AZURE_VOICELIVE_API_KEY`. If you plan to authenticate with Azure AD tokens you will need to extend the backend to obtain a bearer token and swap the request headers accordingly.

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
- Pick **GPT Realtime** or **VoiceLive** before starting a realtime call. VoiceLive uses a direct WebSocket connection and automatically disables the WebRTC transport toggle.
- Click **开始实时通话** to start the selected realtime transport. The button flips to **结束实时通话** while connected so you can hang up at any time. Realtime mode negotiates SDP through the backend, streams microphone audio, and plays Zara’s voice as soon as she speaks. Make sure your environment variables include `AZURE_OPENAI_REALTIME_HOST` (for GPT Realtime) or the VoiceLive counterparts when using VoiceLive. VoiceLive deployments hosted on Azure OpenAI typically expose the session API at `https://<resource-name>.cognitiveservices.azure.com/openai/voicelive/sessions?api-version=<version>`.
- In bridged VoiceLive mode the browser connects to `ws://<backend>/api/voicelive/ws`; the FastAPI service handles the Azure VoiceLive SDK connection, so set `AZURE_VOICELIVE_API_KEY` and ensure the backend has the `azure-ai-voicelive` package installed.

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

## Azure Personal Voice (Optional)

If you have created a Personal Voice speaker profile via Azure Speech, the backend can synthesize replies with your custom voice instead of an Azure OpenAI stock voice.

1. Provision a Personal Voice speaker profile using the Speech SDK tooling. The helper script `src/personal_voice_setup.py` mirrors the official sample and expects the `customvoice` helpers from the Azure Speech samples repository:

   ```bash
   git clone https://github.com/Azure-Samples/cognitive-services-speech-sdk.git
   cp -R cognitive-services-speech-sdk/samples/custom-voice/python/customvoice ./customvoice

   # Create the resources (adjust IDs, paths, names)
   python src/personal_voice_setup.py create \
     --project-id personal-voice-project-1 \
     --consent-id personal-voice-consent-1 \
     --personal-voice-id personal-voice-1 \
     --consent-file TestData/VoiceTalentVerbalStatement.wav \
     --audio-folder TestData/voice \
     --voice-talent-name "Sample Voice Actor" \
     --company-name "Contoso"

   # Capture the printed speaker profile id, then you can synthesize test audio
   python src/personal_voice_setup.py synthesize \
     --speaker-profile-id <speaker_profile_id> \
     --text "This is zero shot voice. Test 2." \
     --output output_sdk.wav

   # Inspect the existing personal voices (set --project-id to narrow results)
   python src/personal_voice_setup.py list --json
   ```

2. Set these environment variables before starting the FastAPI service:

   ```bash
   export AZURE_SPEECH_KEY="..."
   export AZURE_SPEECH_REGION="eastus"
   export AZURE_SPEECH_SPEAKER_PROFILE_ID="<speaker_profile_id>"
   export AZURE_SPEECH_VOICE_NAME="DragonLatestNeural"   # optional override
   export AZURE_SPEECH_VOICE_STYLE="Prompt"              # optional override
   export AZURE_SPEECH_LANGUAGE="en-US"                  # optional override
   ```

With the values in place, every reply streamed from Azure OpenAI is re-synthesized locally using the Personal Voice speaker profile before being returned to the browser/CLI. Remove or unset the speech environment variables to fall back to the stock Azure OpenAI voices.

The same speaker profile settings also enable Personal Voice synthesis for Azure Speech Live Interpreter. Set `AZURE_SPEECH_TRANSLATION_VOICE="personal-voice"` (or override with `AZURE_SPEECH_TRANSLATION_PERSONAL_VOICE_NAME`) and the backend will automatically route translation requests to the universal/v2 endpoint with the zero-shot TTS flight enabled.

## Files

- `src/voice_client.py` – Azure Voice Live WebSocket client
- `src/run_zara_voice.py` – CLI entry point
- `src/personal_voice_setup.py` – helper to create/manage Azure Personal Voice assets
- `app/main.py` – FastAPI bridge between browser and Azure OpenAI
- `requirements.txt` – Python dependencies
- `web/` – React + Vite application for audio chat

## Notes

- The script stores each generated reply as a timestamped WAV file under the `out/` directory.
- The persona instructions in `src/voice_client.py` follow the specification supplied in the task description.
- Network access is required to run the demo against Azure OpenAI; the SDK is not bundled here.
- Browser recording relies on the MediaRecorder API (Chrome, Edge, and Firefox desktop all support it).

## Deployment: Azure Container Apps

You can deploy both the FastAPI backend and the Vite/React frontend to [Azure Container Apps](https://learn.microsoft.com/azure/container-apps/overview). The workflow below assumes you have the Azure CLI installed (v2.53+), the `containerapp` extension added, and you are already logged in (`az login`).

```bash
az extension add --name containerapp
```

1. **Set environment variables** (adjust the names, location, and image tags as needed):

   ```bash
   export RESOURCE_GROUP=voice-group
   export LOCATION=southeastasia
   export ACR_NAME=ankeraiprecr
   export CONTAINERAPPS_ENV=zara-voice-env
   export BACKEND_APP=zara-voice-api
   export FRONTEND_APP=zara-voice-web
   export IMAGE_TAG=v1.0
   ```

   > Tip: run `az account list-locations -o table` if you need to pick a different region.

2. **Provision Azure resources** (resource group, container registry, and a Container Apps environment):

   ```bash
   az group create --name $RESOURCE_GROUP --location $LOCATION
   az acr create --name $ACR_NAME --resource-group $RESOURCE_GROUP --sku Basic
   ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer -o tsv)
   az containerapp env create --name $CONTAINERAPPS_ENV --resource-group $RESOURCE_GROUP --location $LOCATION
   ```

3. **Build and push the backend image** (uses the `Dockerfile` at the repository root):

   ```bash
   az acr build \
     --registry $ACR_NAME \
     --image zara-backend:$IMAGE_TAG \
     -f Dockerfile \
     .
   BACKEND_IMAGE="$ACR_LOGIN_SERVER/zara-backend:$IMAGE_TAG"
   ```

4. **Create the backend container app**. First load your Azure secrets into variables (never commit them):

   ```bash
   export AZURE_OPENAI_ENDPOINT="https://<your-azure-openai-resource>.openai.azure.com"
   export AZURE_OPENAI_API_KEY="<azure-openai-api-key>"
   export AZURE_OPENAI_DEPLOYMENT="gpt-realtime-mini"
   export AZURE_OPENAI_REALTIME_HOST="https://<region>.realtimeapi-preview.ai.azure.com"
   export AZURE_OPENAI_VOICE="alloy"
   export AZURE_OPENAI_SAMPLE_RATE="24000"
   export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
   export AZURE_VOICELIVE_ENDPOINT="https://<your-voicelive-resource>.cognitiveservices.azure.com"
   export AZURE_VOICELIVE_MODEL="gpt-4.1"
   export AZURE_VOICELIVE_API_VERSION="2025-10-01"
   export AZURE_VOICELIVE_API_KEY="<azure-voicelive-key>"
   export AZURE_SPEECH_TRANSLATION_KEY="<azure-speech-translation-key>"
   export AZURE_SPEECH_TRANSLATION_REGION="southeastasia"
   export AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGES="zh-CN"
   export AZURE_SPEECH_TRANSLATION_TARGET_LANGUAGE="en"
   export AZURE_SPEECH_TRANSLATION_VOICE="personal-voice"
   export AZURE_SPEECH_TRANSLATION_AUTO_DETECT="true"
   ```

   Create (or update) the backend container app, providing the registry credentials, secrets, and environment variable bindings in one step:

   ```bash
   az containerapp create \
     --name $BACKEND_APP \
     --resource-group $RESOURCE_GROUP \
     --environment $CONTAINERAPPS_ENV \
     --image $BACKEND_IMAGE \
     --target-port 8000 \
     --ingress external \
     --min-replicas 1 \
     --max-replicas 1 \
     --registry-server $ACR_LOGIN_SERVER \
     --registry-username $(az acr credential show --name $ACR_NAME --query username -o tsv) \
     --registry-password $(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv) \
     --secrets \
       azure-openai-endpoint=$AZURE_OPENAI_ENDPOINT \
       azure-openai-api-key=$AZURE_OPENAI_API_KEY \
       azure-openai-deployment=$AZURE_OPENAI_DEPLOYMENT \
       azure-openai-realtime-host=$AZURE_OPENAI_REALTIME_HOST \
       azure-openai-voice=$AZURE_OPENAI_VOICE \
       azure-openai-sample-rate=$AZURE_OPENAI_SAMPLE_RATE \
       azure-openai-api-version=$AZURE_OPENAI_API_VERSION \
       azure-voicelive-endpoint=$AZURE_VOICELIVE_ENDPOINT \
       azure-voicelive-model=$AZURE_VOICELIVE_MODEL \
       azure-voicelive-api-version=$AZURE_VOICELIVE_API_VERSION \
       azure-voicelive-api-key=$AZURE_VOICELIVE_API_KEY \
       azure-speech-translation-key=$AZURE_SPEECH_TRANSLATION_KEY \
       azure-speech-translation-region=$AZURE_SPEECH_TRANSLATION_REGION \
       azure-speech-translation-source-languages=$AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGES \
       azure-speech-translation-target-language=$AZURE_SPEECH_TRANSLATION_TARGET_LANGUAGE \
       azure-speech-translation-voice=$AZURE_SPEECH_TRANSLATION_VOICE \
       azure-speech-translation-auto-detect=$AZURE_SPEECH_TRANSLATION_AUTO_DETECT \
     --env-vars \
       AZURE_OPENAI_ENDPOINT=secretref:azure-openai-endpoint \
       AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key \
       AZURE_OPENAI_DEPLOYMENT=secretref:azure-openai-deployment \
       AZURE_OPENAI_REALTIME_HOST=secretref:azure-openai-realtime-host \
       AZURE_OPENAI_VOICE=secretref:azure-openai-voice \
       AZURE_OPENAI_SAMPLE_RATE=secretref:azure-openai-sample-rate \
       AZURE_OPENAI_API_VERSION=secretref:azure-openai-api-version \
       AZURE_VOICELIVE_ENDPOINT=secretref:azure-voicelive-endpoint \
       AZURE_VOICELIVE_MODEL=secretref:azure-voicelive-model \
       AZURE_VOICELIVE_API_VERSION=secretref:azure-voicelive-api-version \
       AZURE_VOICELIVE_API_KEY=secretref:azure-voicelive-api-key \
       AZURE_SPEECH_TRANSLATION_KEY=secretref:azure-speech-translation-key \
       AZURE_SPEECH_TRANSLATION_REGION=secretref:azure-speech-translation-region \
       AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGES=secretref:azure-speech-translation-source-languages \
       AZURE_SPEECH_TRANSLATION_TARGET_LANGUAGE=secretref:azure-speech-translation-target-language \
       AZURE_SPEECH_TRANSLATION_VOICE=secretref:azure-speech-translation-voice \
       AZURE_SPEECH_TRANSLATION_AUTO_DETECT=secretref:azure-speech-translation-auto-detect
   ```

   When the command finishes, capture the public Fully Qualified Domain Name (FQDN):

   ```bash
   BACKEND_FQDN=$(az containerapp show --name $BACKEND_APP --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)
   echo "Backend reachable at: https://$BACKEND_FQDN"
   ```

5. **Build and push the frontend image** once the backend endpoint is known. The Vite build embeds the backend URL, so pass it as a build argument:

   ```bash
   az acr build \
     --registry $ACR_NAME \
     --image zara-frontend:$IMAGE_TAG \
     --file web/Dockerfile \
     --build-arg VITE_API_BASE_URL="https://$BACKEND_FQDN" \
     .
   FRONTEND_IMAGE="$ACR_LOGIN_SERVER/zara-frontend:$IMAGE_TAG"
   ```

6. **Create the frontend container app** (static content served by Nginx):

   ```bash
   az containerapp create \
     --name $FRONTEND_APP \
     --resource-group $RESOURCE_GROUP \
     --environment $CONTAINERAPPS_ENV \
     --image $FRONTEND_IMAGE \
     --ingress external \
     --target-port 80 \
     --min-replicas 1 \
     --max-replicas 1 \
     --registry-server $ACR_LOGIN_SERVER \
     --registry-username $(az acr credential show --name $ACR_NAME --query username -o tsv) \
     --registry-password $(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
   ```

   Retrieve the frontend FQDN:

   ```bash
   az containerapp show --name $FRONTEND_APP --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv
   ```

7. **Future updates** just require a new image tag and a `az containerapp update` for the affected service:

   ```bash
   export IMAGE_TAG=v0.1.1
   az acr build --registry $ACR_NAME --image zara-backend:$IMAGE_TAG -f Dockerfile .
   az containerapp update --name $BACKEND_APP --resource-group $RESOURCE_GROUP --image "$ACR_LOGIN_SERVER/zara-backend:$IMAGE_TAG"
   ```

   Repeat the same pattern for the frontend (remember to pass the backend URL build argument when rebuilding the web image).

For production setups consider using managed identities for ACR pulls, locking down the backend ingress to internal traffic only, and fronting the container apps with Azure Front Door or Application Gateway for TLS and WAF management.
