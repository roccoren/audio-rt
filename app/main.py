import base64
import binascii
import logging
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.voice_client import (
    AzureVoiceClient,
    AzureVoiceClientError,
    ZARA_INSTRUCTIONS,
    load_env_config,
    pcm16_to_wav_bytes,
)


app = FastAPI(title="Zara Voice Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger = logging.getLogger("zara.app")


class RespondRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: Optional[str] = Field(default=None, description="Optional text prompt from the user.")
    audio_base64: Optional[str] = Field(
        default=None,
        alias="audioBase64",
        description="Base64 encoded 16-bit PCM audio recorded from the browser.",
    )
    sample_rate: Optional[int] = Field(
        default=None,
        alias="sampleRate",
        description="Sample rate of the provided audio.",
    )
    voice: Optional[str] = Field(default=None, description="Azure voice name override.")


class RespondResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    transcript: str
    audio_base64: str = Field(alias="audioBase64")
    audio_sample_rate: int = Field(alias="audioSampleRate")


class RealtimeSessionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    voice: Optional[str] = Field(default=None, description="Azure voice name override.")
    instructions: Optional[str] = Field(
        default=None, description="Optional instructions override for the realtime session."
    )


class RealtimeSessionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_secret: str = Field(alias="clientSecret")
    session_id: str = Field(alias="sessionId")
    expires_at: Optional[str] = Field(default=None, alias="expiresAt")
    ice_servers: list[dict[str, object]] = Field(
        default_factory=list,
        alias="iceServers",
        description="ICE servers that should be used when constructing the RTCPeerConnection.",
    )
    sdp_url: Optional[str] = Field(
        default=None,
        alias="sdpUrl",
        description="RTC handshake endpoint for WebRTC clients.",
    )
    ws_url: Optional[str] = Field(
        default=None,
        alias="wsUrl",
        description="WebSocket endpoint (including query) for realtime clients.",
    )
    ws_protocols: Optional[list[str]] = Field(
        default=None,
        alias="wsProtocols",
        description="Optional list of WebSocket subprotocols recommended by Azure.",
    )
    session_expires_at: Optional[str] = Field(
        default=None,
        alias="sessionExpiresAt",
        description="Expiry for the realtime session itself when provided by Azure.",
    )


class RealtimeHandshakeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    offer_sdp: str = Field(alias="offerSdp", description="Browser-generated SDP offer.")
    client_secret: str = Field(alias="clientSecret", description="Ephemeral client secret for the realtime session.")


class RealtimeHandshakeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    answer_sdp: str = Field(alias="answerSdp", description="Azure-generated SDP answer.")
    ice_servers: list[dict[str, object]] = Field(
        default_factory=list,
        alias="iceServers",
        description="ICE servers returned from session creation.",
    )
    session_id: Optional[str] = Field(default=None, alias="sessionId")


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/respond", response_model=RespondResponse)
async def respond(payload: RespondRequest) -> RespondResponse:
    if not payload.text and not payload.audio_base64:
        raise HTTPException(status_code=400, detail="Provide text or audioBase64 in the request.")

    audio_chunks = None
    if payload.audio_base64:
        try:
            audio_bytes = base64.b64decode(payload.audio_base64)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="audioBase64 must be valid base64 data.") from exc
        audio_chunks = [audio_bytes]

    config = load_env_config(
        sample_rate_override=payload.sample_rate,
        voice_override=payload.voice,
    )
    client = AzureVoiceClient(config)

    try:
        transcript, audio_bytes = await client.generate_reply(
            user_text=payload.text,
            input_audio_chunks=audio_chunks,
            instructions=ZARA_INSTRUCTIONS,
            voice=payload.voice,
        )
    except AzureVoiceClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    wav_bytes = pcm16_to_wav_bytes(audio_bytes, config.sample_rate_hz)
    audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
    return RespondResponse(
        transcript=transcript,
        audioBase64=audio_base64,
        audioSampleRate=config.sample_rate_hz,
    )


@app.post("/api/realtime/handshake", response_model=RealtimeHandshakeResponse)
async def realtime_handshake(payload: RealtimeHandshakeRequest) -> RealtimeHandshakeResponse:
    offer_sdp = payload.offer_sdp or ""
    if not offer_sdp.strip():
        raise HTTPException(status_code=400, detail="SDP offer must be provided.")

    config = load_env_config()
    if not config.realtime_host:
        raise HTTPException(
            status_code=500,
            detail="Set AZURE_OPENAI_REALTIME_HOST to the regional realtime endpoint "
            "(e.g., https://swedencentral.realtimeapi-preview.ai.azure.com).",
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                config.realtime_host.rstrip("/") + "/v1/realtimertc",
                headers={
                    "Authorization": f"Bearer {payload.client_secret}",
                    "Content-Type": "application/sdp",
                    "Accept": "application/sdp",
                    "OpenAI-Beta": "realtime=v1",
                },
                content=offer_sdp,
            )
    except httpx.HTTPError as exc:
        logger.exception("Realtime handshake request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Realtime handshake failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip() or response.reason_phrase or "Unknown error"
        logger.error(
            "Realtime handshake error response: status=%s body=%s headers=%s",
            response.status_code,
            detail,
            dict(response.headers),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Realtime handshake error ({response.status_code}): {detail}",
        )

    answer_sdp: Optional[str] = response.text

    if not answer_sdp:
        logger.error(
            "Realtime handshake succeeded but answer SDP missing. Headers=%s",
            dict(response.headers),
        )
        raise HTTPException(status_code=502, detail="Realtime handshake succeeded but no SDP answer was returned.")

    return RealtimeHandshakeResponse(answerSdp=answer_sdp, iceServers=[], sessionId=None)


@app.post("/api/realtime/session", response_model=RealtimeSessionResponse)
async def realtime_session(payload: RealtimeSessionRequest) -> RealtimeSessionResponse:
    config = load_env_config(voice_override=payload.voice)

    session_body: dict[str, object] = {
        "model": config.deployment,
        "voice": payload.voice or config.voice,
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": {"type": "server_vad"},
    }
    instructions = (payload.instructions or ZARA_INSTRUCTIONS).strip()
    if instructions:
        session_body["instructions"] = instructions

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            session_response = await client.post(
                f"{config.endpoint}/openai/realtimeapi/sessions",
                params={"api-version": config.api_version},
                headers={
                    "api-key": config.api_key,
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "realtime=v1",
                },
                json=session_body,
            )
    except httpx.HTTPError as exc:
        logger.exception("Realtime session creation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Realtime session creation failed: {exc}") from exc

    if session_response.status_code >= 400:
        detail = session_response.text.strip() or session_response.reason_phrase or "Unknown error"
        logger.error(
            "Realtime session creation error: status=%s body=%s headers=%s",
            session_response.status_code,
            detail,
            dict(session_response.headers),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Realtime session creation error ({session_response.status_code}): {detail}",
        )

    try:
        session_payload = session_response.json()
    except ValueError as exc:
        logger.exception("Invalid JSON from realtime session creation: %s", session_response.text)
        raise HTTPException(status_code=502, detail="Realtime session creation returned invalid JSON.") from exc

    session_info = session_payload.get("session") or session_payload

    client_secret = ""
    client_secret_payload = session_info.get("client_secret")
    if isinstance(client_secret_payload, dict):
        client_secret = (
            client_secret_payload.get("value")
            or client_secret_payload.get("secret")
            or client_secret_payload.get("client_secret")
            or ""
        )
        expires_at = client_secret_payload.get("expires_at")
    elif isinstance(client_secret_payload, str):
        client_secret = client_secret_payload
        expires_at = None
    else:
        expires_at = None

    if not client_secret:
        logger.error("Realtime session creation succeeded but no client secret was returned: %s", session_payload)
        raise HTTPException(status_code=502, detail="Realtime session did not return a client secret.")

    ice_servers = session_info.get("ice_servers", []) or []
    session_id = session_info.get("id", "")
    session_expires_at = session_info.get("expires_at")
    webrtc_info = session_info.get("webrtc") or {}
    websocket_url = session_info.get("websocket_url") or session_info.get("ws_url")
    websocket_info = session_info.get("websocket") or {}
    if isinstance(websocket_info, dict):
        websocket_url = websocket_url or websocket_info.get("url") or websocket_info.get("value")
        websocket_protocols = websocket_info.get("protocols")
        if isinstance(websocket_protocols, list):
            websocket_protocols = [str(item) for item in websocket_protocols if isinstance(item, (str, bytes))]
        else:
            websocket_protocols = None
    else:
        websocket_protocols = None
    if isinstance(websocket_url, dict):
        websocket_url = websocket_url.get("url") or websocket_url.get("value")

    expires_value: Optional[str]
    if expires_at is None:
        expires_value = None
    else:
        expires_value = str(expires_at)

    endpoint_lower = (config.endpoint or "").lower()
    if config.endpoint.startswith("http://"):
        ws_scheme = "ws://"
        ws_base = config.endpoint[len("http://") :]
    elif config.endpoint.startswith("https://"):
        ws_scheme = "wss://"
        ws_base = config.endpoint[len("https://") :]
    else:
        ws_scheme = "wss://"
        ws_base = config.endpoint

    is_azure_resource = "openai.azure.com" in endpoint_lower
    ws_path = "/openai/v1/realtime" if is_azure_resource else "/v1/realtime"

    ws_query_params: dict[str, str] = {}
    if is_azure_resource:
        ws_query_params["deployment"] = config.deployment
    else:
        ws_query_params["model"] = config.deployment
    if is_azure_resource:
        ws_query_params["model"] = config.deployment
        if config.api_key:
            ws_query_params["api-key"] = config.api_key
    if session_id:
        ws_query_params.setdefault("session", session_id)
    if config.api_version:
        ws_query_params.setdefault("api-version", config.api_version)
    ws_query = urlencode(ws_query_params)
    computed_ws_url = f"{ws_scheme}{ws_base}{ws_path}?{ws_query}"

    def ensure_ws_query(url_value: str) -> str:
        try:
            parsed = urlparse(url_value)
        except ValueError:
            return url_value
        query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        updated = False
        if "model" not in query_params:
            query_params["model"] = config.deployment
            updated = True
        if "deployment" not in query_params:
            query_params["deployment"] = config.deployment
            updated = True
        if is_azure_resource and config.api_key and "api-key" not in query_params:
            query_params["api-key"] = config.api_key
            updated = True
        if session_id and "session" not in query_params:
            query_params["session"] = session_id
            updated = True
        if config.api_version and "api-version" not in query_params:
            query_params["api-version"] = config.api_version
            updated = True
        if not updated:
            return url_value
        new_query = urlencode(query_params)
        return urlunparse(parsed._replace(query=new_query))

    normalized_websocket_url = None
    if isinstance(websocket_url, str) and websocket_url:
        normalized_websocket_url = ensure_ws_query(websocket_url)

    return RealtimeSessionResponse(
        clientSecret=client_secret,
        sessionId=session_id,
        expiresAt=expires_value,
        iceServers=ice_servers,
        sdpUrl=(webrtc_info.get("sdp_url") if isinstance(webrtc_info, dict) else None)
        or (config.realtime_host.rstrip("/") + "/v1/realtimertc" if config.realtime_host else None),
        wsUrl=normalized_websocket_url or computed_ws_url,
        wsProtocols=websocket_protocols,
        sessionExpiresAt=str(session_expires_at) if session_expires_at is not None else None,
    )
