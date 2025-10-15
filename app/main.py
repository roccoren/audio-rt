import base64
import binascii
import logging
from typing import Optional

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
        client_secret = client_secret_payload.get("value", "")
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

    expires_value: Optional[str]
    if expires_at is None:
        expires_value = None
    else:
        expires_value = str(expires_at)

    return RealtimeSessionResponse(
        clientSecret=client_secret,
        sessionId=session_id,
        expiresAt=expires_value,
        iceServers=ice_servers,
        sdpUrl=config.realtime_host.rstrip("/") + "/v1/realtimertc",
    )
