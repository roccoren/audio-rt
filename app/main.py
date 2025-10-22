import asyncio
import base64
import binascii
import json
import logging
from functools import lru_cache
from typing import Literal, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from azure.ai.voicelive.aio import connect as voicelive_connect
from azure.ai.voicelive.models import (
    AzureStandardVoice,
    InputAudioFormat,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerVad,
)
from azure.core.credentials import AzureKeyCredential

from src.voice_client import (
    AzureVoiceClient,
    AzureVoiceClientError,
    ZARA_INSTRUCTIONS,
    load_env_config,
    load_voicelive_env_config,
    pcm16_to_wav_bytes,
)
from src.live_interpreter import (
    LiveInterpreterError,
    LiveInterpreterTranslator,
    load_live_interpreter_config,
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
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


@lru_cache()
def _get_live_interpreter() -> LiveInterpreterTranslator:
    config = load_live_interpreter_config()
    return LiveInterpreterTranslator(config)


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

    provider: Literal["gpt-realtime", "voicelive"] = Field(
        default="gpt-realtime", description="Realtime backend to use."
    )
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
    provider: Literal["gpt-realtime", "voicelive"] = Field(
        default="gpt-realtime", description="Realtime backend to use."
    )


class RealtimeHandshakeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    answer_sdp: str = Field(alias="answerSdp", description="Azure-generated SDP answer.")
    ice_servers: list[dict[str, object]] = Field(
        default_factory=list,
        alias="iceServers",
        description="ICE servers returned from session creation.",
    )
    session_id: Optional[str] = Field(default=None, alias="sessionId")


class TranslateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    audio_base64: str = Field(alias="audioBase64", description="Base64-encoded PCM16 audio input.")
    sample_rate: int = Field(alias="sampleRate", description="Sample rate (Hz) of the input audio.")
    target_languages: Optional[list[str]] = Field(
        default=None,
        alias="targetLanguages",
        description="Optional override for translation target language codes.",
    )
    voice: Optional[str] = Field(
        default=None,
        description="Optional Azure speech voice name for synthesized translation.",
    )
    source_language: Optional[str] = Field(
        default=None,
        alias="sourceLanguage",
        description="Optional source language code when auto detection is disabled.",
    )
    auto_detect_source_language: Optional[bool] = Field(
        default=None,
        alias="autoDetectSourceLanguage",
        description="Override to enable/disable automatic source language detection.",
    )


class TranslateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    recognized_text: str = Field(alias="recognizedText")
    translations: dict[str, str]
    audio_base64: Optional[str] = Field(default=None, alias="audioBase64")
    audio_format: Optional[str] = Field(default=None, alias="audioFormat")
    audio_sample_rate: Optional[int] = Field(default=None, alias="audioSampleRate")
    detected_source_language: Optional[str] = Field(default=None, alias="detectedSourceLanguage")


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


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(payload: TranslateRequest) -> TranslateResponse:
    if not payload.audio_base64:
        raise HTTPException(status_code=400, detail="audioBase64 is required for translation.")
    if payload.sample_rate <= 0:
        raise HTTPException(status_code=400, detail="sampleRate must be a positive integer.")

    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="audioBase64 must be valid base64 data.") from exc

    try:
        interpreter = _get_live_interpreter()
    except LiveInterpreterError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = await interpreter.translate_audio(
            audio_bytes=audio_bytes,
            sample_rate_hz=payload.sample_rate,
            target_languages=payload.target_languages,
            voice_name=payload.voice,
            source_language=payload.source_language,
            auto_detect=payload.auto_detect_source_language,
        )
    except LiveInterpreterError as exc:
        logger.error("Live Interpreter error: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unexpected Live Interpreter failure: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the translation request.",
        ) from exc

    translated_audio_base64 = None
    if result.audio_data:
        translated_audio_base64 = base64.b64encode(result.audio_data).decode("ascii")

    return TranslateResponse(
        recognizedText=result.recognized_text,
        translations=result.translations,
        audioBase64=translated_audio_base64,
        audioFormat=result.audio_format,
        audioSampleRate=result.audio_sample_rate,
        detectedSourceLanguage=result.detected_source_language,
    )


async def _create_voicelive_session(payload: RealtimeSessionRequest) -> RealtimeSessionResponse:
    config = load_voicelive_env_config(voice_override=payload.voice)

    instructions = (payload.instructions or config.instructions or ZARA_INSTRUCTIONS).strip()
    voice_name = payload.voice or config.voice

    session_body: dict[str, object] = {
        "model": config.model,
        "modalities": ["audio"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        },
    }
    if instructions:
        session_body["instructions"] = instructions
    if voice_name:
        session_body["voice"] = voice_name

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if config.api_key:
        headers["api-key"] = config.api_key
        headers["Api-Key"] = config.api_key

    base_endpoint = config.endpoint.rstrip("/")
    if ".cognitiveservices.azure.com" in base_endpoint:
        base_endpoint = base_endpoint.replace(".cognitiveservices.azure.com", ".openai.azure.com")

    params: dict[str, str] = {}
    if config.api_version:
        params["api-version"] = config.api_version
    if config.model:
        params.setdefault("model", config.model)

    is_azure_openai = ".openai.azure.com" in base_endpoint
    if is_azure_openai and config.model:
        params.setdefault("deployment", config.model)

    candidate_paths = [
        "",  # direct /sessions
        "/v1",
        "/voicelive",
        "/voice-live",
        "/voicelive/v1",
        "/voice-live/v1",
        "/openai/voice-live",
        "/openai/voice-live/v1",
        "/ai",
        "/openai",
        "/openai/realtimeapi",
        "/openai/voicelive",
        "/voiceliveapi",
    ]
    candidate_urls: list[str] = []
    for path in candidate_paths:
        path = path.rstrip("/")
        if path:
            candidate_urls.append(f"{base_endpoint}{path}/sessions")
        else:
            candidate_urls.append(f"{base_endpoint}/sessions")
    candidate_urls = list(dict.fromkeys(candidate_urls))

    session_response: Optional[httpx.Response] = None
    last_error: Optional[Exception] = None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for idx, session_url in enumerate(candidate_urls):
                try:
                    response = await client.post(session_url, headers=headers, params=params, json=session_body)
                except httpx.HTTPError as exc:
                    last_error = exc
                    logger.debug("VoiceLive session request to %s failed: %s", session_url, exc)
                    continue

                if response.status_code == 404 and idx < len(candidate_urls) - 1:
                    logger.debug(
                        "VoiceLive session URL %s returned 404; trying next fallback (status=%s body=%s).",
                        session_url,
                        response.status_code,
                        response.text,
                    )
                    continue

                session_response = response
                logger.info(
                    "VoiceLive session request candidate=%s status=%s", session_url, response.status_code
                )
                break
    except httpx.HTTPError as exc:
        logger.exception("VoiceLive session creation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"VoiceLive session creation failed: {exc}") from exc

    if session_response is None:
        if last_error:
            raise HTTPException(status_code=502, detail=f"VoiceLive session creation failed: {last_error}") from last_error
        raise HTTPException(status_code=502, detail="VoiceLive session creation failed: no valid endpoint responded.")

    if session_response.status_code >= 400:
        detail = session_response.text.strip() or session_response.reason_phrase or "Unknown error"
        logger.error(
            "VoiceLive session creation error: status=%s body=%s headers=%s",
            session_response.status_code,
            detail,
            dict(session_response.headers),
        )
        raise HTTPException(
            status_code=502,
            detail=f"VoiceLive session creation error ({session_response.status_code}): {detail}",
        )

    try:
        session_payload = session_response.json()
    except ValueError as exc:
        logger.exception("VoiceLive session returned invalid JSON: %s", session_response.text)
        raise HTTPException(status_code=502, detail="VoiceLive session creation returned invalid JSON.") from exc

    session_info = session_payload.get("session") or session_payload
    if not isinstance(session_info, dict):
        logger.error("VoiceLive session payload missing 'session' object: %s", session_payload)
        raise HTTPException(status_code=502, detail="VoiceLive session payload was not in the expected format.")

    client_secret_value = (
        session_info.get("client_secret")
        or session_info.get("clientSecret")
        or session_info.get("secret")
        or session_info.get("session_secret")
    )
    client_secret = ""
    expires_at = None
    if isinstance(client_secret_value, dict):
        client_secret = (
            client_secret_value.get("value")
            or client_secret_value.get("secret")
            or client_secret_value.get("client_secret")
            or ""
        )
        expires_at = client_secret_value.get("expires_at") or client_secret_value.get("expiry")
    elif isinstance(client_secret_value, str):
        client_secret = client_secret_value
    if not client_secret:
        logger.error("VoiceLive session response missing client secret: %s", session_payload)
        raise HTTPException(status_code=502, detail="VoiceLive session did not return a client secret.")

    session_id = (
        session_info.get("id")
        or session_info.get("session_id")
        or session_info.get("sessionId")
        or session_info.get("session")
        or ""
    )

    webrtc_info = session_info.get("webrtc") or session_info.get("rtc") or {}
    websocket_info = session_info.get("websocket") or {}

    ice_servers = session_info.get("ice_servers") or session_info.get("iceServers")
    if not ice_servers and isinstance(webrtc_info, dict):
        ice_servers = webrtc_info.get("ice_servers") or webrtc_info.get("iceServers")
    if not isinstance(ice_servers, list):
        ice_servers = []

    websocket_url = (
        session_info.get("websocket_url")
        or session_info.get("ws_url")
        or session_info.get("wsUrl")
        or websocket_info.get("url")
        or websocket_info.get("value")
    )
    websocket_protocols = websocket_info.get("protocols") if isinstance(websocket_info, dict) else None
    if not websocket_url and isinstance(session_info.get("websocket"), str):
        websocket_url = session_info["websocket"]
    if isinstance(websocket_protocols, list):
        websocket_protocols = [str(item) for item in websocket_protocols if isinstance(item, (str, bytes))]
    else:
        websocket_protocols = None
    if not websocket_url:
        logger.error("VoiceLive session response missing websocket URL: %s", session_payload)
        raise HTTPException(status_code=502, detail="VoiceLive session did not return a websocket URL.")

    sdp_url = None
    if isinstance(webrtc_info, dict):
        sdp_url = webrtc_info.get("sdp_url") or webrtc_info.get("sdpUrl")

    session_expires_at = session_info.get("expires_at") or session_info.get("session_expires_at")

    expires_value = str(expires_at) if expires_at is not None else None
    session_expires_value = str(session_expires_at) if session_expires_at is not None else None

    return RealtimeSessionResponse(
        clientSecret=client_secret,
        sessionId=str(session_id),
        expiresAt=expires_value,
        iceServers=ice_servers,
        sdpUrl=sdp_url,
        wsUrl=websocket_url,
        wsProtocols=websocket_protocols,
        sessionExpiresAt=session_expires_value,
    )


DEFAULT_AZURE_VOICE = "en-US-Ava:DragonHDLatestNeural"


def _resolve_voicelive_voice(voice: Optional[str]) -> Union[AzureStandardVoice, str]:
    voice_value = (voice or "").strip()
    if not voice_value:
        voice_value = DEFAULT_AZURE_VOICE
    if voice_value.startswith("en-US-") or voice_value.startswith("en-CA-"):
        return AzureStandardVoice(name=voice_value, type="azure-standard")
    # Fallback to default Azure voice when a non-Azure voice is supplied.
    return AzureStandardVoice(name=DEFAULT_AZURE_VOICE, type="azure-standard")


def _build_voicelive_session_config(instructions: str, voice: Optional[str]) -> RequestSession:
    resolved_instructions = (instructions or ZARA_INSTRUCTIONS).strip() or ZARA_INSTRUCTIONS
    return RequestSession(
        modalities=[Modality.TEXT, Modality.AUDIO],
        instructions=resolved_instructions,
        voice=_resolve_voicelive_voice(voice),
        input_audio_format=InputAudioFormat.PCM16,
        output_audio_format=OutputAudioFormat.PCM16,
        turn_detection=ServerVad(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500),
    )


def _sanitize_voicelive_payload(value: object) -> object:
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {key: _sanitize_voicelive_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_voicelive_payload(item) for item in value]
    return value


def _ensure_modality_list(values: Optional[list]) -> list[Modality]:
    result: list[Modality] = []
    for item in values or []:
        if isinstance(item, Modality):
            result.append(item)
            continue
        if isinstance(item, str):
            lowered = item.lower()
            if "audio" in lowered:
                result.append(Modality.AUDIO)
                continue
            if "text" in lowered:
                result.append(Modality.TEXT)
                continue
    if not result:
        result = [Modality.TEXT, Modality.AUDIO]
    return result


async def _dispatch_voicelive_message(
    connection,
    message: dict,
    *,
    default_voice: Optional[Union[str, AzureStandardVoice]],
) -> None:
    message_type = (message.get("type") or "").strip()
    if not message_type:
        return
    if message_type == "input_audio_buffer.append":
        audio = message.get("audio")
        if audio:
            await connection.input_audio_buffer.append(audio=audio)
        return
    if message_type == "input_audio_buffer.commit":
        await connection.input_audio_buffer.commit()
        return
    if message_type == "input_audio_buffer.clear":
        await connection.input_audio_buffer.clear()
        return
    if message_type == "session.update":
        session_payload = message.get("session") or {}
        session_payload.pop("type", None)
        turn_detection = session_payload.get("turn_detection") or session_payload.get("turnDetection")
        modalities = session_payload.get("modalities")
        if modalities is None and "output_modalities" in session_payload:
            modalities = session_payload.pop("output_modalities")
        instructions = session_payload.get("instructions") or ZARA_INSTRUCTIONS
        voice_obj: Union[str, AzureStandardVoice]
        if isinstance(default_voice, AzureStandardVoice):
            voice_obj = default_voice
        else:
            voice_obj = _resolve_voicelive_voice(default_voice)
        session_update = RequestSession(
            modalities=_ensure_modality_list(modalities),
            instructions=instructions,
            voice=voice_obj,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=turn_detection or ServerVad(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500),
        )
        await connection.session.update(session=session_update)
        return
    if message_type == "response.create":
        response_payload = dict(message.get("response") or {})
        if "output_modalities" in response_payload and "modalities" not in response_payload:
            response_payload["modalities"] = response_payload.pop("output_modalities")
        if "outputAudioFormat" in response_payload and "output_audio_format" not in response_payload:
            response_payload["output_audio_format"] = response_payload.pop("outputAudioFormat")
        if default_voice and "voice" not in response_payload:
            if isinstance(default_voice, AzureStandardVoice):
                response_payload["voice"] = default_voice.as_dict()
            else:
                response_payload["voice"] = _resolve_voicelive_voice(default_voice).as_dict()
        if "modalities" in response_payload:
            response_payload["modalities"] = _ensure_modality_list(response_payload["modalities"])
        await connection.response.create(
            response=response_payload or None,
            additional_instructions=message.get("additional_instructions")
            or message.get("additionalInstructions"),
        )
        return
    if message_type == "response.cancel":
        await connection.response.cancel(
            response_id=message.get("response_id") or message.get("responseId")
        )
        return


async def _consume_voicelive_messages(
    connection,
    websocket: WebSocket,
    logger: logging.Logger,
    *,
    default_voice: Optional[Union[str, AzureStandardVoice]],
) -> None:
    while True:
        try:
            message = await websocket.receive_text()
        except WebSocketDisconnect:
            break
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON VoiceLive client message: %s", message)
            continue
        try:
            await _dispatch_voicelive_message(connection, payload, default_voice=default_voice)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to dispatch VoiceLive message: %s", exc)
            await websocket.send_json({"type": "error", "message": f"{exc}"})


@app.websocket("/api/voicelive/ws")
async def voicelive_bridge(websocket: WebSocket) -> None:
    await websocket.accept()
    raw_query = websocket.scope.get("query_string") or b""
    query_params = dict(parse_qsl(raw_query.decode("utf-8"))) if raw_query else {}
    voice_override = query_params.get("voice")
    instructions_override = query_params.get("instructions")

    try:
        config = load_voicelive_env_config(voice_override=voice_override)
    except Exception as exc:  # noqa: BLE001
        logger.exception("VoiceLive configuration error: %s", exc)
        await websocket.send_json({"type": "error", "message": f"Configuration error: {exc}"})
        await websocket.close(code=1011, reason="Configuration error")
        return

    if not config.api_key:
        await websocket.send_json(
            {"type": "error", "message": "AZURE_VOICELIVE_API_KEY is required for bridged VoiceLive calls."}
        )
        await websocket.close(code=1011, reason="Missing VoiceLive API key")
        return

    instructions = (instructions_override or config.instructions or ZARA_INSTRUCTIONS).strip()

    credential = AzureKeyCredential(config.api_key)

    connect_endpoint = config.endpoint
    if ".cognitiveservices.azure.com" in connect_endpoint:
        connect_endpoint = connect_endpoint.replace(".cognitiveservices.azure.com", ".openai.azure.com")

    is_azure_endpoint = ".openai.azure.com" in connect_endpoint
    query: dict[str, str] = {}
    if config.model:
        query["model"] = config.model
        if is_azure_endpoint:
            query.setdefault("deployment", config.model)

    try:
        async with voicelive_connect(
            endpoint=connect_endpoint,
            credential=credential,
            model=config.model,
            api_version=config.api_version,
            query=query,
        ) as connection:
            session_created = asyncio.Event()
            session_error: dict[str, object] = {}

            async def forward_events():
                try:
                    async for event in connection:
                        if not session_created.is_set() and event.type == "session.created":
                            session_created.set()
                        payload = _sanitize_voicelive_payload(event.as_dict())
                        await websocket.send_json(payload)
                except WebSocketDisconnect:
                    pass
                except Exception as exc:  # noqa: BLE001
                    logger.exception("VoiceLive event forwarding failed: %s", exc)
                    session_error["exception"] = exc
                    session_created.set()
                    raise

            forward_task = asyncio.create_task(forward_events())

            try:
                initial_session = _build_voicelive_session_config(instructions, config.voice)
                await connection.session.update(session=initial_session)
                await asyncio.wait_for(session_created.wait(), timeout=5.0)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Initial VoiceLive session.update failed: %s", exc)
                await websocket.send_json({"type": "error", "message": f"session.update failed: {exc}"})
                forward_task.cancel()
                raise

            if session_error:
                forward_task.cancel()
                raise session_error.get("exception") or RuntimeError("VoiceLive session failed to establish.")

            session_voice = _resolve_voicelive_voice(config.voice if isinstance(config.voice, str) else None)

            consume_task = asyncio.create_task(
                _consume_voicelive_messages(connection, websocket, logger, default_voice=session_voice)
            )

            done, pending = await asyncio.wait(
                [forward_task, consume_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc:
                    raise exc
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("VoiceLive websocket bridge error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": f"{exc}"})
        except Exception:  # noqa: BLE001
            pass
        await websocket.close(code=1011, reason="VoiceLive bridge error")
        return
    finally:
        try:
            await websocket.close(code=1000)
        except Exception:  # noqa: BLE001
            pass

@app.post("/api/realtime/handshake", response_model=RealtimeHandshakeResponse)
async def realtime_handshake(payload: RealtimeHandshakeRequest) -> RealtimeHandshakeResponse:
    offer_sdp = payload.offer_sdp or ""
    if not offer_sdp.strip():
        raise HTTPException(status_code=400, detail="SDP offer must be provided.")
    if payload.provider == "voicelive":
        raise HTTPException(status_code=400, detail="VoiceLive WebRTC handshake is not supported by this endpoint.")

    config = load_env_config()
    if not config.realtime_host:
        raise HTTPException(
            status_code=500,
            detail="Set AZURE_OPENAI_REALTIME_HOST to the regional realtime endpoint "
            "(e.g., https://swedencentral.realtimeapi-preview.ai.azure.com).",
        )

    try:
        # Increase timeout for realtime handshake - Azure can be slow to respond
        timeout_config = httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
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
    except httpx.TimeoutException as exc:
        logger.exception("Realtime handshake timeout: %s", exc)
        raise HTTPException(
            status_code=504,
            detail=f"Realtime handshake timed out after 60 seconds. The Azure endpoint may be overloaded or unavailable.",
        ) from exc
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
    if payload.provider == "voicelive":
        return await _create_voicelive_session(payload)

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
