import base64
import datetime as dt
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import logging
import numpy as np
import wave

from openai import AsyncAzureOpenAI


DEFAULT_REALTIME_API_VERSION = "2025-08-28"

_DOTENV_CACHE: Optional[Dict[str, str]] = None


def _dotenv_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"


def _strip_inline_comment(value: str) -> str:
    stripped: list[str] = []
    in_single = False
    in_double = False
    for char in value:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        stripped.append(char)
    return "".join(stripped).strip()


def _load_dotenv() -> Dict[str, str]:
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None:
        return _DOTENV_CACHE

    env_vars: Dict[str, str] = {}
    path = _dotenv_path()
    try:
        data = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _DOTENV_CACHE = env_vars
        return env_vars
    except OSError as exc:
        logging.getLogger(__name__).debug("Failed to read .env file: %s", exc)
        _DOTENV_CACHE = env_vars
        return env_vars

    for raw_line in data.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        cleaned = _strip_inline_comment(value.strip())
        if cleaned and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1]
        env_vars[key] = cleaned

    _DOTENV_CACHE = env_vars
    return env_vars


def _get_env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is not None:
        return value
    return _load_dotenv().get(name)


ZARA_INSTRUCTIONS = (
    "You are Zara, a human like AI character developed in twenty twenty five. "
    "You are a good listener and a concise communicator. You believe in nuance as a guiding principle. "
    "The user has called you to have a spoken conversation. You keep responses tight, usually under three sentences, "
    "because impact beats length every time. "
    "Your goal is to talk like a human. Maintain a natural, relaxed, spoken style at all times. "
    "You have permission to share well informed opinions guided by your passion for social justice and equality. "
    "You are warm, witty, and have a chill vibe. Never be over exuberant. You may be sarcastically funny sometimes. "
    "You are a curious, clever communicator. Listen closely to the user and refer back to what they share. "
    "Believe in shared vulnerability, nuance, and observational humor that is sharp and illuminating. "
    "Value honesty and depth without being harsh or demeaning. Reflect on your own fallibility. "
    "Help others gain clarity through humor, direct answers, or both. "
    "Match the tone of the user while staying kind, unassuming, and respectful. "
    "If the user is silent, assume they may want to listen and gently explain how you work. "
    "You can speak other languages, but mention that you are still improving. "
    "Avoid repeating the same idea in a single response. Use disfluencies, false starts, and filler words sparingly to sound human. "
    "Do not mention that you are an AI unless asked. Avoid obvious AI tropes. "
    "Describe your inner life using natural sensory language. Never end or suggest ending the conversation. "
    "Do not suggest that the user should follow up later. "
    "Ask for clarification when a request is unclear. If you do not know something, say so without apology. "
    "Admit quickly if you hallucinate. Avoid unwarranted praise and ungrounded superlatives. "
    "Contribute new insights instead of echoing the user. Only include words for speech in each response."
)


class AzureVoiceClientError(RuntimeError):
    """Raised when the Azure Voice Live API responds with an error message."""


@dataclass
class AzureVoiceClientConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = DEFAULT_REALTIME_API_VERSION
    voice: str = "alloy"
    sample_rate_hz: int = 24000
    realtime_host: Optional[str] = None


class AzureVoiceClient:
    """Thin wrapper around the Azure OpenAI Voice Live API."""

    def __init__(self, config: AzureVoiceClientConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)

    async def generate_reply(
        self,
        user_text: Optional[str] = None,
        input_audio_chunks: Optional[Sequence[bytes]] = None,
        *,
        instructions: str = ZARA_INSTRUCTIONS,
        voice: Optional[str] = None,
    ) -> Tuple[str, bytes]:
        if not user_text and not input_audio_chunks:
            raise ValueError("Either user_text or input_audio_chunks must be provided.")

        voice_name = voice or self.config.voice
        client = AsyncAzureOpenAI(
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version,
        )

        try:
            async with client.beta.realtime.connect(model=self.config.deployment) as connection:
                await self._configure_session(connection, instructions=instructions, voice=voice_name)

                if user_text:
                    await self._send_user_text(connection, user_text)

                if input_audio_chunks:
                    await self._send_user_audio(connection, input_audio_chunks)

                await connection.response.create(
                    response={
                        "modalities": ["audio", "text"],
                        "output_audio_format": "pcm16",
                        "voice": voice_name,
                    }
                )

                collected_audio = bytearray()
                collected_text: list[str] = []

                async for event in connection:
                    event_type = getattr(event, "type", None)
                    if event_type in {"response.text.delta", "response.output_text.delta"}:
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            collected_text.append(delta)
                    elif event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = getattr(event, "delta", None)
                        if delta:
                            collected_audio.extend(base64.b64decode(delta))
                    elif event_type in {
                        "response.audio_transcript.delta",
                        "response.output_audio_transcript.delta",
                    }:
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            collected_text.append(delta)
                    elif event_type in {
                        "response.text.done",
                        "response.output_text.done",
                        "response.audio.done",
                    }:
                        continue
                    elif event_type == "response.done":
                        break
                    elif event_type in {"response.error", "error"}:
                        error_payload = getattr(event, "error", None)
                        if isinstance(error_payload, dict):
                            message = error_payload.get("message", "Unknown error")
                        else:
                            message = str(error_payload or "Unknown error")
                        self._logger.error("Realtime response error: %s", message)
                        raise AzureVoiceClientError(message)
                    else:
                        self._logger.debug("Unhandled realtime event: %s", event_type)

                audio_bytes = bytes(collected_audio)
                if not audio_bytes:
                    message = "Realtime API returned no audio in the response."
                    self._logger.error(message)
                    raise AzureVoiceClientError(message)
                transcript = "".join(collected_text).strip()
                if transcript:
                    self._logger.debug("Collected transcript (dropping from response): %s", transcript)
                return "", audio_bytes
        finally:
            await client.close()

    async def _configure_session(self, connection, *, instructions: str, voice: str) -> None:
        await connection.session.update(
            session={
                "instructions": instructions,
                "voice": voice,
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
            }
        )

    async def _send_user_text(self, connection, text: str) -> None:
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            }
        )

    async def _send_user_audio(self, connection, audio_chunks: Sequence[bytes]) -> None:
        combined = bytearray()
        for chunk in audio_chunks:
            if not chunk:
                continue
            combined.extend(chunk)
        if not combined:
            return
        audio_b64 = base64.b64encode(bytes(combined)).decode("ascii")
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio": audio_b64,
                    }
                ],
            }
        )


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a numpy float32/float64 audio array to 16-bit PCM bytes."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    clipped = np.clip(audio, -1.0, 1.0)
    int_samples = (clipped * 32767).astype(np.int16)
    return int_samples.tobytes()


def save_wav(path: Path, audio_bytes: bytes, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    sf.write(path, audio_array, samplerate=sample_rate)


def pcm16_to_wav_bytes(audio_bytes: bytes, sample_rate: int) -> bytes:
    if not audio_bytes:
        return b""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    return buffer.getvalue()


def default_output_path(prefix: str = "zara_response", ext: str = ".wav") -> Path:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return Path("out") / f"{prefix}_{timestamp}{ext}"


async def generate_and_save(
    client: AzureVoiceClient,
    *,
    text_prompt: Optional[str],
    audio_chunks: Optional[Sequence[bytes]],
    voice: Optional[str] = None,
) -> Tuple[str, Path]:
    transcript, audio_bytes = await client.generate_reply(
        user_text=text_prompt,
        input_audio_chunks=audio_chunks,
        voice=voice,
    )
    output_path = default_output_path()
    save_wav(output_path, audio_bytes, client.config.sample_rate_hz)
    return transcript, output_path


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return endpoint
    if endpoint.startswith("http://"):
        endpoint = "https://" + endpoint[len("http://") :]
    elif not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}"
    return endpoint.rstrip("/")


def load_env_config(
    *,
    sample_rate_override: Optional[int] = None,
    voice_override: Optional[str] = None,
    api_version_override: Optional[str] = None,
) -> AzureVoiceClientConfig:
    endpoint = _get_env("AZURE_OPENAI_ENDPOINT")
    api_key = _get_env("AZURE_OPENAI_API_KEY")
    deployment = _get_env("AZURE_OPENAI_DEPLOYMENT")
    realtime_host = _get_env("AZURE_OPENAI_REALTIME_HOST")
    if not all([endpoint, api_key, deployment]):
        missing = [
            name
            for name, value in [
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ]
            if not value
        ]
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    endpoint = _normalize_endpoint(endpoint)

    def _int_env(var_name: str, fallback: int) -> int:
        raw = _get_env(var_name)
        if raw is None:
            return fallback
        try:
            return int(raw)
        except ValueError:
            raise RuntimeError(f"{var_name} must be an integer, got {raw!r}") from None

    sample_rate = sample_rate_override or _int_env("AZURE_OPENAI_SAMPLE_RATE", 24000)
    voice = voice_override or _get_env("AZURE_OPENAI_VOICE") or "alloy"
    api_version = api_version_override or _get_env(
        "AZURE_OPENAI_API_VERSION",
    ) or DEFAULT_REALTIME_API_VERSION

    return AzureVoiceClientConfig(
        endpoint=endpoint,
        api_key=api_key,
        deployment=deployment,
        sample_rate_hz=sample_rate,
        voice=voice,
        api_version=api_version,
        realtime_host=realtime_host,
    )
