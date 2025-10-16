import asyncio
import base64
import datetime as dt
import functools
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import logging
import numpy as np
import wave
from xml.sax.saxutils import escape as xml_escape

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


def _extract_text_from_response_payload(payload: dict) -> list[str]:
    segments: list[str] = []
    if not isinstance(payload, dict):
        return segments
    outputs = payload.get("output")
    if isinstance(outputs, list):
        for output in outputs:
            if not isinstance(output, dict):
                continue
            content_items = output.get("content")
            if not isinstance(content_items, list):
                continue
            for item in content_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                text_value = item.get("text")
                if item_type in {"output_text", "text"} and isinstance(text_value, str):
                    segments.append(text_value)
    return segments


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


class PersonalVoiceError(RuntimeError):
    """Raised when Azure Personal Voice synthesis fails."""


@dataclass
class PersonalVoiceConfig:
    speech_key: str
    speech_region: str
    speaker_profile_id: str
    voice_name: str = "DragonLatestNeural"
    style: Optional[str] = "Prompt"
    language: str = "en-US"

    def build_ssml(self, text: str, base_voice_override: Optional[str] = None) -> str:
        """Construct the SSML payload for a personal voice synthesis request."""
        if not text.strip():
            raise PersonalVoiceError("Cannot synthesize empty text with personal voice.")
        voice_name = (base_voice_override or self.voice_name or "").strip()
        if not voice_name:
            raise PersonalVoiceError("A base voice name (e.g. DragonLatestNeural) is required for personal voice.")

        escaped_text = xml_escape(text)
        ssml_parts = [
            "<speak version='1.0' xml:lang='{lang}' xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts'>".format(lang=self.language),
            "<voice name='{name}'>".format(name=voice_name),
            "<mstts:ttsembedding speakerProfileId='{profile}'/>".format(profile=self.speaker_profile_id),
        ]
        if self.style:
            ssml_parts.append("<mstts:express-as style='{style}'>".format(style=self.style))
        ssml_parts.append("<lang xml:lang='{lang}'>{text}</lang>".format(lang=self.language, text=escaped_text))
        if self.style:
            ssml_parts.append("</mstts:express-as>")
        ssml_parts.append("</voice></speak>")
        return "".join(ssml_parts)


class PersonalVoiceSynthesizer:
    """Helper that turns text into PCM audio using Azure Personal Voice."""

    def __init__(self, config: PersonalVoiceConfig, sample_rate_hz: int, *, logger: logging.Logger):
        self._config = config
        self._sample_rate = sample_rate_hz
        self._logger = logger.getChild("PersonalVoice")
        self._speechsdk = None

    def _ensure_speechsdk(self):
        if self._speechsdk is not None:
            return self._speechsdk
        try:
            import azure.cognitiveservices.speech as speechsdk  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency error
            raise PersonalVoiceError(
                "azure-cognitiveservices-speech is required to synthesize personal voice audio."
            ) from exc
        self._speechsdk = speechsdk
        return speechsdk

    def _resolve_output_format(self, speechsdk):
        mapping = {
            16000: speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
            24000: speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
            48000: speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
        }
        if self._sample_rate not in mapping:
            raise PersonalVoiceError(
                f"Sample rate {self._sample_rate} Hz is not supported for raw PCM synthesis by Azure Speech."
            )
        return mapping[self._sample_rate]

    def _synthesize_blocking(self, text: str, base_voice_override: Optional[str]) -> bytes:
        speechsdk = self._ensure_speechsdk()
        speech_config = speechsdk.SpeechConfig(
            subscription=self._config.speech_key,
            region=self._config.speech_region,
        )
        speech_config.set_speech_synthesis_output_format(self._resolve_output_format(speechsdk))

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pcm")
        os.close(tmp_fd)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        ssml = self._config.build_ssml(text, base_voice_override=base_voice_override)
        try:
            result = synthesizer.speak_ssml_async(ssml).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                try:
                    audio_data = Path(tmp_path).read_bytes()
                except OSError as exc:
                    raise PersonalVoiceError(f"Failed to read synthesized audio: {exc}") from exc
                if not audio_data:
                    audio_data = result.audio_data or b""
                if not audio_data:
                    raise PersonalVoiceError("Personal voice synthesis returned no audio bytes.")
                return bytes(audio_data)
            if result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                message = details.error_details or str(details.reason)
                raise PersonalVoiceError(f"Personal voice synthesis canceled: {message}")
            raise PersonalVoiceError(f"Unexpected personal voice synthesis result: {result.reason}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    async def synthesize(self, text: str, *, base_voice_override: Optional[str]) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self._synthesize_blocking, text, base_voice_override),
        )


@dataclass
class AzureVoiceClientConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = DEFAULT_REALTIME_API_VERSION
    voice: str = "alloy"
    sample_rate_hz: int = 24000
    realtime_host: Optional[str] = None
    personal_voice: Optional[PersonalVoiceConfig] = None


class AzureVoiceClient:
    """Thin wrapper around the Azure OpenAI Voice Live API."""

    def __init__(self, config: AzureVoiceClientConfig):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._personal_voice_synthesizer: Optional[PersonalVoiceSynthesizer] = None

    def _get_personal_voice_synthesizer(self) -> Optional[PersonalVoiceSynthesizer]:
        if not self.config.personal_voice:
            return None
        if self._personal_voice_synthesizer is None:
            self._personal_voice_synthesizer = PersonalVoiceSynthesizer(
                self.config.personal_voice,
                self.config.sample_rate_hz,
                logger=self._logger,
            )
        return self._personal_voice_synthesizer

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

        personal_voice_synth = self._get_personal_voice_synthesizer()
        voice_override = (voice or "").strip() or None

        voice_name = voice_override or self.config.voice
        realtime_output_modalities = ["audio", "text"] if personal_voice_synth is None else ["text"]

        client = AsyncAzureOpenAI(
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version,
        )

        try:
            async with client.beta.realtime.connect(model=self.config.deployment) as connection:
                await self._configure_session(
                    connection,
                    instructions=instructions,
                    output_modalities=realtime_output_modalities,
                    voice_name=None if personal_voice_synth else voice_name,
                )

                if user_text:
                    await self._send_user_text(connection, user_text)

                if input_audio_chunks:
                    await self._send_user_audio(connection, input_audio_chunks)

                await connection.response.create(
                    response={
                        "modalities": list(dict.fromkeys(realtime_output_modalities)),
                        **({"voice": voice_name} if not personal_voice_synth and voice_name else {}),
                    }
                )

                collected_audio = bytearray()
                collected_text: list[str] = []
                final_text_segments: list[str] = []

                async for event in connection:
                    event_type = getattr(event, "type", None)
                    if event_type in {
                        "response.text.delta",
                        "response.output_text.delta",
                        "response.output_audio_transcript.delta",
                    }:
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            collected_text.append(delta)
                    elif event_type in {
                        "response.audio.delta",
                        "response.output_audio.delta",
                    }:
                        if personal_voice_synth is not None:
                            continue
                        delta = getattr(event, "delta", None)
                        if delta:
                            collected_audio.extend(base64.b64decode(delta))
                    elif event_type in {
                        "response.text.done",
                        "response.output_text.done",
                        "response.audio.done",
                    }:
                        continue
                    elif event_type == "response.done":
                        response_payload = getattr(event, "response", None)
                        final_text_segments.extend(_extract_text_from_response_payload(response_payload))
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

                transcript = "".join(collected_text).strip()
                if not transcript and final_text_segments:
                    transcript = "".join(final_text_segments).strip()
                if transcript:
                    self._logger.debug("Collected transcript from realtime response: %s", transcript)
                if personal_voice_synth is not None:
                    if not transcript:
                        raise AzureVoiceClientError(
                            "Realtime API returned no text to synthesize with the personal voice."
                        )
                    base_voice = voice_override or self.config.personal_voice.voice_name
                    try:
                        audio_bytes = await personal_voice_synth.synthesize(
                            transcript,
                            base_voice_override=base_voice,
                        )
                    except PersonalVoiceError as exc:
                        self._logger.error("Personal voice synthesis failed: %s", exc)
                        raise AzureVoiceClientError(str(exc)) from exc
                    return transcript, audio_bytes

                audio_bytes = bytes(collected_audio)
                if not audio_bytes:
                    message = "Realtime API returned no audio in the response."
                    self._logger.error(message)
                    raise AzureVoiceClientError(message)
                return transcript, audio_bytes
        finally:
            await client.close()

    async def _configure_session(
        self,
        connection,
        *,
        instructions: str,
        output_modalities: Sequence[str],
        voice_name: Optional[str],
    ) -> None:
        session_payload: dict[str, object] = {
            "instructions": instructions,
        }
        modalities = list(dict.fromkeys(output_modalities))
        if modalities:
            session_payload["modalities"] = modalities
        if voice_name:
            session_payload["voice"] = voice_name
        await connection.session.update(session=session_payload)

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

    speech_key = _get_env("AZURE_SPEECH_KEY")
    speech_region = _get_env("AZURE_SPEECH_REGION")
    speaker_profile_id = (
        _get_env("AZURE_SPEECH_SPEAKER_PROFILE_ID")
        or _get_env("AZURE_PERSONAL_VOICE_SPEAKER_PROFILE_ID")
        or _get_env("AZURE_PERSONAL_VOICE_ID")
    )
    personal_voice: Optional[PersonalVoiceConfig] = None
    if any([speech_key, speech_region, speaker_profile_id]):
        missing = [
            name
            for name, value in [
                ("AZURE_SPEECH_KEY", speech_key),
                ("AZURE_SPEECH_REGION", speech_region),
                ("AZURE_SPEECH_SPEAKER_PROFILE_ID", speaker_profile_id),
            ]
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Personal voice configuration is incomplete. Set the following environment variable(s): "
                + ", ".join(missing)
            )
        personal_voice = PersonalVoiceConfig(
            speech_key=speech_key,
            speech_region=speech_region,
            speaker_profile_id=speaker_profile_id,
            voice_name=_get_env("AZURE_SPEECH_VOICE_NAME") or "DragonLatestNeural",
            style=_get_env("AZURE_SPEECH_VOICE_STYLE") or "Prompt",
            language=_get_env("AZURE_SPEECH_LANGUAGE") or "en-US",
        )

    return AzureVoiceClientConfig(
        endpoint=endpoint,
        api_key=api_key,
        deployment=deployment,
        sample_rate_hz=sample_rate,
        voice=voice,
        api_version=api_version,
        realtime_host=realtime_host,
        personal_voice=personal_voice,
    )
