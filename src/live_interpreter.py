import asyncio
import functools
import io
import logging
import os
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple


class LiveInterpreterError(RuntimeError):
    """Raised when the Azure Live Interpreter translation flow fails."""


@dataclass(frozen=True)
class LiveInterpreterConfig:
    """Configuration for Azure Speech translation (Live Interpreter) sessions."""

    subscription_key: str
    endpoint: Optional[str] = None
    region: Optional[str] = None
    voice_name: Optional[str] = None
    target_languages: Tuple[str, ...] = field(default_factory=lambda: ("en",))
    source_languages: Optional[Tuple[str, ...]] = None
    auto_detect_source_language: bool = True
    default_source_language: Optional[str] = None

    def with_overrides(
        self,
        *,
        target_languages: Optional[Sequence[str]] = None,
        voice_name: Optional[str] = None,
        source_language: Optional[str] = None,
        auto_detect: Optional[bool] = None,
    ) -> "LiveInterpreterConfig":
        """Return a copy of the config applying per-request overrides."""
        targets = _normalize_languages(target_languages) or self.target_languages
        if not targets:
            raise LiveInterpreterError("At least one target language must be specified for translation.")

        sources: Optional[Tuple[str, ...]]
        if source_language:
            sources = _normalize_languages([source_language])
        else:
            sources = self.source_languages

        return LiveInterpreterConfig(
            subscription_key=self.subscription_key,
            endpoint=self.endpoint,
            region=self.region,
            voice_name=voice_name or self.voice_name,
            target_languages=targets,
            source_languages=sources,
            auto_detect_source_language=self.auto_detect_source_language if auto_detect is None else auto_detect,
            default_source_language=self.default_source_language,
        )


@dataclass
class TranslationResult:
    """Result returned from a translation request."""

    recognized_text: str
    translations: Dict[str, str]
    audio_data: Optional[bytes] = None
    audio_format: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    detected_source_language: Optional[str] = None


def _normalize_languages(languages: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Return a tuple of cleaned language codes."""
    if not languages:
        return tuple()
    normalized = []
    for lang in languages:
        if not lang:
            continue
        # Handle both regular comma (,) and Chinese comma (，)
        # Split in case multiple languages are incorrectly joined
        parts = lang.replace('，', ',').split(',')
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                normalized.append(cleaned)
    return tuple(dict.fromkeys(normalized))


def _parse_bool(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _load_env_var(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is not None:
        return value
    # Fallback to .env via voice_client loader when available.
    try:
        from src.voice_client import _get_env as _voice_get_env  # type: ignore[attr-defined]
    except Exception:
        return None
    return _voice_get_env(name)


def load_live_interpreter_config() -> LiveInterpreterConfig:
    """
    Load Live Interpreter configuration from environment variables.

    Required:
        - AZURE_SPEECH_TRANSLATION_KEY or AZURE_SPEECH_KEY
        - (AZURE_SPEECH_TRANSLATION_ENDPOINT or AZURE_SPEECH_REGION)

    Optional:
        - AZURE_SPEECH_TRANSLATION_ENDPOINT
        - AZURE_SPEECH_TRANSLATION_REGION
        - AZURE_SPEECH_TRANSLATION_VOICE
        - AZURE_SPEECH_TRANSLATION_TARGET_LANGUAGES (comma-separated)
        - AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGES (comma-separated)
        - AZURE_SPEECH_TRANSLATION_AUTO_DETECT (true/false)
    """
    subscription_key = (
        _load_env_var("AZURE_SPEECH_TRANSLATION_KEY") or _load_env_var("AZURE_SPEECH_KEY")
    )
    if not subscription_key:
        raise LiveInterpreterError(
            "Set AZURE_SPEECH_TRANSLATION_KEY (or AZURE_SPEECH_KEY) to enable Live Interpreter translation."
        )

    endpoint = _load_env_var("AZURE_SPEECH_TRANSLATION_ENDPOINT")
    region = (
        _load_env_var("AZURE_SPEECH_TRANSLATION_REGION") or _load_env_var("AZURE_SPEECH_REGION")
    )
    if not endpoint and not region:
        raise LiveInterpreterError(
            "Live Interpreter requires AZURE_SPEECH_TRANSLATION_ENDPOINT or AZURE_SPEECH_TRANSLATION_REGION."
        )

    voice_name = _load_env_var("AZURE_SPEECH_TRANSLATION_VOICE")
    targets = _normalize_languages(
        _load_env_var("AZURE_SPEECH_TRANSLATION_TARGET_LANGUAGES").split(",")
        if _load_env_var("AZURE_SPEECH_TRANSLATION_TARGET_LANGUAGES")
        else None
    )
    if not targets:
        targets = ("en",)

    source_languages_raw = _load_env_var("AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGES")
    source_languages = _normalize_languages(source_languages_raw.split(",")) if source_languages_raw else None
    default_source_language = (
        _load_env_var("AZURE_SPEECH_TRANSLATION_DEFAULT_SOURCE_LANGUAGE")
        or _load_env_var("AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGE")
    )
    if default_source_language:
        default_source_language = default_source_language.strip()
        if not default_source_language:
            default_source_language = None

    auto_detect = _parse_bool(
        _load_env_var("AZURE_SPEECH_TRANSLATION_AUTO_DETECT"),
        default=True,
    )

    return LiveInterpreterConfig(
        subscription_key=subscription_key,
        endpoint=endpoint,
        region=region,
        voice_name=voice_name,
        target_languages=targets,
        source_languages=source_languages if source_languages else None,
        auto_detect_source_language=auto_detect,
        default_source_language=default_source_language,
    )


class LiveInterpreterTranslator:
    """Blocking Azure Speech translator wrapped with asyncio helpers."""

    def __init__(self, config: LiveInterpreterConfig):
        self._logger = logging.getLogger(__name__).getChild("LiveInterpreter")
        self._config = config
        self._speechsdk = None

    def _ensure_speechsdk(self):
        if self._speechsdk is not None:
            return self._speechsdk
        try:
            import azure.cognitiveservices.speech as speechsdk  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LiveInterpreterError(
                "azure-cognitiveservices-speech is required for Live Interpreter translation."
            ) from exc
        self._speechsdk = speechsdk
        return speechsdk

    async def translate_audio(
        self,
        audio_bytes: bytes,
        sample_rate_hz: int,
        *,
        target_languages: Optional[Sequence[str]] = None,
        voice_name: Optional[str] = None,
        source_language: Optional[str] = None,
        auto_detect: Optional[bool] = None,
    ) -> TranslationResult:
        """Translate audio bytes using the Azure Speech translation service."""
        loop = asyncio.get_running_loop()
        config = self._config.with_overrides(
            target_languages=target_languages,
            voice_name=voice_name,
            source_language=source_language,
            auto_detect=auto_detect,
        )
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._translate_blocking,
                audio_bytes,
                sample_rate_hz,
                config=config,
            ),
        )

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------
    def _translate_blocking(
        self,
        audio_bytes: bytes,
        sample_rate_hz: int,
        *,
        config: LiveInterpreterConfig,
    ) -> TranslationResult:
        self._logger.info(
            "Starting translation - endpoint=%s, region=%s, target_langs=%s, source_lang=%s, voice=%s, auto_detect=%s",
            config.endpoint,
            config.region,
            config.target_languages,
            config.source_languages or config.default_source_language,
            config.voice_name,
            config.auto_detect_source_language,
        )
        speechsdk = self._ensure_speechsdk()
        translation_config = self._create_translation_config(speechsdk, config)

        wav_path = self._write_temp_wav(audio_bytes, sample_rate_hz)
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)

        recognizer_kwargs = {
            "translation_config": translation_config,
            "audio_config": audio_config,
        }

        auto_detect_config = None
        recognition_language: Optional[str] = None

        if config.source_languages:
            unique_sources = tuple(dict.fromkeys(config.source_languages))
            self._logger.debug("Source languages provided: %s", unique_sources)
            if config.auto_detect_source_language and len(unique_sources) > 1:
                self._logger.info("Using auto-detect with multiple source languages: %s", unique_sources)
                auto_detect_config = self._create_auto_detect_config(speechsdk, unique_sources)
            else:
                recognition_language = unique_sources[0]
                self._logger.info("Using fixed source language: %s", recognition_language)
        else:
            if config.auto_detect_source_language:
                # When auto-detect is enabled and no specific source languages are provided,
                # use open range detection to allow Azure to detect any supported language
                self._logger.info("Using auto-detect with open range (no specific source languages)")
                auto_detect_config = self._create_auto_detect_config(speechsdk, [])
            else:
                recognition_language = config.default_source_language or "en-US"
                self._logger.info("Auto-detect disabled, using language: %s", recognition_language)

        # CRITICAL: Do NOT set speech_recognition_language when using auto_detect_config
        # This causes SPXERR_INVALID_ARG error
        if auto_detect_config is not None:
            self._logger.info("Adding auto_detect_source_language_config to recognizer")
            recognizer_kwargs["auto_detect_source_language_config"] = auto_detect_config
        else:
            self._logger.info("Setting speech_recognition_language=%s", recognition_language or "en-US")
            translation_config.speech_recognition_language = recognition_language or "en-US"

        recognizer = speechsdk.translation.TranslationRecognizer(**recognizer_kwargs)

        recognized_text: Optional[str] = None
        translations: Dict[str, str] = {}
        synthesized_audio = bytearray()
        detected_language: Optional[str] = None
        cancellation_details: Optional[str] = None

        def _handle_recognized(evt):
            nonlocal recognized_text, translations, detected_language
            result = getattr(evt, "result", None)
            if not result:
                return
            if getattr(result, "text", None):
                recognized_text = result.text
            if getattr(result, "translations", None):
                translations = dict(result.translations)
            if getattr(result, "language", None):
                detected_language = result.language

        def _handle_synthesizing(evt):
            data = getattr(evt, "result", None)
            if not data:
                return
            audio = getattr(data, "audio", None)
            if audio:
                synthesized_audio.extend(audio)

        def _handle_canceled(evt):
            nonlocal cancellation_details
            result = getattr(evt, "result", None)
            if result and hasattr(result, "error_details") and result.error_details:
                cancellation_details = result.error_details

        recognizer.recognized.connect(_handle_recognized)
        recognizer.synthesizing.connect(_handle_synthesizing)
        recognizer.canceled.connect(_handle_canceled)

        self._logger.debug("Starting recognition with kwargs: %s", list(recognizer_kwargs.keys()))
        try:
            result = recognizer.recognize_once_async().get()
            self._logger.debug("Recognition completed with reason: %s", result.reason)
        except Exception as recognition_error:
            self._logger.error("Recognition failed: %s", recognition_error, exc_info=True)
            raise
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        if result.reason == speechsdk.ResultReason.TranslatedSpeech:
            recognized_text = result.text
            translations = dict(result.translations)
            if hasattr(result, "language"):
                detected_language = result.language
        elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            # Provide more context for NoMatch errors
            no_match_details = getattr(result, "no_match_details", None)
            if no_match_details:
                reason = getattr(no_match_details, "reason", None)
                self._logger.warning("Speech recognition NoMatch reason: %s", reason)
            self._logger.warning(
                "Speech not recognized - audio_length=%d bytes, sample_rate=%d Hz, config=%s",
                len(audio_bytes),
                sample_rate_hz,
                {
                    "source_lang": config.source_languages or config.default_source_language,
                    "target_langs": config.target_languages,
                    "auto_detect": config.auto_detect_source_language,
                },
            )
            raise LiveInterpreterError(
                "Speech could not be recognized. This may be due to low audio quality, "
                "background noise, or unsupported language. Please ensure clear audio input."
            )
        elif result.reason == speechsdk.ResultReason.Canceled:
            error_details = cancellation_details or getattr(result.cancellation_details, "error_details", None)
            cancellation_reason = getattr(result.cancellation_details, "reason", None) if hasattr(result, "cancellation_details") else None
            self._logger.error(
                "Translation canceled - reason=%s, error_details=%s",
                cancellation_reason,
                error_details,
            )
            message = error_details or "Translation request was canceled."
            raise LiveInterpreterError(message)

        audio_bytes = bytes(synthesized_audio) if synthesized_audio else None
        audio_format = None
        audio_sample_rate = None
        if audio_bytes and len(audio_bytes) > 44:
            with io.BytesIO(audio_bytes) as buffer:
                try:
                    with wave.open(buffer, "rb") as wav_file:
                        audio_format = "wav"
                        audio_sample_rate = wav_file.getframerate()
                except (wave.Error, EOFError):
                    audio_format = None
                    audio_sample_rate = None

        return TranslationResult(
            recognized_text=recognized_text or "",
            translations=translations,
            audio_data=audio_bytes,
            audio_format=audio_format,
            audio_sample_rate=audio_sample_rate,
            detected_source_language=detected_language,
        )

    def _create_translation_config(self, speechsdk, config: LiveInterpreterConfig):
        if config.endpoint:
            # Use from_endpoint method like the Java sample
            try:
                translation_config = speechsdk.translation.SpeechTranslationConfig.from_endpoint(
                    endpoint=config.endpoint,
                    subscription_key=config.subscription_key
                )
            except (AttributeError, TypeError):
                # Fallback for older SDK versions
                translation_config = speechsdk.translation.SpeechTranslationConfig(
                    subscription=config.subscription_key,
                    region=config.region or "swedencentral",
                )
                translation_config.set_property(
                    speechsdk.PropertyId.SpeechServiceConnection_Endpoint,
                    config.endpoint,
                )
        else:
            translation_config = speechsdk.translation.SpeechTranslationConfig(
                subscription=config.subscription_key,
                region=config.region,
            )

        for language in config.target_languages:
            translation_config.add_target_language(language)
        if config.voice_name:
            translation_config.voice_name = config.voice_name
        return translation_config

    def _create_auto_detect_config(self, speechsdk, languages: Sequence[str]):
        cleaned = _normalize_languages(languages)
        if not cleaned:
            raise LiveInterpreterError(
                "Auto language detection requires at least one source language. "
                "Set AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGES to a comma-separated list."
            )
        # Use from_open_range() like the Java sample when no specific languages provided
        # Or from_languages() when specific languages are provided
        try:
            if len(cleaned) == 0:
                return speechsdk.languageconfig.AutoDetectSourceLanguageConfig.from_open_range()
            else:
                return speechsdk.languageconfig.AutoDetectSourceLanguageConfig.from_languages(list(cleaned))
        except AttributeError:
            # Fallback for older SDK versions
            return speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=list(cleaned))

    def _write_temp_wav(self, audio_bytes: bytes, sample_rate_hz: int) -> str:
        if not audio_bytes:
            raise LiveInterpreterError("Audio data is required for translation.")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate_hz)
                wav_file.writeframes(audio_bytes)
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        return path
