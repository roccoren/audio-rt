import asyncio
import functools
import io
import logging
import os
import threading
import wave
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple


class LiveInterpreterError(RuntimeError):
    """Raised when the Azure Live Interpreter translation flow fails."""


@dataclass(frozen=True)
class LiveInterpreterPersonalVoiceConfig:
    """Optional personal voice settings for Live Interpreter synthesis."""

    speaker_profile_id: str
    voice_name: str = "personal-voice"
    style: Optional[str] = None
    language: Optional[str] = None


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
    personal_voice: Optional[LiveInterpreterPersonalVoiceConfig] = None

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
            personal_voice=self.personal_voice,
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

    personal_voice: Optional[LiveInterpreterPersonalVoiceConfig] = None
    speaker_profile_id = (
        _load_env_var("AZURE_SPEECH_SPEAKER_PROFILE_ID")
        or _load_env_var("AZURE_PERSONAL_VOICE_SPEAKER_PROFILE_ID")
        or _load_env_var("AZURE_PERSONAL_VOICE_ID")
    )
    if speaker_profile_id:
        cleaned_profile_id = speaker_profile_id.strip()
        if not cleaned_profile_id:
            raise LiveInterpreterError("Personal voice speaker profile id is empty.")
        personal_voice_voice_name = (
            _load_env_var("AZURE_SPEECH_TRANSLATION_PERSONAL_VOICE_NAME")
            or voice_name
            or "personal-voice"
        )
        personal_voice = LiveInterpreterPersonalVoiceConfig(
            speaker_profile_id=cleaned_profile_id,
            voice_name=personal_voice_voice_name,
            style=_load_env_var("AZURE_SPEECH_VOICE_STYLE"),
            language=_load_env_var("AZURE_SPEECH_LANGUAGE"),
        )
        if not voice_name:
            voice_name = personal_voice.voice_name

    if not endpoint and region:
        endpoint = f"wss://{region}.stt.speech.microsoft.com/speech/universal/v2?setfeature=zeroshotttsflight"

    return LiveInterpreterConfig(
        subscription_key=subscription_key,
        endpoint=endpoint,
        region=region,
        voice_name=voice_name,
        target_languages=targets,
        source_languages=source_languages if source_languages else None,
        auto_detect_source_language=auto_detect,
        default_source_language=default_source_language,
        personal_voice=personal_voice,
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
        if not audio_bytes:
            raise LiveInterpreterError("Audio data is required for translation.")
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

        recognizer_kwargs: Dict[str, object] = {"translation_config": translation_config}

        auto_detect_config = None
        recognition_language: Optional[str] = None

        if config.auto_detect_source_language:
            source_candidates = config.source_languages or tuple()
            try:
                auto_detect_config = self._create_auto_detect_config(speechsdk, source_candidates)
            except LiveInterpreterError as exc:
                if source_candidates:
                    self._logger.warning(
                        "Falling back to fixed recognition language after auto-detect error: %s",
                        exc,
                    )
                    recognition_language = source_candidates[0]
                else:
                    raise

        if not config.auto_detect_source_language or auto_detect_config is None:
            recognition_language = (
                recognition_language
                or (config.source_languages[0] if config.source_languages else None)
                or config.default_source_language
                or "en-US"
            )
            translation_config.speech_recognition_language = recognition_language
            self._logger.info("Using fixed recognition language: %s", recognition_language)
        else:
            recognizer_kwargs["auto_detect_source_language_config"] = auto_detect_config
            self._logger.info("Auto language detection enabled (open range).")

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=sample_rate_hz,
            bits_per_sample=16,
            channels=1,
        )
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer_kwargs["audio_config"] = audio_config

        recognizer = speechsdk.translation.TranslationRecognizer(**recognizer_kwargs)

        recognized_text: Optional[str] = None
        translations: Dict[str, str] = {}
        synthesized_audio = bytearray()
        detected_language: Optional[str] = None
        error_message: Optional[str] = None

        result_ready = threading.Event()
        session_stopped = threading.Event()

        def _mark_result_ready() -> None:
            if not result_ready.is_set():
                result_ready.set()

        def _handle_recognized(evt):
            nonlocal recognized_text, translations, detected_language, error_message
            result = getattr(evt, "result", None)
            if not result:
                return

            detected_language = result.properties.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult,
                getattr(result, "language", None),
            )

            if result.reason == speechsdk.ResultReason.TranslatedSpeech:
                recognized_text = result.text or recognized_text
                translations = dict(result.translations)
                _mark_result_ready()
            elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
                recognized_text = result.text or recognized_text
                translations = dict(result.translations)
                _mark_result_ready()
            elif result.reason == speechsdk.ResultReason.NoMatch:
                no_match_details = getattr(result, "no_match_details", None)
                detail_reason = getattr(no_match_details, "reason", None) if no_match_details else None
                self._logger.warning("Speech recognition NoMatch reason: %s", detail_reason)
                error_message = (
                    "Speech could not be recognized. This may be due to low audio quality, "
                    "background noise, or unsupported language. Please ensure clear audio input."
                )
                _mark_result_ready()

        def _handle_synthesizing(evt):
            data = getattr(evt, "result", None)
            if not data:
                return
            audio = getattr(data, "audio", None)
            if audio:
                synthesized_audio.extend(audio)

        def _handle_canceled(evt):
            nonlocal error_message
            result = getattr(evt, "result", None)
            if result and getattr(result, "error_details", None):
                error_message = result.error_details
            reason = getattr(evt, "reason", None)
            cancellation_details = getattr(evt, "cancellation_details", None)
            details_message: Optional[str] = None
            if cancellation_details is not None:
                error_details = getattr(cancellation_details, "error_details", None)
                if error_details:
                    details_message = error_details
                else:
                    details_message = str(getattr(cancellation_details, "reason", "") or "")
            if not error_message and result is not None:
                try:
                    cancellation = speechsdk.CancellationDetails.from_result(result)
                except Exception:
                    cancellation = None
                if cancellation is not None and getattr(cancellation, "error_details", None):
                    details_message = cancellation.error_details
            if details_message:
                error_message = details_message
            if not error_message:
                error_message = f"Translation canceled ({reason})" if reason else "Translation canceled unexpectedly."
            self._logger.error("Translation canceled: %s", reason or "Unknown")
            _mark_result_ready()
            session_stopped.set()

        def _handle_session_stopped(_: object) -> None:
            session_stopped.set()

        recognizer.recognized.connect(_handle_recognized)
        recognizer.synthesizing.connect(_handle_synthesizing)
        recognizer.canceled.connect(_handle_canceled)
        recognizer.session_stopped.connect(_handle_session_stopped)

        def _push_audio() -> None:
            try:
                push_stream.write(audio_bytes)
            finally:
                push_stream.close()

        feeder = threading.Thread(target=_push_audio, name="LiveInterpreterFeeder", daemon=True)

        self._logger.debug("Starting continuous recognition with kwargs: %s", list(recognizer_kwargs.keys()))
        try:
            recognizer.start_continuous_recognition()
            feeder.start()
            if not result_ready.wait(timeout=60):
                error_message = "Timed out waiting for Azure Speech translation."
            recognizer.stop_continuous_recognition()
            session_stopped.wait(timeout=5)
        except Exception as recognition_error:
            self._logger.error("Recognition failed: %s", recognition_error, exc_info=True)
            raise
        finally:
            try:
                feeder.join(timeout=1)
            except RuntimeError:
                pass

        if error_message:
            raise LiveInterpreterError(error_message)

        audio_data = bytes(synthesized_audio) if synthesized_audio else None
        audio_format = None
        audio_sample_rate = None
        if audio_data and len(audio_data) > 44:
            with io.BytesIO(audio_data) as buffer:
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
            audio_data=audio_data,
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

        def _set_property_value(name: str, value: Optional[str], *, aliases: Sequence[str] = ()) -> None:
            if not value:
                return
            candidates = (name, *aliases)
            last_error: Optional[Exception] = None
            for candidate in candidates:
                property_id = getattr(getattr(speechsdk, "PropertyId", None), candidate, None)
                if property_id is not None:
                    translation_config.set_property(property_id, value)
                    return
                set_property_by_name = getattr(translation_config, "set_property_by_name", None)
                if callable(set_property_by_name):
                    try:
                        set_property_by_name(candidate, value)
                        return
                    except Exception as exc:  # pragma: no cover - defensive fallback
                        last_error = exc
                        continue
                try:
                    translation_config.set_property(candidate, value)
                    return
                except Exception as exc:  # pragma: no cover - defensive fallback
                    last_error = exc
            if last_error is not None:
                self._logger.warning("Azure SDK missing property support: %s=%s (%s)", name, value, last_error)

        if config.personal_voice:
            voice_to_use = config.voice_name or config.personal_voice.voice_name
            if voice_to_use:
                translation_config.voice_name = voice_to_use
            _set_property_value("SpeechServiceConnection_SpeakerProfileId", config.personal_voice.speaker_profile_id)
            _set_property_value("SpeechServiceConnection_SynthesisStyle", config.personal_voice.style)
            _set_property_value("SpeechServiceConnection_SynthesisLanguage", config.personal_voice.language)
        elif config.voice_name:
            translation_config.voice_name = config.voice_name

        translation_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageId,
            "UND",
        )
        return translation_config

    def _create_auto_detect_config(self, speechsdk, languages: Sequence[str]):
        cleaned = _normalize_languages(languages)
        languageconfig = getattr(speechsdk, "languageconfig", None)
        if languageconfig is None:
            raise LiveInterpreterError("Installed azure-cognitiveservices-speech SDK does not expose languageconfig.")
        auto_detect_cls = getattr(languageconfig, "AutoDetectSourceLanguageConfig", None)
        if auto_detect_cls is None:
            raise LiveInterpreterError(
                "Auto language detection is unavailable in this version of azure-cognitiveservices-speech."
            )

        if cleaned:
            from_languages = getattr(auto_detect_cls, "from_languages", None)
            if callable(from_languages):
                return from_languages(list(cleaned))
            try:
                return auto_detect_cls(languages=list(cleaned))
            except TypeError as exc:  # pragma: no cover - defensive fallback
                raise LiveInterpreterError(
                    "Auto language detection is not supported by the installed Speech SDK."
                ) from exc

        # No explicit languages provided, fall back to the SDK's open-range detection.
        try:
            return auto_detect_cls()
        except TypeError:
            from_open_range = getattr(auto_detect_cls, "from_open_range", None)
            if callable(from_open_range):
                return from_open_range()
            raise LiveInterpreterError(
                "Auto language detection requires specifying AZURE_SPEECH_TRANSLATION_SOURCE_LANGUAGES "
                "when the installed Speech SDK does not support open-range detection."
            )
