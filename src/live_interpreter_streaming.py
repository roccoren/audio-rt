"""Continuous Live Interpreter with Personal Voice using Azure Speech SDK.

This module provides real-time speech translation with continuous recognition,
similar to the Flask sample. It supports:
- Continuous microphone input with auto language detection
- Real-time translation to target language(s)
- Speech synthesis with personal voice (Live Interpreter)
- WebSocket streaming for browser audio
- Audio playback of synthesized translation

Based on Azure Speech SDK Live Interpreter functionality.
"""

import asyncio
import base64
import io
import logging
import os
import queue
import re
import threading
import wave
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class StreamingTranslationConfig:
    """Configuration for streaming Live Interpreter sessions."""
    subscription_key: str
    region: str
    target_language: str = "en"
    voice_name: str = "personal-voice"
    speaker_profile_id: Optional[str] = None
    auto_detect: bool = True
    output_sample_rate: Optional[int] = None


@dataclass
class TranslationEvent:
    """Event from streaming translation."""
    event_type: str  # 'recognizing', 'recognized', 'audio', 'error', 'started', 'stopped'
    detected_language: Optional[str] = None
    original_text: Optional[str] = None
    translation: Optional[str] = None
    audio_data: Optional[bytes] = None
    audio_base64: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    offset: Optional[float] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None


class LiveInterpreterStreamingSession:
    """Continuous streaming translation session using Azure Speech SDK Live Interpreter."""
    
    def __init__(self, config: StreamingTranslationConfig):
        self.config = config
        self._speechsdk = None
        self._recognizer: Optional[Any] = None
        self._push_stream: Optional[Any] = None
        self._is_running = False
        self._event_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()
        self._session_stopped = threading.Event()
        
    def _ensure_speechsdk(self):
        """Lazy load Azure Speech SDK."""
        if self._speechsdk is not None:
            return self._speechsdk
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise RuntimeError(
                "azure-cognitiveservices-speech is required for Live Interpreter."
            ) from exc
        self._speechsdk = speechsdk
        return speechsdk
    
    def _get_speech_config(self):
        """Get Speech configuration with Live Interpreter v2 endpoint."""
        speechsdk = self._ensure_speechsdk()
        
        # Use v2 endpoint for Live Interpreter with personal voice
        v2_endpoint = (
            f"wss://{self.config.region}.stt.speech.microsoft.com/"
            f"speech/universal/v2?setfeature=zeroshotttsflight"
        )
        
        translation_config = speechsdk.translation.SpeechTranslationConfig(
            subscription=self.config.subscription_key,
            endpoint=v2_endpoint
        )
        
        # Add target language
        translation_config.add_target_language(self.config.target_language)
        
        # Enable Live Interpreter with personal voice
        translation_config.voice_name = self.config.voice_name
        
        # Set speaker profile ID for personal voice if provided
        if self.config.speaker_profile_id:
            translation_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_SpeakerProfileId,
                self.config.speaker_profile_id
            )
        
        output_format = self._resolve_output_format(speechsdk)
        if output_format is not None:
            try:
                translation_config.set_speech_synthesis_output_format(output_format)
            except AttributeError:
                translation_config.speech_synthesis_output_format = output_format

        logger.info(
            "Live Interpreter config: endpoint=%s, target=%s, voice=%s, profile=%s",
            v2_endpoint, self.config.target_language, self.config.voice_name,
            self.config.speaker_profile_id
        )
        
        return translation_config

    def _resolve_output_format(self, speechsdk):
        sample_rate = self.config.output_sample_rate
        if not sample_rate:
            return None

        enum_cls = getattr(speechsdk, "SpeechSynthesisOutputFormat", None)
        if enum_cls is None:
            return None

        candidate_formats = [
            (48000, "Raw48Khz16BitMonoPcm"),
            (44100, "Raw44Khz16BitMonoPcm"),
            (32000, "Raw32Khz16BitMonoPcm"),
            (24000, "Raw24Khz16BitMonoPcm"),
            (22050, "Raw22Khz16BitMonoPcm"),
            (16000, "Raw16Khz16BitMonoPcm"),
            (8000, "Raw8Khz16BitMonoPcm"),
        ]

        mapping: dict[int, object] = {}
        for rate, attr in candidate_formats:
            value = getattr(enum_cls, attr, None)
            if value is not None:
                mapping[rate] = value

        if not mapping:
            return None

        if sample_rate not in mapping:
            sample_rate = min(mapping.keys(), key=lambda rate: abs(rate - sample_rate))

        return mapping.get(sample_rate)
    
    def start(self):
        """Start continuous recognition session."""
        if self._is_running:
            logger.warning("Session already running")
            return
        
        speechsdk = self._ensure_speechsdk()
        translation_config = self._get_speech_config()
        
        # Create audio config with push stream for browser audio
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)
        
        # Create auto-detect config for universal language detection
        recognizer_kwargs = {
            "translation_config": translation_config,
            "audio_config": audio_config
        }
        
        if self.config.auto_detect:
            try:
                auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig()
                recognizer_kwargs["auto_detect_source_language_config"] = auto_detect_config
                logger.info("Universal language auto-detection enabled")
            except Exception as e:
                logger.warning("Auto-detect not available, using default: %s", e)
        
        # Create recognizer
        self._recognizer = speechsdk.translation.TranslationRecognizer(**recognizer_kwargs)
        
        # Connect event handlers
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.synthesizing.connect(self._on_synthesizing)
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.session_stopped.connect(self._on_session_stopped)
        
        # Start continuous recognition
        self._recognizer.start_continuous_recognition()
        self._is_running = True
        self._session_stopped.clear()
        
        # Queue started event
        self._event_queue.put(TranslationEvent(
            event_type='started',
            translation=f"Live Interpreter started (target: {self.config.target_language})"
        ))
        
        logger.info("✅ Continuous recognition started")
    
    def stop(self):
        """Stop continuous recognition session."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._recognizer:
            try:
                self._recognizer.stop_continuous_recognition()
            except Exception as e:
                logger.error("Error stopping recognizer: %s", e)
        
        if self._push_stream:
            try:
                self._push_stream.close()
            except Exception as e:
                logger.error("Error closing stream: %s", e)
        
        # Queue stopped events
        self._event_queue.put(TranslationEvent(event_type='stopped'))
        self._audio_queue.put({'type': 'stopped'})
        
        # Wait for session to stop
        self._session_stopped.wait(timeout=5)
        
        logger.info("✅ Translation stopped")
    
    def push_audio(self, audio_bytes: bytes):
        """Push audio data to the recognition stream."""
        if not self._is_running or not self._push_stream:
            return
        
        try:
            self._push_stream.write(audio_bytes)
        except Exception as e:
            logger.error("Error pushing audio: %s", e)
    
    def get_event(self, timeout: float = 1.0) -> Optional[TranslationEvent]:
        """Get next translation event from queue (blocking)."""
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_audio(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get next audio event from queue (blocking)."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _on_recognizing(self, evt):
        """Handle interim recognition results."""
        speechsdk = self._speechsdk
        
        # Get detected language
        detected_lang = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult,
            'unknown'
        )
        
        if evt.result.text:
            translation = evt.result.translations.get(self.config.target_language, '')
            
            logger.debug(
                "🔄 [%s] Recognizing: %s → %s",
                detected_lang, evt.result.text, translation or '(translating...)'
            )
            
            self._event_queue.put(TranslationEvent(
                event_type='recognizing',
                detected_language=detected_lang,
                original_text=evt.result.text,
                translation=translation,
                offset=evt.result.offset / 10_000_000.0
            ))
    
    def _on_recognized(self, evt):
        """Handle final recognition results."""
        speechsdk = self._speechsdk
        
        # Get detected language
        detected_lang = evt.result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult,
            'unknown'
        )
        
        logger.debug(
            "✅ RECOGNIZED - Language: %s, Reason: %s, Text: '%s'",
            detected_lang, evt.result.reason, evt.result.text
        )
        
        if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
            if evt.result.text:
                translation = evt.result.translations.get(self.config.target_language, '')
                
                logger.info(
                    "✅ [%s] Recognized: %s → %s",
                    detected_lang, evt.result.text, translation
                )
                
                self._event_queue.put(TranslationEvent(
                    event_type='recognized',
                    detected_language=detected_lang,
                    original_text=evt.result.text,
                    translation=translation,
                    offset=evt.result.offset / 10_000_000.0,
                    duration=evt.result.duration / 10_000_000.0
                ))
        
        elif evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if evt.result.text:
                self._event_queue.put(TranslationEvent(
                    event_type='recognized',
                    detected_language=detected_lang,
                    original_text=evt.result.text,
                    translation='',
                    offset=evt.result.offset / 10_000_000.0,
                    duration=evt.result.duration / 10_000_000.0
                ))
        
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.warning("⚠️ NOMATCH: Speech could not be recognized")
            no_match_details = evt.result.no_match_details
            if no_match_details:
                logger.warning("   Reason: %s", no_match_details.reason)
    
    def _on_synthesizing(self, evt):
        """Handle synthesized audio from Live Interpreter."""
        if evt.result.audio and len(evt.result.audio) > 0:
            raw_audio = evt.result.audio
            pcm_bytes = raw_audio
            sample_rate: Optional[int] = None

            # Attempt to read declared output format (e.g. audio-24khz-16bit-mono-pcm)
            format_property = None
            if getattr(evt.result, "properties", None) is not None:
                try:
                    speechsdk = self._ensure_speechsdk()
                    format_property = getattr(speechsdk.PropertyId, "SpeechServiceResponse_AudioOutputFormat", None)
                except Exception:
                    format_property = None
            if format_property is not None:
                try:
                    format_value = evt.result.properties.get(format_property, None)  # type: ignore[attr-defined]
                except Exception:
                    format_value = None
                if isinstance(format_value, str):
                    match = re.search(r"(\d+)\s*k(?:hz|Hz)", format_value)
                    if match:
                        try:
                            sample_rate = int(match.group(1)) * 1000
                        except ValueError:
                            sample_rate = None

            # Strip WAV headers when present.
            if len(raw_audio) >= 12 and raw_audio[:4] == b"RIFF" and raw_audio[8:12] == b"WAVE":
                try:
                    with wave.open(io.BytesIO(raw_audio), "rb") as wav_file:
                        sample_rate = sample_rate or wav_file.getframerate()
                        pcm_bytes = wav_file.readframes(wav_file.getnframes())
                except (wave.Error, RuntimeError):
                    pcm_bytes = raw_audio

            # Infer sample rate via duration metadata when necessary.
            if sample_rate is None:
                audio_duration = getattr(evt.result, "audio_duration", None)
                if audio_duration is None:
                    audio_duration = getattr(evt.result, "duration", None)
                if audio_duration:
                    try:
                        if hasattr(audio_duration, "total_seconds"):
                            duration_seconds = audio_duration.total_seconds()
                        else:
                            duration_seconds = float(audio_duration) / 10_000_000.0
                        if duration_seconds and duration_seconds > 0:
                            total_samples = len(pcm_bytes) / 2.0
                            sample_rate = int(round(total_samples / duration_seconds))
                    except Exception:
                        sample_rate = None

            if sample_rate is None:
                sample_rate = self.config.output_sample_rate or 16000

            audio_b64 = base64.b64encode(pcm_bytes).decode("utf-8")

            logger.info(
                "🔊 Audio synthesized: raw=%d bytes, pcm=%d bytes, sample_rate=%s",
                len(raw_audio),
                len(pcm_bytes),
                sample_rate,
            )

            self._audio_queue.put({
                "type": "audio",
                "data": audio_b64,
                "size": len(pcm_bytes),
                "sampleRate": sample_rate,
            })

            self._event_queue.put(TranslationEvent(
                event_type='audio',
                audio_data=pcm_bytes,
                audio_base64=audio_b64,
                audio_sample_rate=sample_rate,
            ))
    
    def _on_canceled(self, evt):
        """Handle cancellation."""
        error_msg = f'❌ Canceled: {evt.reason}'
        
        if hasattr(evt, 'cancellation_details'):
            details = evt.cancellation_details
            if hasattr(details, 'error_details') and details.error_details:
                error_msg += f' - {details.error_details}'
        
        logger.error(error_msg)
        
        self._event_queue.put(TranslationEvent(
            event_type='error',
            error_message=error_msg
        ))
    
    def _on_session_stopped(self, evt):
        """Handle session stop."""
        logger.info("🛑 SESSION STOPPED")
        self._session_stopped.set()
        self._event_queue.put(TranslationEvent(event_type='session_stopped'))


def load_streaming_config_from_env() -> StreamingTranslationConfig:
    """Load streaming configuration from environment variables."""
    subscription_key = (
        os.getenv('AZURE_SPEECH_TRANSLATION_KEY') or
        os.getenv('SPEECH__SUBSCRIPTION__KEY') or
        os.getenv('AZURE_SPEECH_KEY')
    )
    if not subscription_key:
        raise ValueError("AZURE_SPEECH_TRANSLATION_KEY is required")
    
    region = (
        os.getenv('AZURE_SPEECH_TRANSLATION_REGION') or
        os.getenv('SPEECH__SERVICE__REGION') or
        os.getenv('AZURE_SPEECH_REGION') or
        'southeastasia'
    )
    
    target_language = os.getenv('AZURE_SPEECH_TRANSLATION_TARGET_LANGUAGE', 'en')
    voice_name = os.getenv('AZURE_SPEECH_TRANSLATION_VOICE', 'personal-voice')
    speaker_profile_id = os.getenv('AZURE_SPEECH_SPEAKER_PROFILE_ID')
    
    auto_detect = os.getenv('AZURE_SPEECH_TRANSLATION_AUTO_DETECT', 'true').lower() in ('true', '1', 'yes')

    output_sample_rate_raw = os.getenv('AZURE_SPEECH_TRANSLATION_OUTPUT_SAMPLE_RATE')
    try:
        output_sample_rate = int(output_sample_rate_raw) if output_sample_rate_raw else None
    except ValueError:
        logger.warning("Invalid AZURE_SPEECH_TRANSLATION_OUTPUT_SAMPLE_RATE=%s; ignoring.", output_sample_rate_raw)
        output_sample_rate = None
    
    return StreamingTranslationConfig(
        subscription_key=subscription_key,
        region=region,
        target_language=target_language,
        voice_name=voice_name,
        speaker_profile_id=speaker_profile_id,
        auto_detect=auto_detect,
        output_sample_rate=output_sample_rate,
    )
