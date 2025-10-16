#!/usr/bin/env python3
"""
Utility helpers for managing Azure Personal Voice resources and synthesizing audio.

The module wraps the sample workflow published in the Azure Cognitive Services Speech SDK
repository so it can live next to the realtime demo. It expects the `customvoice` helper
package to be available. Copy the folder from:
https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/samples/custom-voice/python/customvoice
into the project root (keeping the same `customvoice/` directory name) before running the
commands below.

Examples:

    # Create the project, upload consent + audio, and print the speaker profile id
    python src/personal_voice_setup.py create \\
        --project-id personal-voice-project-1 \\
        --consent-id personal-voice-consent-1 \\
        --personal-voice-id personal-voice-1 \\
        --consent-file TestData/VoiceTalentVerbalStatement.wav \\
        --audio-folder TestData/voice \\
        --voice-talent-name "Sample Voice Actor" \\
        --company-name "Contoso"

    # Synthesize text once you have the speaker profile id
    python src/personal_voice_setup.py synthesize \\
        --speaker-profile-id <speaker_profile_id> \\
        --text "This is zero shot voice. Test 2." \\
        --output output_sdk.wav

    # Delete the project resources when you are done experimenting
    python src/personal_voice_setup.py cleanup \\
        --project-id personal-voice-project-1 \\
        --consent-id personal-voice-consent-1 \\
        --personal-voice-id personal-voice-1

    # List all personal voices (optionally scoped to a project)
    python src/personal_voice_setup.py list --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
SCRIPT_ROOT = Path(__file__).resolve().parent.parent

def _add_path(path: Path) -> None:
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

for candidate in {PROJECT_ROOT, SCRIPT_ROOT}:
    _add_path(candidate)
    customvoice_dir = candidate / "customvoice"
    if customvoice_dir.is_dir():
        _add_path(customvoice_dir)

try:
    import customvoice  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - optional dependency
    if getattr(exc, "name", "") == "customvoice":
        raise SystemExit(
            "The 'customvoice' helper package is missing. Copy it from the Azure Speech SDK "
            "samples repository (see docstring for link) before running this script."
        ) from exc
    raise SystemExit(f"Failed to import a dependency from the 'customvoice' package: {exc}") from exc

try:
    import azure.cognitiveservices.speech as speechsdk  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "azure-cognitiveservices-speech is required. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc


LOGGER = logging.getLogger("personal_voice")


def create_personal_voice(
    config: customvoice.Config,
    *,
    project_id: str,
    consent_id: str,
    personal_voice_id: str,
    consent_file: Path,
    voice_talent_name: str,
    company_name: str,
    audio_folder: Path,
    locale: str,
) -> str:
    """Create a personal voice project, upload consent + audio, and return the speaker profile id."""
    project = customvoice.Project.create(config, project_id, customvoice.ProjectKind.PersonalVoice)
    LOGGER.info("Project created (id=%s)", project.id)

    consent = customvoice.Consent.create(
        config,
        project_id,
        consent_id,
        voice_talent_name,
        company_name,
        str(consent_file),
        locale,
    )
    if consent.status == customvoice.Status.Failed:
        raise RuntimeError(f"Create consent failed (consent id={consent.id})")
    LOGGER.info("Consent status=%s (id=%s)", consent.status, consent.id)

    personal_voice = customvoice.PersonalVoice.create(
        config,
        project_id,
        personal_voice_id,
        consent_id,
        str(audio_folder),
    )
    if personal_voice.status == customvoice.Status.Failed:
        raise RuntimeError(f"Create personal voice failed (id={personal_voice.id})")
    LOGGER.info(
        "Personal voice status=%s (id=%s, speaker_profile_id=%s)",
        personal_voice.status,
        personal_voice.id,
        personal_voice.speaker_profile_id,
    )
    return personal_voice.speaker_profile_id


def synthesize_once(
    *,
    speech_key: str,
    speech_region: str,
    speaker_profile_id: str,
    text: str,
    output_path: Path,
    base_voice: str,
    locale: str,
    style: str,
) -> None:
    """Synthesize the provided text into a WAV file using the personal voice."""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_path))
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    ssml = (
        "<speak version='1.0' xml:lang='{locale}' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='{voice}'>"
        "<mstts:ttsembedding speakerProfileId='{profile}'/>"
        "<mstts:express-as style='{style}'>"
        "<lang xml:lang='{locale}'>{text}</lang>"
        "</mstts:express-as>"
        "</voice></speak>"
    ).format(
        locale=locale,
        voice=base_voice,
        profile=speaker_profile_id,
        style=style,
        text=text,
    )

    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        LOGGER.info("Speech synthesized to %s", output_path)
        LOGGER.debug("Result id: %s", result.result_id)
        return
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        message = details.error_details or str(details.reason)
        raise RuntimeError(f"Speech synthesis canceled: {message}")
    raise RuntimeError(f"Unexpected synthesis result: {result.reason}")


def cleanup_resources(
    config: customvoice.Config,
    *,
    project_id: str,
    consent_id: str,
    personal_voice_id: str,
) -> None:
    """Delete the generated personal voice resources."""
    customvoice.PersonalVoice.delete(config, personal_voice_id)
    customvoice.Consent.delete(config, consent_id)
    customvoice.Project.delete(config, project_id)
    LOGGER.info("Deleted project=%s, consent=%s, personal_voice=%s", project_id, consent_id, personal_voice_id)


def list_personal_voices(
    config: customvoice.Config,
    *,
    project_id: str | None,
    output_json: bool,
) -> None:
    """List existing personal voices for the Speech resource."""
    voices = customvoice.PersonalVoice.list(config, project_id)
    if not voices:
        scope = f"project {project_id!r}" if project_id else "the Speech resource"
        print(f"No personal voices found for {scope}.")
        return

    if output_json:
        payload = [
            {
                "id": voice.id,
                "displayName": getattr(voice, "display_name", ""),
                "description": getattr(voice, "description", ""),
                "status": getattr(voice.status, "name", ""),
                "projectId": getattr(voice, "project_id", ""),
                "consentId": getattr(voice, "consent_id", ""),
                "speakerProfileId": getattr(voice, "speaker_profile_id", ""),
                "createdDateTime": getattr(voice, "created_date_time", ""),
                "lastActionDateTime": getattr(voice, "last_action_date_time", ""),
            }
            for voice in voices
        ]
        print(json.dumps(payload, indent=2))
        return

    print(f"Found {len(voices)} personal voice(s):")
    for voice in voices:
        print(
            f"- id={voice.id} status={voice.status.name} speaker_profile_id={voice.speaker_profile_id} "
            f"project_id={voice.project_id} display_name={getattr(voice, 'display_name', '')}"
        )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--speech-key",
        default=os.environ.get("AZURE_SPEECH_KEY"),
        help="Azure Speech resource key (defaults to AZURE_SPEECH_KEY).",
    )
    parser.add_argument(
        "--speech-region",
        default=os.environ.get("AZURE_SPEECH_REGION"),
        help="Azure Speech resource region (defaults to AZURE_SPEECH_REGION).",
    )
    parser.add_argument(
        "--locale",
        default=os.environ.get("AZURE_SPEECH_LANGUAGE", "en-US"),
        help="Locale for consent/audio assets (default: en-US).",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azure Personal Voice helper script.")
    parser.set_defaults(func=None)
    _add_common_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a personal voice project + resources.")
    create_parser.add_argument("--project-id", required=True)
    create_parser.add_argument("--consent-id", required=True)
    create_parser.add_argument("--personal-voice-id", required=True)
    create_parser.add_argument("--consent-file", required=True, type=Path)
    create_parser.add_argument("--voice-talent-name", required=True)
    create_parser.add_argument("--company-name", required=True)
    create_parser.add_argument("--audio-folder", required=True, type=Path)
    create_parser.set_defaults(func="create")

    synth_parser = subparsers.add_parser("synthesize", help="Generate a WAV file using a personal voice.")
    synth_parser.add_argument("--speaker-profile-id", required=True)
    synth_parser.add_argument("--text", required=True)
    synth_parser.add_argument("--output", required=True, type=Path)
    synth_parser.add_argument(
        "--voice-name",
        default=os.environ.get("AZURE_SPEECH_VOICE_NAME", "DragonLatestNeural"),
        help="Base voice name for the SSML payload.",
    )
    synth_parser.add_argument(
        "--style",
        default=os.environ.get("AZURE_SPEECH_VOICE_STYLE", "Prompt"),
        help="Speaking style for the SSML payload.",
    )
    synth_parser.set_defaults(func="synthesize")

    list_parser = subparsers.add_parser("list", help="List personal voices in the Speech resource.")
    list_parser.add_argument(
        "--project-id",
        default=None,
        help="Optional project id filter (shows all voices when omitted).",
    )
    list_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print the list as JSON.",
    )
    list_parser.set_defaults(func="list")

    cleanup_parser = subparsers.add_parser("cleanup", help="Delete the personal voice resources.")
    cleanup_parser.add_argument("--project-id", required=True)
    cleanup_parser.add_argument("--consent-id", required=True)
    cleanup_parser.add_argument("--personal-voice-id", required=True)
    cleanup_parser.set_defaults(func="cleanup")

    return parser.parse_args()


def require_credentials(args: argparse.Namespace) -> tuple[str, str]:
    key = (args.speech_key or "").strip()
    region = (args.speech_region or "").strip()
    if not key or not region:
        raise SystemExit(
            "Provide Azure Speech credentials via --speech-key/--speech-region "
            "or set AZURE_SPEECH_KEY/AZURE_SPEECH_REGION."
        )
    return key, region


def main() -> None:
    args = parse_args()
    speech_key, speech_region = require_credentials(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = customvoice.Config(speech_key, speech_region, LOGGER)

    if args.func == "create":
        speaker_profile_id = create_personal_voice(
            config,
            project_id=args.project_id,
            consent_id=args.consent_id,
            personal_voice_id=args.personal_voice_id,
            consent_file=args.consent_file,
            voice_talent_name=args.voice_talent_name,
            company_name=args.company_name,
            audio_folder=args.audio_folder,
            locale=args.locale,
        )
        print(speaker_profile_id)
        return

    if args.func == "synthesize":
        synthesize_once(
            speech_key=speech_key,
            speech_region=speech_region,
            speaker_profile_id=args.speaker_profile_id,
            text=args.text,
            output_path=args.output,
            base_voice=args.voice_name,
            locale=args.locale,
            style=args.style,
        )
        return

    if args.func == "list":
        list_personal_voices(
            config,
            project_id=args.project_id,
            output_json=args.as_json,
        )
        return

    if args.func == "cleanup":
        cleanup_resources(
            config,
            project_id=args.project_id,
            consent_id=args.consent_id,
            personal_voice_id=args.personal_voice_id,
        )
        return

    raise SystemExit("No command provided.")


if __name__ == "__main__":
    main()
