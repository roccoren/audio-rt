"""
Small debugging utility that mirrors the browser flow:

1. Requests a new realtime session from the local FastAPI backend.
2. Opens the websocket to Azure using the returned URL and subprotocol hints.
3. Prints detailed logs for the entire handshake and first few messages.

Run with:
    python src/debug_realtime_client.py --api-base http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
from typing import Iterable, List

import httpx
import websockets
from websockets.exceptions import InvalidHandshake, InvalidStatusCode, NegotiationError


DEFAULT_INSTRUCTIONS = "Debug realtime session"


def _extend_protocols(session_protocols: Iterable[str] | None, client_secret: str) -> List[str]:
    """Build the websocket subprotocol list Azure expects."""
    ordered: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        if value in seen:
            return
        seen.add(value)
        ordered.append(value)

    add("realtime")
    add(f"openai-ephemeral-key-v1.{client_secret}")
    add(f"openai-insecure-session-token-v1.{client_secret}")

    if session_protocols:
        for protocol in session_protocols:
            if not isinstance(protocol, str):
                continue
            normalized = protocol.replace("{SESSION_SECRET}", client_secret).strip()
            if normalized == "openai-ephemeral-key-v1":
                normalized = f"openai-ephemeral-key-v1.{client_secret}"
            elif normalized == "openai-insecure-session-token-v1":
                normalized = f"openai-insecure-session-token-v1.{client_secret}"
            add(normalized)

    return ordered


async def fetch_session(api_base: str, instructions: str) -> dict:
    url = api_base.rstrip("/") + "/api/realtime/session"
    print(f"[session] POST {url}", flush=True)
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            url,
            json={"instructions": instructions},
            headers={"Content-Type": "application/json"},
        )
    print(f"[session] status={response.status_code}", flush=True)
    if response.status_code >= 400:
        print("[session] body:", response.text, file=sys.stderr, flush=True)
        response.raise_for_status()
    payload = response.json()
    print(json.dumps(payload, indent=2), flush=True)
    return payload


async def connect_websocket(
    ws_url: str,
    protocols: List[str],
    instructions: str,
    *,
    message_timeout: float,
    max_messages: int,
    test_tone_ms: float | None,
) -> None:
    print(f"[ws] connecting to {ws_url}", flush=True)
    print(f"[ws] subprotocols -> {protocols}", flush=True)
    try:
        async with websockets.connect(
            ws_url,
            subprotocols=protocols,
            open_timeout=5.0,
            close_timeout=5.0,
            ping_interval=None,
            ping_timeout=None,
        ) as websocket:
            print(f"[ws] handshake complete, server accepted subprotocol={websocket.subprotocol!r}", flush=True)

            session_update = json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": instructions,
                        "output_modalities": ["audio"],
                    },
                }
            )
            await websocket.send(session_update)
            print("[ws] sent session.update", flush=True)

            response_create = json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                    },
                }
            )
            await websocket.send(response_create)
            print("[ws] sent response.create", flush=True)

            if test_tone_ms and test_tone_ms > 0:
                sample_rate = 24000
                total_samples = max(1, int(sample_rate * (test_tone_ms / 1000.0)))
                amplitude = 0.3
                pcm_bytes = bytearray()
                for index in range(total_samples):
                    value = amplitude * math.sin(2 * math.pi * 440 * index / sample_rate)
                    int_sample = int(max(-1.0, min(1.0, value)) * 32767)
                    pcm_bytes.extend(int_sample.to_bytes(2, byteorder="little", signed=True))
                chunk_base64 = base64.b64encode(pcm_bytes).decode("ascii")
                append_message = json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": chunk_base64,
                    }
                )
                await websocket.send(append_message)
                print(f"[ws] sent test audio chunk ({test_tone_ms:.1f} ms)", flush=True)
                await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                print("[ws] sent commit for test chunk", flush=True)

            for index in range(max_messages):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=message_timeout)
                except asyncio.TimeoutError:
                    print(f"[ws] no message received within {message_timeout}s, stopping.", flush=True)
                    break
                if isinstance(message, bytes):
                    print(f"[ws] message[{index}] {len(message)} bytes", flush=True)
                else:
                    print(f"[ws] message[{index}] {message}", flush=True)

    except InvalidStatusCode as exc:
        print(f"[ws] handshake rejected with status={exc.status_code}", file=sys.stderr, flush=True)
        headers = getattr(exc, "headers", None)
        if headers:
            print("[ws] response headers:", headers, file=sys.stderr, flush=True)
        raise
    except NegotiationError as exc:
        print(f"[ws] subprotocol negotiation failed: {exc}", file=sys.stderr, flush=True)
        raise
    except InvalidHandshake as exc:
        print(f"[ws] invalid handshake: {exc}", file=sys.stderr, flush=True)
        raise


async def async_main(args: argparse.Namespace) -> None:
    session_info = await fetch_session(args.api_base, args.instructions)
    client_secret = session_info.get("clientSecret")
    ws_url = session_info.get("wsUrl")
    ws_protocols = session_info.get("wsProtocols")

    if not client_secret or not ws_url:
        raise RuntimeError("Session response missing clientSecret or wsUrl.")

    protocols = _extend_protocols(ws_protocols, client_secret)
    await connect_websocket(
        ws_url,
        protocols,
        args.instructions,
        message_timeout=args.message_timeout,
        max_messages=args.max_messages,
        test_tone_ms=args.test_tone_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Azure OpenAI realtime websocket handshake.")
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000",
        help="FastAPI backend base URL (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--instructions",
        default=DEFAULT_INSTRUCTIONS,
        help="Instructions string passed when creating the session.",
    )
    parser.add_argument(
        "--message-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for each incoming websocket message (default: 3).",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=5,
        help="Maximum number of websocket messages to print before exiting.",
    )
    parser.add_argument(
        "--test-tone-ms",
        type=float,
        default=0.0,
        help="When >0, send a synthetic 440Hz tone of the given duration (ms) after connecting.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(async_main(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
