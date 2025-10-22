# GPT Realtime WebSocket Sample

This directory contains reusable sample code for communicating with Azure OpenAI's GPT Realtime API via WebSocket.

## Overview

The [`gpt_realtime_websocket_sample.py`](gpt_realtime_websocket_sample.py) module provides a clean, production-ready client for WebSocket communication with the Azure OpenAI Realtime API.

## Features

- ✅ WebSocket connection management with Azure OpenAI
- ✅ Session configuration (instructions, voice, modalities)
- ✅ Send text and/or audio input
- ✅ Receive streaming text and audio responses
- ✅ Comprehensive error handling
- ✅ Detailed logging support
- ✅ Type hints for better IDE support

## Installation

Install the required dependencies:

```bash
pip install openai
```

## Quick Start

```python
import asyncio
from examples.gpt_realtime_websocket_sample import RealtimeWebSocketClient, RealtimeConfig

async def main():
    # Configure the client
    config = RealtimeConfig(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key-here",
        deployment="gpt-4o-realtime-preview",
        api_version="2025-08-28",
        voice="alloy",
    )
    
    # Create the client
    client = RealtimeWebSocketClient(config)
    
    # Send a message and get response
    transcript, audio_bytes = await client.send_message(
        user_text="Hello, how are you?",
        instructions="You are a friendly assistant.",
    )
    
    print(f"AI Response: {transcript}")
    print(f"Audio size: {len(audio_bytes)} bytes")

# Run the example
asyncio.run(main())
```

## Configuration

### RealtimeConfig

The `RealtimeConfig` dataclass contains all necessary configuration:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `endpoint` | str | Yes | - | Azure OpenAI endpoint URL |
| `api_key` | str | Yes | - | Azure OpenAI API key |
| `deployment` | str | Yes | - | Deployment name (e.g., "gpt-4o-realtime-preview") |
| `api_version` | str | No | "2025-08-28" | API version |
| `voice` | str | No | "alloy" | Default voice for audio output |
| `sample_rate_hz` | int | No | 24000 | Audio sample rate |

## Usage Examples

### Example 1: Text Input with Audio Output

```python
transcript, audio = await client.send_message(
    user_text="Tell me a joke.",
    instructions="You are a comedian.",
)
```

### Example 2: Custom Voice

```python
transcript, audio = await client.send_message(
    user_text="What's the weather?",
    voice="shimmer",  # Use a different voice
)
```

### Example 3: Text-Only Output

```python
transcript, audio = await client.send_message(
    user_text="Count from 1 to 5.",
    output_modalities=["text"],  # No audio output
)
```

### Example 4: Audio Input

```python
# Assuming you have PCM16 audio data
audio_chunks = [audio_chunk1, audio_chunk2]

transcript, audio = await client.send_message(
    input_audio_chunks=audio_chunks,
    instructions="Transcribe and respond to the user.",
)
```

### Example 5: Both Text and Audio Input

```python
transcript, audio = await client.send_message(
    user_text="Here's what I'm thinking about:",
    input_audio_chunks=audio_chunks,
    instructions="Listen and respond thoughtfully.",
)
```

## API Reference

### RealtimeWebSocketClient

#### `__init__(config: RealtimeConfig)`

Initialize the client with configuration.

#### `async send_message(...) -> Tuple[str, bytes]`

Send a message and receive a response.

**Parameters:**
- `user_text` (Optional[str]): Text message from the user
- `input_audio_chunks` (Optional[Sequence[bytes]]): List of PCM16 audio chunks
- `instructions` (str): System instructions for the AI (default: "You are a helpful assistant.")
- `voice` (Optional[str]): Voice to use, overrides config default
- `output_modalities` (Optional[Sequence[str]]): Output types ["text", "audio"]

**Returns:**
- `Tuple[str, bytes]`: (transcript_text, audio_bytes)

**Raises:**
- `ValueError`: If neither text nor audio input is provided
- `RuntimeError`: If the API returns an error

## Event Types Handled

The client handles the following WebSocket events:

- `response.text.delta` - Streaming text deltas
- `response.output_text.delta` - Output text deltas
- `response.output_audio_transcript.delta` - Audio transcript deltas
- `response.audio.delta` - Streaming audio deltas (base64 encoded)
- `response.output_audio.delta` - Output audio deltas
- `response.text.done` - Text completion
- `response.output_text.done` - Output text completion
- `response.audio.done` - Audio completion
- `response.done` - Full response completion
- `response.error` - Error events
- `error` - General errors

## Audio Format

- **Input Audio**: PCM16 (16-bit signed integer, mono)
- **Output Audio**: PCM16 (16-bit signed integer, mono)
- **Sample Rate**: Configurable (default: 24000 Hz)
- **Encoding**: Base64 for WebSocket transmission

## Logging

Enable logging to see detailed debug information:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Error Handling

The client provides comprehensive error handling:

```python
try:
    transcript, audio = await client.send_message(
        user_text="Hello!",
    )
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Environment Variables (Optional)

You can load configuration from environment variables:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-realtime-preview"
export AZURE_OPENAI_API_VERSION="2025-08-28"
export AZURE_OPENAI_VOICE="alloy"
```

## Integration with Original Code

This sample code is extracted from the main [`voice_client.py`](../src/voice_client.py) module. The key differences:

- **Simplified**: Removed personal voice synthesis and VoiceLive API support
- **Focused**: Only WebSocket communication with Realtime API
- **Reusable**: Clean interface for easy integration
- **Documented**: Comprehensive comments and examples

## License

This code is part of the audio-rt project.