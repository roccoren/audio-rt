"""
Sample code for communicating with Azure OpenAI GPT Realtime API via WebSocket.

This module demonstrates how to:
1. Connect to the Azure OpenAI Realtime API using WebSocket
2. Configure a session with instructions and voice settings
3. Send user text and audio input
4. Receive streaming text and audio responses
5. Handle errors and events

Usage:
    from examples.gpt_realtime_websocket_sample import RealtimeWebSocketClient, RealtimeConfig
    
    config = RealtimeConfig(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key",
        deployment="gpt-4o-realtime-preview",
        api_version="2025-08-28"
    )
    
    client = RealtimeWebSocketClient(config)
    transcript, audio_bytes = await client.send_message(
        user_text="Hello, how are you?",
        instructions="You are a helpful assistant."
    )
"""

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from openai import AsyncAzureOpenAI


@dataclass
class RealtimeConfig:
    """Configuration for Azure OpenAI Realtime API connection."""
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2025-08-28"
    voice: str = "alloy"
    sample_rate_hz: int = 24000


class RealtimeWebSocketClient:
    """
    A reusable client for WebSocket communication with Azure OpenAI Realtime API.
    
    This class handles:
    - WebSocket connection management
    - Session configuration
    - Sending text and audio input
    - Receiving and parsing streaming responses
    - Error handling
    """
    
    def __init__(self, config: RealtimeConfig):
        """
        Initialize the Realtime WebSocket client.
        
        Args:
            config: Configuration containing endpoint, API key, deployment, etc.
        """
        self.config = config
        self._logger = logging.getLogger(__name__)
    
    async def send_message(
        self,
        user_text: Optional[str] = None,
        input_audio_chunks: Optional[Sequence[bytes]] = None,
        *,
        instructions: str = "You are a helpful assistant.",
        voice: Optional[str] = None,
        output_modalities: Optional[Sequence[str]] = None,
    ) -> Tuple[str, bytes]:
        """
        Send a message to the Realtime API and receive the response.
        
        Args:
            user_text: Text message from the user (optional if audio is provided)
            input_audio_chunks: List of audio byte chunks (optional if text is provided)
            instructions: System instructions for the AI assistant
            voice: Voice to use for audio output (overrides config default)
            output_modalities: List of output types ["text", "audio"] (defaults to both)
        
        Returns:
            Tuple of (transcript_text, audio_bytes)
        
        Raises:
            ValueError: If neither text nor audio input is provided
            RuntimeError: If the API returns an error
        """
        if not user_text and not input_audio_chunks:
            raise ValueError("Either user_text or input_audio_chunks must be provided.")
        
        voice_name = voice or self.config.voice
        modalities = output_modalities or ["text", "audio"]
        
        # Create Azure OpenAI client
        client = AsyncAzureOpenAI(
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version,
        )
        
        try:
            # Connect to the Realtime API via WebSocket
            async with client.beta.realtime.connect(model=self.config.deployment) as connection:
                # Step 1: Configure the session
                await self._configure_session(
                    connection,
                    instructions=instructions,
                    output_modalities=modalities,
                    voice_name=voice_name,
                )
                
                # Step 2: Send user input (text and/or audio)
                if user_text:
                    await self._send_user_text(connection, user_text)
                
                if input_audio_chunks:
                    await self._send_user_audio(connection, input_audio_chunks)
                
                # Step 3: Create a response request
                await connection.response.create(
                    response={
                        "modalities": list(dict.fromkeys(modalities)),
                        "voice": voice_name,
                    }
                )
                
                # Step 4: Collect streaming response
                collected_audio = bytearray()
                collected_text: list[str] = []
                final_text_segments: list[str] = []
                
                async for event in connection:
                    event_type = getattr(event, "type", None)
                    
                    # Handle text delta events
                    if event_type in {
                        "response.text.delta",
                        "response.output_text.delta",
                        "response.output_audio_transcript.delta",
                    }:
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            collected_text.append(delta)
                            self._logger.debug(f"Text delta: {delta}")
                    
                    # Handle audio delta events
                    elif event_type in {
                        "response.audio.delta",
                        "response.output_audio.delta",
                    }:
                        delta = getattr(event, "delta", None)
                        if delta:
                            # Decode base64 audio data
                            audio_chunk = base64.b64decode(delta)
                            collected_audio.extend(audio_chunk)
                            self._logger.debug(f"Audio delta: {len(audio_chunk)} bytes")
                    
                    # Handle completion events
                    elif event_type in {
                        "response.text.done",
                        "response.output_text.done",
                        "response.audio.done",
                    }:
                        self._logger.debug(f"Event: {event_type}")
                        continue
                    
                    # Handle response done event
                    elif event_type == "response.done":
                        response_payload = getattr(event, "response", None)
                        final_text_segments.extend(
                            self._extract_text_from_response(response_payload)
                        )
                        self._logger.info("Response completed")
                        break
                    
                    # Handle error events
                    elif event_type in {"response.error", "error"}:
                        error_payload = getattr(event, "error", None)
                        if isinstance(error_payload, dict):
                            message = error_payload.get("message", "Unknown error")
                        else:
                            message = str(error_payload or "Unknown error")
                        self._logger.error(f"Realtime API error: {message}")
                        raise RuntimeError(f"Realtime API error: {message}")
                    
                    # Log unhandled events
                    else:
                        self._logger.debug(f"Unhandled event type: {event_type}")
                
                # Combine text from deltas and final payload
                transcript = "".join(collected_text).strip()
                if not transcript and final_text_segments:
                    transcript = "".join(final_text_segments).strip()
                
                audio_bytes = bytes(collected_audio)
                
                self._logger.info(
                    f"Response collected: {len(transcript)} chars, {len(audio_bytes)} bytes audio"
                )
                
                return transcript, audio_bytes
        
        finally:
            await client.close()
    
    async def _configure_session(
        self,
        connection,
        *,
        instructions: str,
        output_modalities: Sequence[str],
        voice_name: str,
    ) -> None:
        """
        Configure the WebSocket session with instructions and settings.
        
        Args:
            connection: Active WebSocket connection
            instructions: System instructions for the AI
            output_modalities: List of output types ["text", "audio"]
            voice_name: Voice to use for audio output
        """
        session_payload: dict = {
            "instructions": instructions,
            "modalities": list(dict.fromkeys(output_modalities)),
            "voice": voice_name,
        }
        
        await connection.session.update(session=session_payload)
        self._logger.debug(f"Session configured: {session_payload}")
    
    async def _send_user_text(self, connection, text: str) -> None:
        """
        Send text input from the user.
        
        Args:
            connection: Active WebSocket connection
            text: User's text message
        """
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
        self._logger.debug(f"Sent user text: {text[:50]}...")
    
    async def _send_user_audio(self, connection, audio_chunks: Sequence[bytes]) -> None:
        """
        Send audio input from the user.
        
        Args:
            connection: Active WebSocket connection
            audio_chunks: List of PCM16 audio byte chunks
        """
        # Combine all audio chunks
        combined = bytearray()
        for chunk in audio_chunks:
            if chunk:
                combined.extend(chunk)
        
        if not combined:
            return
        
        # Encode to base64
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
        self._logger.debug(f"Sent user audio: {len(combined)} bytes")
    
    def _extract_text_from_response(self, payload: dict) -> list[str]:
        """
        Extract text segments from the response payload.
        
        Args:
            payload: Response payload from the API
        
        Returns:
            List of text segments
        """
        segments: list[str] = []
        if not isinstance(payload, dict):
            return segments
        
        outputs = payload.get("output")
        if not isinstance(outputs, list):
            return segments
        
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


# Example usage
async def example_usage():
    """Demonstrate how to use the RealtimeWebSocketClient."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = RealtimeConfig(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key-here",
        deployment="gpt-4o-realtime-preview",
        api_version="2025-08-28",
        voice="alloy",
    )
    
    # Create client
    client = RealtimeWebSocketClient(config)
    
    # Example 1: Send text message
    print("Example 1: Sending text message...")
    transcript, audio = await client.send_message(
        user_text="Tell me a short joke.",
        instructions="You are a funny comedian. Keep responses brief.",
    )
    print(f"Transcript: {transcript}")
    print(f"Audio size: {len(audio)} bytes")
    
    # Example 2: Send with custom voice
    print("\nExample 2: Using different voice...")
    transcript, audio = await client.send_message(
        user_text="What's the weather like?",
        instructions="You are a helpful weather assistant.",
        voice="shimmer",
    )
    print(f"Transcript: {transcript}")
    
    # Example 3: Text-only output
    print("\nExample 3: Text-only output...")
    transcript, audio = await client.send_message(
        user_text="Count from 1 to 5.",
        output_modalities=["text"],
    )
    print(f"Transcript: {transcript}")
    print(f"Audio size: {len(audio)} bytes (should be 0)")


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())