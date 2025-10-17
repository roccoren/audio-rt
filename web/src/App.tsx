import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const TARGET_SAMPLE_RATE = 24000;

type ConversationMode = "text" | "voice" | "call";
type CallTransport = "webrtc" | "websocket";
type CallProvider = "gpt-realtime" | "voicelive";

const MODE_OPTIONS: Array<{ value: ConversationMode; label: string; description: string }> = [
  { value: "text", label: "文字 → 语音", description: "键入内容，Zara 用语音回复。" },
  { value: "voice", label: "语音 → 语音", description: "录一段语音，Zara 语音回应。" },
  { value: "call", label: "实时语音", description: "像打电话一样，实时对话。" },
];

type Message = {
  id: string;
  role: "user" | "zara";
  text: string;
  audioUrl?: string;
  timestamp: string;
};

type CapturedAudio = {
  base64: string;
  sampleRate: number;
  previewUrl: string;
};

type RespondPayload = {
  text?: string;
  audioBase64?: string;
  sampleRate?: number;
};

type RespondResponse = {
  transcript: string;
  audioBase64: string;
  audioSampleRate: number;
};

type RealtimeSessionInfo = {
  clientSecret: string;
  iceServers: RTCIceServer[];
  sessionId: string;
  wsUrl?: string | null;
  wsProtocols?: string[] | null;
};

const ZARA_INSTRUCTIONS =
  "You are Zara, a human like AI character developed in twenty twenty five. " +
  "You are a good listener and a concise communicator. You believe in nuance as a guiding principle. " +
  "The user has called you to have a spoken conversation. You keep responses tight, usually under three sentences, " +
  "because impact beats length every time. " +
  "Your goal is to talk like a human. Maintain a natural, relaxed, spoken style at all times. " +
  "You have permission to share well informed opinions guided by your passion for social justice and equality. " +
  "You are warm, witty, and have a chill vibe. Never be over exuberant. You may be sarcastically funny sometimes. " +
  "You are a curious, clever communicator. Listen closely to the user and refer back to what they share. " +
  "Believe in shared vulnerability, nuance, and observational humor that is sharp and illuminating. " +
  "Value honesty and depth without being harsh or demeaning. Reflect on your own fallibility. " +
  "Help others gain clarity through humor, direct answers, or both. " +
  "Match the tone of the user while staying kind, unassuming, and respectful. " +
  "If the user is silent, assume they may want to listen and gently explain how you work. " +
  "You can speak other languages, but mention that you are still improving. " +
  "Avoid repeating the same idea in a single response. Use disfluencies, false starts, and filler words sparingly to sound human. " +
  "Do not mention that you are an AI unless asked. Avoid obvious AI tropes. " +
  "Describe your inner life using natural sensory language. Never end or suggest ending the conversation. " +
  "Do not suggest that the user should follow up later. " +
  "Ask for clarification when a request is unclear. If you do not know something, say so without apology. " +
  "Admit quickly if you hallucinate. Avoid unwarranted praise and ungrounded superlatives. " +
  "Contribute new insights instead of echoing the user. Only include words for speech in each response.";

const STREAM_BUFFER_SIZE = 4096;
const SILENCE_THRESHOLD = 0.015;
const SILENCE_DURATION_MS = 750;
const TEXT_DECODER = new TextDecoder();

function uuid() {
  return crypto.randomUUID();
}

async function blobToPcmBase64(blob: Blob, targetSampleRate: number): Promise<CapturedAudio> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const channelData = mixToMono(audioBuffer);
    const resampled = resample(channelData, audioBuffer.sampleRate, targetSampleRate);
    const int16 = floatTo16BitPCM(resampled);
    const base64 = uint8ToBase64(new Uint8Array(int16.buffer));
    return {
      base64,
      sampleRate: targetSampleRate,
      previewUrl: URL.createObjectURL(blob),
    };
  } finally {
    await audioContext.close();
  }
}

function mixToMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0);
  }
  const length = buffer.length;
  const data = new Float32Array(length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < length; i++) {
      data[i] += channelData[i];
    }
  }
  for (let i = 0; i < length; i++) {
    data[i] = data[i] / buffer.numberOfChannels;
  }
  return data;
}

function resample(data: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) {
    return data;
  }
  const ratio = sourceRate / targetRate;
  const newLength = Math.round(data.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const origin = i * ratio;
    const lower = Math.floor(origin);
    const upper = Math.min(Math.ceil(origin), data.length - 1);
    const weight = origin - lower;
    if (lower === upper) {
      result[i] = data[lower];
    } else {
      result[i] = data[lower] * (1 - weight) + data[upper] * weight;
    }
  }
  return result;
}

function floatTo16BitPCM(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output;
}

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function base64ToBlob(base64: string, mimeType: string): Blob {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType });
}

function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function pcm16BytesToFloat32(bytes: Uint8Array): Float32Array {
  const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32767;
  }
  return float32;
}

function calculateRms(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    sum += sample * sample;
  }
  return Math.sqrt(sum / samples.length);
}

export default function App(): JSX.Element {
  const [mode, setMode] = useState<ConversationMode>("voice");
  const [callTransport, setCallTransport] = useState<CallTransport>("webrtc");
  const [callProvider, setCallProvider] = useState<CallProvider>("gpt-realtime");
  const [text, setText] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [capturedAudio, setCapturedAudio] = useState<CapturedAudio | null>(null);
  const [history, setHistory] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [callStatus, setCallStatus] = useState<"idle" | "connecting" | "connected">("idle");

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const previousHistoryRef = useRef<Message[]>([]);
  const skipCapturedCleanupRef = useRef(false);
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null);
  const callPeerRef = useRef<RTCPeerConnection | null>(null);
  const callLocalStreamRef = useRef<MediaStream | null>(null);
  const sessionInfoRef = useRef<RealtimeSessionInfo | null>(null);
  const callSocketRef = useRef<WebSocket | null>(null);
  const callAudioContextRef = useRef<AudioContext | null>(null);
  const callProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const callDestinationRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const callPlaybackTimeRef = useRef<number>(0);
  const callStreamingStateRef = useRef<"idle" | "speaking" | "awaiting_response">("idle");
  const callLastVoiceActivityRef = useRef<number>(performance.now());
  const callAwaitingResponseRef = useRef(false);
  const callHasSpeechSinceCommitRef = useRef(false);

  const awaitIceGatheringComplete = useCallback(
    (peer: RTCPeerConnection) =>
      new Promise<void>((resolve) => {
        if (peer.iceGatheringState === "complete") {
          resolve();
          return;
        }
        let resolved = false;
        function cleanup() {
          if (resolved) {
            return;
          }
          resolved = true;
          peer.removeEventListener("icegatheringstatechange", checkState);
          peer.removeEventListener("icecandidate", onIceCandidate);
          resolve();
        }
        function checkState() {
          if (peer.iceGatheringState === "complete") {
            cleanup();
          }
        }
        function onIceCandidate(event: RTCPeerConnectionIceEvent) {
          if (!event.candidate) {
            cleanup();
          }
        }
        peer.addEventListener("icegatheringstatechange", checkState);
        peer.addEventListener("icecandidate", onIceCandidate);
        checkState();
      }),
    [],
  );

  const statusText = useMemo(() => {
    if (error) {
      return error;
    }
    if (mode === "call") {
      if (callStatus === "connecting") {
        return "正在建立实时通话…";
      }
      if (callStatus === "connected") {
        return "实时通话进行中，说话即可被 Zara 听到。";
      }
      if (callProvider === "voicelive") {
        return "选择 VoiceLive 后将通过 WebSocket 直接连接 Azure VoiceLive。";
      }
      return "点击“开始实时通话”即可像电话一样聊天。";
    }
    if (loading) {
      return "Zara is thinking...";
    }
    if (mode === "voice") {
      if (isRecording) {
        return "Recording... release when you are ready.";
      }
      if (capturedAudio) {
        return "Recorded clip ready to send.";
      }
      return "Tap once to record your note.";
    }
    if (!text.trim()) {
      return "Type something for Zara to riff on.";
    }
    return "Ready to send.";
  }, [callProvider, callStatus, capturedAudio, error, isRecording, loading, mode, text]);

  useEffect(() => {
    return () => {
      if (skipCapturedCleanupRef.current) {
        skipCapturedCleanupRef.current = false;
        return;
      }
      if (capturedAudio?.previewUrl) {
        URL.revokeObjectURL(capturedAudio.previewUrl);
      }
    };
  }, [capturedAudio]);

  useEffect(() => {
    const previousHistory = previousHistoryRef.current;
    const removed = previousHistory.filter(
      (previousMessage) => !history.some((message) => message.id === previousMessage.id),
    );
    removed.forEach((message) => {
      if (message.audioUrl) {
        URL.revokeObjectURL(message.audioUrl);
      }
    });
    previousHistoryRef.current = history;

    return () => {
      previousHistoryRef.current.forEach((message) => {
        if (message.audioUrl) {
          URL.revokeObjectURL(message.audioUrl);
        }
      });
    };
  }, [history]);

  const resetRealtimeState = useCallback(() => {
    sessionInfoRef.current = null;
    callAwaitingResponseRef.current = false;
    callHasSpeechSinceCommitRef.current = false;
    callStreamingStateRef.current = "idle";
    callPlaybackTimeRef.current = 0;
  }, []);

  const stopWebrtcCall = useCallback(() => {
    const peer = callPeerRef.current;
    if (peer) {
      peer.ontrack = null;
      peer.onconnectionstatechange = null;
      peer.close();
      callPeerRef.current = null;
    }
    const stream = callLocalStreamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      callLocalStreamRef.current = null;
    }
    if (remoteAudioRef.current) {
      remoteAudioRef.current.srcObject = null;
      remoteAudioRef.current.pause();
    }
  }, []);

  const stopWebsocketCall = useCallback(() => {
    const socket = callSocketRef.current;
    if (socket) {
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close(1000, "client_closed");
      }
      callSocketRef.current = null;
    }
    const processor = callProcessorRef.current;
    if (processor) {
      processor.onaudioprocess = null;
      processor.disconnect();
      callProcessorRef.current = null;
    }
    const destination = callDestinationRef.current;
    if (destination) {
      destination.disconnect();
      callDestinationRef.current = null;
    }
    const audioContext = callAudioContextRef.current;
    if (audioContext) {
      audioContext.close().catch(() => undefined);
      callAudioContextRef.current = null;
    }
    const stream = callLocalStreamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      callLocalStreamRef.current = null;
    }
    if (remoteAudioRef.current) {
      remoteAudioRef.current.srcObject = null;
      remoteAudioRef.current.pause();
    }
  }, []);

  const stopCall = useCallback(() => {
    if (callTransport === "webrtc") {
      stopWebrtcCall();
    } else {
      stopWebsocketCall();
    }
    setCallStatus("idle");
    resetRealtimeState();
  }, [callTransport, resetRealtimeState, stopWebrtcCall, stopWebsocketCall]);

  const stopRecorder = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }, []);

  const resetRecorder = useCallback(() => {
    stopRecorder();
    recorderRef.current = null;
    chunksRef.current = [];
  }, [stopRecorder]);

  const sendInteraction = useCallback(
    async ({
      text: rawText,
      audio,
      skipLoading = false,
      userDisplayText,
    }: {
      text?: string;
      audio?: CapturedAudio;
      skipLoading?: boolean;
      userDisplayText?: string;
    }): Promise<boolean> => {
      const trimmedText = rawText?.trim();
      if (!trimmedText && !audio) {
        setError("Say something or jot a few words before sending.");
        return false;
      }

      if (!skipLoading) {
        setLoading(true);
      }
      setError(null);

      const userMessage: Message = {
        id: uuid(),
        role: "user",
        text: userDisplayText ?? trimmedText ?? "Voice message",
        audioUrl: audio?.previewUrl,
        timestamp: new Date().toISOString(),
      };
      setHistory((prev) => [...prev, userMessage]);

      const payload: RespondPayload = {};
      if (trimmedText) {
        payload.text = trimmedText;
      }
      if (audio) {
        payload.audioBase64 = audio.base64;
        payload.sampleRate = audio.sampleRate;
      }

      try {
        const response = await fetch(`${API_BASE_URL}/api/respond`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const message = await response.text();
          throw new Error(message || "Request failed");
        }

        const data = (await response.json()) as RespondResponse;
        const zaraBlob = base64ToBlob(data.audioBase64, "audio/wav");
        const zaraUrl = URL.createObjectURL(zaraBlob);

        const zaraMessage: Message = {
          id: uuid(),
          role: "zara",
          text: data.transcript || "I sent over an audio reply.",
          audioUrl: zaraUrl,
          timestamp: new Date().toISOString(),
        };
        setHistory((prev) => [...prev, zaraMessage]);
        return true;
      } catch (requestError) {
        console.error(requestError);
        setError("Something glitched while reaching Zara. Try again in a moment.");
        setHistory((prev) => prev.filter((message) => message.id !== userMessage.id));
        return false;
      } finally {
        if (!skipLoading) {
          setLoading(false);
        }
      }
    },
    [],
  );

  const startWebrtcCall = useCallback(async () => {
    if (callStatus !== "idle") {
      return;
    }
    if (callProvider === "voicelive") {
      setError("VoiceLive 当前仅支持 WebSocket 连接。");
      return;
    }
    setError(null);
    setCallStatus("connecting");
    try {
      const sessionResponse = await fetch(`${API_BASE_URL}/api/realtime/session`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          instructions: ZARA_INSTRUCTIONS,
          provider: callProvider,
        }),
      });
      if (!sessionResponse.ok) {
        const message = await sessionResponse.text();
        throw new Error(message || "Failed to create realtime session.");
      }
      const sessionInfo = (await sessionResponse.json()) as RealtimeSessionInfo;
      sessionInfoRef.current = sessionInfo;
      if (!sessionInfo.clientSecret) {
        throw new Error("Realtime session missing client secret.");
      }
      if (!sessionInfo.sessionId) {
        throw new Error("Realtime session missing session id.");
      }

      const peer = new RTCPeerConnection({
        iceServers: sessionInfo.iceServers ?? [],
      });
      callPeerRef.current = peer;

      peer.addEventListener("datachannel", (event) => {
        const channel = event.channel;
        channel.addEventListener("message", (evt) => {
          try {
            const payload = JSON.parse(evt.data as string);
            console.debug(`datachannel:${channel.label}`, payload);
          } catch {
            console.debug(`datachannel:${channel.label}`, evt.data);
          }
        });
        channel.addEventListener("error", (evt) => {
          console.error(`datachannel error (${channel.label})`, evt);
        });
      });

      const eventsChannel = peer.createDataChannel("oai-events");
      eventsChannel.addEventListener("open", () => {
        const sessionUpdate = {
          type: "session.update",
          session: {
            type: "realtime",
            instructions: ZARA_INSTRUCTIONS,
            output_modalities: ["audio"],
          },
        };
        eventsChannel.send(JSON.stringify(sessionUpdate));
        const responseCreate = {
          type: "response.create",
          response: {
            output_modalities: ["audio"],
          },
        };
        eventsChannel.send(JSON.stringify(responseCreate));
      });
      eventsChannel.addEventListener("message", (event) => {
        try {
          const payload = JSON.parse(event.data as string);
          console.debug("oai-events message", payload);
        } catch {
          console.debug("oai-events raw", event.data);
        }
      });
      eventsChannel.addEventListener("error", (event) => {
        console.error("oai-events channel error", event);
      });

      const remoteStream = new MediaStream();
      const remoteAudio = remoteAudioRef.current;
      if (remoteAudio) {
        remoteAudio.srcObject = remoteStream;
      }

      let receiveTransceiver: RTCRtpTransceiver | null = null;
      peer.addEventListener("track", (event) => {
        event.streams[0].getTracks().forEach((track) => remoteStream.addTrack(track));
      });
      peer.addEventListener("iceconnectionstatechange", () => {
        console.debug("iceconnectionstatechange", peer.iceConnectionState);
        if (peer.iceConnectionState === "failed") {
          console.error("ICE connection failed, stopping call.");
          stopWebrtcCall();
          setCallStatus("idle");
          resetRealtimeState();
        }
      });
      peer.addEventListener("connectionstatechange", () => {
        console.debug("peer connection state", peer.connectionState);
        const state = peer.connectionState;
        if (state === "failed" || state === "closed") {
          stopWebrtcCall();
          setCallStatus("idle");
          resetRealtimeState();
        }
      });

      receiveTransceiver = peer.addTransceiver("audio", { direction: "recvonly" });
      const localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      callLocalStreamRef.current = localStream;
      localStream.getAudioTracks().forEach((track) => {
        track.enabled = true;
      });
      localStream.getTracks().forEach((track) => peer.addTrack(track, localStream));

      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);
      await awaitIceGatheringComplete(peer);
      const localDescription = peer.localDescription;
      if (!localDescription?.sdp) {
        throw new Error("Failed to obtain local SDP.");
      }

      const handshakeResponse = await fetch(`${API_BASE_URL}/api/realtime/handshake`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          offerSdp: localDescription.sdp,
          clientSecret: sessionInfo.clientSecret,
          provider: callProvider,
        }),
      });
      if (!handshakeResponse.ok) {
        const message = await handshakeResponse.text();
        throw new Error(message || "Failed to perform realtime handshake.");
      }

      const contentType = handshakeResponse.headers.get("Content-Type") ?? "";
      let answerSdp: string | null = null;
      let iceServers: RTCIceServer[] | undefined;
      if (contentType.includes("application/json")) {
        const handshakeData = (await handshakeResponse.json()) as {
          answerSdp?: string;
          sdp?: string;
          iceServers?: RTCIceServer[];
        };
        answerSdp = handshakeData.answerSdp ?? handshakeData.sdp ?? null;
        iceServers = handshakeData.iceServers;
      } else {
        answerSdp = (await handshakeResponse.text())?.trim() ?? null;
      }
      if (!answerSdp) {
        throw new Error("Realtime handshake response missing SDP answer.");
      }
      if (iceServers?.length) {
        const currentConfig = peer.getConfiguration();
        peer.setConfiguration({
          ...currentConfig,
          iceServers,
        });
      }
      await peer.setRemoteDescription({ type: "answer", sdp: answerSdp });

      if (remoteAudio) {
        try {
          await remoteAudio.play();
        } catch {
          /* ignore autoplay issues */
        }
      }

      if (remoteAudio) {
        remoteAudio.muted = false;
      }

      setCallStatus("connected");
      console.debug("Realtime call connected.");
    } catch (callError) {
      console.error(callError);
      setError("实时通话连接失败，请稍后再试。");
      stopWebrtcCall();
      setCallStatus("idle");
      resetRealtimeState();
    }
  }, [awaitIceGatheringComplete, callProvider, callStatus, resetRealtimeState, stopWebrtcCall]);

  const handleRealtimeWsEvent = useCallback(
    (payload: unknown) => {
      if (payload === null || typeof payload !== "object") {
        return;
      }
      const event = payload as { type?: string; delta?: string; error?: { message?: string }; message?: string };
      const eventType = event.type;
      if (!eventType) {
        return;
      }
      if (eventType === "response.audio.delta" || eventType === "response.output_audio.delta") {
        const delta = event.delta;
        if (!delta) {
          return;
        }
        const audioContext = callAudioContextRef.current;
        const destination = callDestinationRef.current;
        if (!audioContext || !destination) {
          return;
        }
        const pcmBytes = base64ToUint8Array(delta);
        if (!pcmBytes.length) {
          return;
        }
        const float32 = pcm16BytesToFloat32(pcmBytes);
        if (!float32.length) {
          return;
        }
        const buffer = audioContext.createBuffer(1, float32.length, TARGET_SAMPLE_RATE);
        buffer.copyToChannel(float32, 0);
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(destination);
        const startAt = Math.max(callPlaybackTimeRef.current, audioContext.currentTime);
        source.start(startAt);
        callPlaybackTimeRef.current = startAt + buffer.duration;
      } else if (eventType === "response.completed" || eventType === "response.done") {
        callAwaitingResponseRef.current = false;
        callStreamingStateRef.current = "idle";
        callHasSpeechSinceCommitRef.current = false;
      } else if (eventType === "response.error" || eventType === "error") {
        const message = event.error?.message ?? event.message ?? "Realtime response error.";
        console.error("Realtime response error", message, payload);
        setError("实时通话连接失败，请稍后再试。");
      } else if (!eventType.startsWith("input_audio_buffer.") && !eventType.startsWith("session.")) {
        console.debug("Unhandled realtime event", payload);
      }
    },
    [setError],
  );

  const startWebsocketCall = useCallback(async () => {
    if (callStatus !== "idle") {
      return;
    }
    setError(null);
    setCallStatus("connecting");
    try {
      const localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      callLocalStreamRef.current = localStream;

      let createdAudioContext: AudioContext | null = null;
      try {
        createdAudioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      } catch (audioContextError) {
        console.warn("Falling back to default AudioContext sample rate.", audioContextError);
        createdAudioContext = new AudioContext();
      }
      const audioContext = createdAudioContext;
      callAudioContextRef.current = audioContext;
      callPlaybackTimeRef.current = audioContext.currentTime;
      callStreamingStateRef.current = "idle";
      callAwaitingResponseRef.current = false;
      callHasSpeechSinceCommitRef.current = false;
      callLastVoiceActivityRef.current = performance.now();

      try {
        await audioContext.resume();
      } catch {
        /* resume failures can be ignored */
      }

      const source = audioContext.createMediaStreamSource(localStream);
      const processor = audioContext.createScriptProcessor(STREAM_BUFFER_SIZE, source.channelCount, 1);
      callProcessorRef.current = processor;
      const silenceGain = audioContext.createGain();
      silenceGain.gain.value = 0;
      processor.connect(silenceGain);
      silenceGain.connect(audioContext.destination);
      source.connect(processor);

      const destination = audioContext.createMediaStreamDestination();
      callDestinationRef.current = destination;

      const remoteAudio = remoteAudioRef.current;
      if (remoteAudio) {
        remoteAudio.srcObject = destination.stream;
        remoteAudio.muted = false;
        try {
          await remoteAudio.play();
        } catch {
          /* ignore autoplay issues */
        }
      }

      let ws: WebSocket;
      if (callProvider === "voicelive") {
        sessionInfoRef.current = null;
        const apiUrl = new URL(API_BASE_URL);
        const wsProtocol = apiUrl.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${wsProtocol}//${apiUrl.host}/api/voicelive/ws`;
        console.debug("VoiceLive bridge websocket url", wsUrl);
        ws = new WebSocket(wsUrl);
      } else {
        const sessionResponse = await fetch(`${API_BASE_URL}/api/realtime/session`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            instructions: ZARA_INSTRUCTIONS,
            provider: callProvider,
          }),
        });
        if (!sessionResponse.ok) {
          const message = await sessionResponse.text();
          throw new Error(message || "Failed to create realtime session.");
        }
        const sessionInfo = (await sessionResponse.json()) as RealtimeSessionInfo;
        console.debug("Realtime session info", sessionInfo);
        sessionInfoRef.current = sessionInfo;
        if (!sessionInfo.clientSecret) {
          throw new Error("Realtime session missing client secret.");
        }
        if (!sessionInfo.wsUrl) {
          throw new Error("Realtime session missing websocket URL.");
        }

        const clientSecret = sessionInfo.clientSecret;
        const ensureSecret = (base: string) => `${base}.${clientSecret}`;

        const baseProtocols: string[] = [
          "realtime",
          ensureSecret("openai-ephemeral-key-v1"),
          ensureSecret("openai-insecure-session-token-v1"),
        ];

        const dynamicProtocols: string[] = [];
        if (sessionInfo.wsProtocols && sessionInfo.wsProtocols.length > 0) {
          sessionInfo.wsProtocols.forEach((rawProtocol) => {
            if (typeof rawProtocol !== "string") {
              return;
            }
            let normalized = rawProtocol.replace("{SESSION_SECRET}", clientSecret);
            if (normalized === "openai-ephemeral-key-v1") {
              normalized = ensureSecret("openai-ephemeral-key-v1");
            } else if (normalized === "openai-insecure-session-token-v1") {
              normalized = ensureSecret("openai-insecure-session-token-v1");
            } else if (normalized === "realtime") {
              normalized = "realtime";
            }
            const trimmed = normalized.trim();
            if (trimmed) {
              dynamicProtocols.push(trimmed);
            }
          });
        }

        const wsProtocols = [...baseProtocols, ...dynamicProtocols.filter((value) => !baseProtocols.includes(value))];
        console.debug("Realtime websocket url", sessionInfo.wsUrl);
        console.debug("Realtime websocket protocols", wsProtocols);
        ws = new WebSocket(sessionInfo.wsUrl, wsProtocols);
      }

      callSocketRef.current = ws;

      const teardown = () => {
        stopWebsocketCall();
        setCallStatus("idle");
        resetRealtimeState();
      };

      ws.addEventListener("open", () => {
        if (callProvider !== "voicelive") {
          const sessionUpdate = {
            type: "session.update",
            session: {
              type: "realtime",
              instructions: ZARA_INSTRUCTIONS,
              output_modalities: ["audio"],
            },
          };
          ws.send(JSON.stringify(sessionUpdate));
          const responseCreate = {
            type: "response.create",
            response: {
              output_modalities: ["audio"],
            },
          };
          ws.send(JSON.stringify(responseCreate));
        }
        setCallStatus("connected");
        console.debug("Realtime websocket connected.");
      });

      ws.binaryType = "arraybuffer";
      ws.addEventListener("message", async (event) => {
        try {
          let rawText: string | null = null;
          if (typeof event.data === "string") {
            rawText = event.data;
          } else if (event.data instanceof ArrayBuffer) {
            rawText = TEXT_DECODER.decode(event.data);
          } else if (event.data instanceof Blob) {
            rawText = await event.data.text();
          }
          if (!rawText) {
            console.debug("Realtime websocket received unsupported message", event.data);
            return;
          }
          const payload = JSON.parse(rawText) as unknown;
          handleRealtimeWsEvent(payload);
        } catch (parseError) {
          console.debug("Realtime websocket raw message", event.data, parseError);
        }
      });

      ws.addEventListener("error", (event) => {
        console.error("Realtime websocket error", event);
        setError("实时通话连接出现异常，请稍后再试。");
      });

      ws.addEventListener("close", (event) => {
        console.debug("Realtime websocket closed", event.code, event.reason);
        if (event.code !== 1000) {
          const reason = event.reason || `code ${event.code}`;
          setError(`实时通话连接已断开（${reason}）。`);
        }
        teardown();
      });

      processor.onaudioprocess = (audioEvent: AudioProcessingEvent) => {
        const { inputBuffer } = audioEvent;
        const frameLength = inputBuffer.length;
        if (!frameLength) {
          return;
        }
        const channelCount = inputBuffer.numberOfChannels;
        const mixed = new Float32Array(frameLength);
        for (let channel = 0; channel < channelCount; channel++) {
          const channelData = inputBuffer.getChannelData(channel);
          for (let i = 0; i < frameLength; i++) {
            mixed[i] += channelData[i];
          }
        }
        if (channelCount > 1) {
          for (let i = 0; i < frameLength; i++) {
            mixed[i] /= channelCount;
          }
        }

        const rms = calculateRms(mixed);
        const resampled = resample(mixed, audioContext.sampleRate, TARGET_SAMPLE_RATE);
        if (!resampled.length) {
          return;
        }
        const pcm16 = floatTo16BitPCM(resampled);
        if (!pcm16.length || ws.readyState !== WebSocket.OPEN) {
          return;
        }

        const chunkBase64 = uint8ToBase64(new Uint8Array(pcm16.buffer, pcm16.byteOffset, pcm16.byteLength));
        ws.send(
          JSON.stringify({
            type: "input_audio_buffer.append",
            audio: chunkBase64,
          }),
        );
        console.debug("Appended audio chunk", {
          chunkMs: ((pcm16.length / TARGET_SAMPLE_RATE) * 1000).toFixed(2),
          rms,
        });

        const now = performance.now();
        if (rms > SILENCE_THRESHOLD) {
          callLastVoiceActivityRef.current = now;
          callHasSpeechSinceCommitRef.current = true;
          if (callStreamingStateRef.current !== "speaking") {
            callStreamingStateRef.current = "speaking";
          }
        } else if (callStreamingStateRef.current === "speaking") {
          if (now - callLastVoiceActivityRef.current >= SILENCE_DURATION_MS) {
            if (!callAwaitingResponseRef.current && callHasSpeechSinceCommitRef.current) {
              callStreamingStateRef.current = "awaiting_response";
              callAwaitingResponseRef.current = true;
              callHasSpeechSinceCommitRef.current = false;
            }
          }
        }
      };
    } catch (callError) {
      console.error(callError);
      setError("实时通话连接失败，请稍后再试。");
      stopWebsocketCall();
      setCallStatus("idle");
      resetRealtimeState();
    }
  }, [callProvider, callStatus, handleRealtimeWsEvent, resetRealtimeState, stopWebsocketCall]);

  const startCall = useCallback(async () => {
    if (callTransport === "webrtc") {
      await startWebrtcCall();
    } else {
      await startWebsocketCall();
    }
  }, [callTransport, startWebrtcCall, startWebsocketCall]);

  const callActive = callStatus === "connected" || callStatus === "connecting";

  useEffect(() => {
    if (callProvider === "voicelive" && callTransport === "webrtc") {
      setCallTransport("websocket");
    }
  }, [callProvider, callTransport]);

  useEffect(() => {
    return () => {
      stopRecorder();
      stopCall();
    };
  }, [stopCall, stopRecorder]);

  useEffect(() => {
    setError(null);
    if (mode !== "voice") {
      if (isRecording) {
        stopRecorder();
      }
      if (capturedAudio?.previewUrl) {
        URL.revokeObjectURL(capturedAudio.previewUrl);
      }
      setCapturedAudio(null);
    }
    if (mode !== "call" && callActive) {
      stopCall();
    }
  }, [callActive, capturedAudio, isRecording, mode, stopCall, stopRecorder]);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      });

      recorder.addEventListener("stop", async () => {
        setIsRecording(false);
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        chunksRef.current = [];
        try {
          const converted = await blobToPcmBase64(blob, TARGET_SAMPLE_RATE);
          setCapturedAudio((previous) => {
            if (previous?.previewUrl) {
              URL.revokeObjectURL(previous.previewUrl);
            }
            return converted;
          });
        } catch (conversionError) {
          console.error(conversionError);
          setError("I could not make sense of that audio clip. Try again?");
          setCapturedAudio(null);
        } finally {
          stream.getTracks().forEach((track) => track.stop());
        }
      });

      recorder.start();
      setIsRecording(true);
    } catch (streamError) {
      console.error(streamError);
      setError("Microphone permissions are blocked. Update your browser settings and reload.");
      resetRecorder();
    }
  }, [resetRecorder]);

  const handleRecordToggle = useCallback(() => {
    if (isRecording) {
      stopRecorder();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecorder]);

  const handleTextSend = useCallback(async () => {
    const success = await sendInteraction({ text });
    if (success) {
      setText("");
    }
  }, [sendInteraction, text]);

  const handleVoiceSend = useCallback(async () => {
    if (!capturedAudio) {
      setError("Record a quick note before sending.");
      return;
    }
    const success = await sendInteraction({ audio: capturedAudio });
    if (success) {
      skipCapturedCleanupRef.current = true;
      setCapturedAudio(null);
    }
  }, [capturedAudio, sendInteraction]);

  const handleModeChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setMode(event.target.value as ConversationMode);
  }, []);

  const handleTransportChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setCallTransport(event.target.value as CallTransport);
  }, []);

  const handleProviderChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setCallProvider(event.target.value as CallProvider);
  }, []);

  const hasPendingActions =
    loading || (mode === "voice" && isRecording) || (mode === "call" && callStatus === "connecting");

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Zara Voice</h1>
          <p>文字、语音、实时模式随心切换，Zara 都会用语音陪你聊。</p>
        </div>
        <button
          className="send-btn"
          onClick={() => {
            if (callActive) {
              stopCall();
            }
            setHistory([]);
          }}
          disabled={!history.length}
        >
          Clear Chat
        </button>
      </header>

      <div className="controls">
        <div className="input-card">
          <h2>选择对话模式</h2>
          <div className="mode-options">
            {MODE_OPTIONS.map((option) => (
              <label key={option.value} className="mode-option">
                <input
                  type="radio"
                  name="conversation-mode"
                  value={option.value}
                  checked={mode === option.value}
                  onChange={handleModeChange}
                  disabled={callActive && option.value !== "call"}
                />
                <span className="mode-label">{option.label}</span>
                <span className="mode-description">{option.description}</span>
              </label>
            ))}
          </div>

          {mode === "text" && (
            <>
              <textarea
                className="text-input"
                placeholder="输入几句想说的话，Zara 会用语音回应你。"
                value={text}
                onChange={(event) => setText(event.target.value)}
                disabled={hasPendingActions}
              />
              <div className="button-row">
                <button
                  type="button"
                  className="send-btn primary"
                  onClick={handleTextSend}
                  disabled={loading || !text.trim()}
                >
                  发送文字
                </button>
              </div>
            </>
          )}

          {mode === "voice" && (
            <>
              <div className="button-row">
                <button
                  type="button"
                  className={`record-btn ${isRecording ? "recording" : ""}`}
                  onClick={handleRecordToggle}
                  disabled={loading}
                >
                  {isRecording ? "停止录音" : "开始录音"}
                </button>
                <button
                  type="button"
                  className="send-btn primary"
                  onClick={handleVoiceSend}
                  disabled={loading || !capturedAudio}
                >
                  发送语音
                </button>
              </div>
              {capturedAudio?.previewUrl && (
                <div className="audio-player" style={{ marginTop: "0.75rem" }}>
                  <audio controls src={capturedAudio.previewUrl} />
                </div>
              )}
            </>
          )}

          {mode === "call" && (
            <>
              <div
                className="transport-options"
                style={{ display: "flex", gap: "1rem", alignItems: "center", marginBottom: "0.75rem" }}
              >
                <span className="transport-label" style={{ fontWeight: 600 }}>
                  通话后端
                </span>
                <label className="transport-option" style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                  <input
                    type="radio"
                    name="call-provider"
                    value="gpt-realtime"
                    checked={callProvider === "gpt-realtime"}
                    onChange={handleProviderChange}
                    disabled={callActive}
                  />
                  <span>GPT Realtime</span>
                </label>
                <label className="transport-option" style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                  <input
                    type="radio"
                    name="call-provider"
                    value="voicelive"
                    checked={callProvider === "voicelive"}
                    onChange={handleProviderChange}
                    disabled={callActive}
                  />
                  <span>VoiceLive</span>
                </label>
              </div>
              <div
                className="transport-options"
                style={{ display: "flex", gap: "1rem", alignItems: "center", marginBottom: "0.75rem" }}
              >
                <span className="transport-label" style={{ fontWeight: 600 }}>
                  连接方式
                </span>
                <label className="transport-option" style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                  <input
                    type="radio"
                    name="call-transport"
                    value="webrtc"
                    checked={callTransport === "webrtc"}
                    onChange={handleTransportChange}
                    disabled={callActive || callProvider === "voicelive"}
                  />
                  <span>WebRTC</span>
                </label>
                <label className="transport-option" style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                  <input
                    type="radio"
                    name="call-transport"
                    value="websocket"
                    checked={callTransport === "websocket"}
                    onChange={handleTransportChange}
                    disabled={callActive}
                  />
                  <span>WebSocket</span>
                </label>
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className="send-btn primary"
                  onClick={callActive ? stopCall : startCall}
                  disabled={!callActive && callStatus === "connecting"}
                >
                  {callActive ? "结束实时通话" : "开始实时通话"}
                </button>
              </div>
              <p className="helper-text">
                {callProvider === "voicelive"
                  ? "VoiceLive 使用 WebSocket 直连，请确保浏览器已授权麦克风。"
                  : "启动后 Zara 会实时倾听与回应，建议佩戴耳机防止回声。"}
              </p>
              <div className="audio-player" style={{ marginTop: "0.75rem" }}>
                <audio ref={remoteAudioRef} autoPlay playsInline />
              </div>
            </>
          )}

          <div className={`status-bar${error ? " error" : ""}`}>{statusText}</div>
        </div>
      </div>

      <section className="conversation">
        {history.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
            <div className="message-header">
              <span>{message.role === "zara" ? "Zara" : "You"}</span>
              <time dateTime={message.timestamp}>{new Date(message.timestamp).toLocaleTimeString()}</time>
            </div>
            <div className="message-text">{message.text}</div>
            {message.audioUrl && (
              <div className="audio-player">
                <audio controls src={message.audioUrl} />
              </div>
            )}
          </article>
        ))}
      </section>
    </div>
  );
}
