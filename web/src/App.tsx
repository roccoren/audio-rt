import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const TARGET_SAMPLE_RATE = 24000;

type ConversationMode = "text" | "voice" | "call";
type CallTransport = "webrtc" | "websocket";
type CallProvider = "gpt-realtime" | "voicelive" | "live-interpreter";

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

type RealtimeResponseState = {
  audioChunks: Uint8Array[];
  textParts: string[];
  timestamp: string;
};

type TranslateResponse = {
  recognizedText: string;
  translations: Record<string, string>;
  audioBase64?: string | null;
  audioFormat?: string | null;
  audioSampleRate?: number | null;
  detectedSourceLanguage?: string | null;
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

function concatUint8Arrays(chunks: Uint8Array[]): Uint8Array {
  if (!chunks.length) {
    return new Uint8Array(0);
  }
  let total = 0;
  chunks.forEach((chunk) => {
    total += chunk.byteLength;
  });
  const combined = new Uint8Array(total);
  let offset = 0;
  chunks.forEach((chunk) => {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  });
  return combined;
}

function collectTextCandidates(value: unknown, parts: string[]): void {
  if (value === null || value === undefined) {
    return;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed) {
      parts.push(trimmed);
    }
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectTextCandidates(item, parts));
    return;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    for (const [key, nested] of entries) {
      if (nested === null || nested === undefined) {
        continue;
      }
      const lowered = key.toLowerCase();
      if (
        lowered.includes("text") ||
        lowered.includes("caption") ||
        lowered.includes("transcript") ||
        lowered.includes("message") ||
        lowered.includes("value") ||
        lowered.includes("content") ||
        lowered.endsWith("delta") ||
        lowered === "output"
      ) {
        collectTextCandidates(nested, parts);
      }
    }
  }
}

function appendTextDelta(delta: unknown, parts: string[]): void {
  collectTextCandidates(delta, parts);
}

function extractTextFromResponsePayload(payload: unknown): string[] {
  const segments: string[] = [];
  if (!payload || typeof payload !== "object") {
    return segments;
  }
  const container = payload as { output?: unknown; outputs?: unknown; content?: unknown; text?: unknown };

  const outputs = Array.isArray(container.output)
    ? container.output
    : Array.isArray(container.outputs)
      ? container.outputs
      : null;

  if (outputs) {
    outputs.forEach((output) => {
      if (!output || typeof output !== "object") {
        if (typeof output === "string") {
          segments.push(output);
        }
        return;
      }
      const contentItems = (output as { content?: unknown }).content;
      if (Array.isArray(contentItems)) {
        contentItems.forEach((item) => {
          if (item && typeof item === "object") {
            const itemType = (item as { type?: unknown }).type;
            const textValue = (item as { text?: unknown }).text;
            if ((itemType === "output_text" || itemType === "text") && typeof textValue === "string") {
              segments.push(textValue);
            } else if (typeof textValue === "string" && !itemType) {
              segments.push(textValue);
            }
          } else if (typeof item === "string") {
            segments.push(item);
          }
        });
      }
    });
  }

  const content = Array.isArray(container.content) ? container.content : null;
  if (content) {
    content.forEach((item) => appendTextDelta(item, segments));
  }

  const textValue = container.text;
  if (typeof textValue === "string" && textValue.trim()) {
    segments.push(textValue);
  }

  const outputText = (container as { output_text?: unknown; outputText?: unknown }).output_text;
  if (outputText !== undefined) {
    appendTextDelta(outputText, segments);
  }
  const camelOutputText = (container as { outputText?: unknown }).outputText;
  if (camelOutputText !== undefined) {
    appendTextDelta(camelOutputText, segments);
  }

  const messages = (container as { messages?: unknown }).messages;
  if (messages !== undefined) {
    appendTextDelta(messages, segments);
  }

  return segments;
}

function pcm16ChunksToWavBlob(chunks: Uint8Array[], sampleRate: number): Blob | null {
  if (!chunks.length) {
    return null;
  }
  let totalBytes = 0;
  chunks.forEach((chunk) => {
    totalBytes += chunk.byteLength;
  });
  if (!totalBytes) {
    return null;
  }

  const headerSize = 44;
  const buffer = new ArrayBuffer(headerSize + totalBytes);
  const view = new DataView(buffer);

  let offset = 0;
  const writeString = (value: string) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
    offset += value.length;
  };

  writeString("RIFF");
  view.setUint32(offset, 36 + totalBytes, true);
  offset += 4;
  writeString("WAVE");
  writeString("fmt ");
  view.setUint32(offset, 16, true);
  offset += 4;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  const byteRate = sampleRate * 2;
  view.setUint32(offset, byteRate, true);
  offset += 4;
  const blockAlign = 2;
  view.setUint16(offset, blockAlign, true);
  offset += 2;
  view.setUint16(offset, 16, true);
  offset += 2;
  writeString("data");
  view.setUint32(offset, totalBytes, true);
  offset += 4;

  const pcmData = new Uint8Array(buffer, headerSize, totalBytes);
  let dataOffset = 0;
  chunks.forEach((chunk) => {
    pcmData.set(chunk, dataOffset);
    dataOffset += chunk.byteLength;
  });

  return new Blob([buffer], { type: "audio/wav" });
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
  const [callTranscripts, setCallTranscripts] = useState<
    Record<string, { text: string; isFinal: boolean; timestamp: string }>
  >({});

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
  const callResponseStateRef = useRef<Map<string, RealtimeResponseState>>(new Map());
  const liveInterpreterPendingChunksRef = useRef<Uint8Array[]>([]);
  const liveInterpreterSendingRef = useRef(false);
  const liveInterpreterSequenceRef = useRef(0);

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
      if (callProvider === "live-interpreter") {
        return "Live Interpreter 将把你的语音翻译成目标语言并播报译文。";
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
    callResponseStateRef.current.clear();
    liveInterpreterPendingChunksRef.current = [];
    liveInterpreterSendingRef.current = false;
    liveInterpreterSequenceRef.current = 0;
    setCallTranscripts({});
  }, []);

  const ensureRealtimeResponseState = useCallback((responseId: string): RealtimeResponseState => {
    const store = callResponseStateRef.current;
    let state = store.get(responseId);
    if (!state) {
      state = {
        audioChunks: [],
        textParts: [],
        timestamp: new Date().toISOString(),
      };
      store.set(responseId, state);
    }
    return state;
  }, []);

  const getRealtimeResponseState = useCallback(
    (responseId: string | null | undefined): [string, RealtimeResponseState] => {
      const store = callResponseStateRef.current;
      const trimmed = (responseId || "").trim();
      if (trimmed) {
        let state = store.get(trimmed);
        if (!state) {
          const fallback = store.get("__default__");
          if (fallback) {
            store.delete("__default__");
            state = fallback;
            store.set(trimmed, state);
          } else {
            state = ensureRealtimeResponseState(trimmed);
          }
        }
        return [trimmed, state];
      }
      const key = "__default__";
      return [key, ensureRealtimeResponseState(key)];
    },
    [ensureRealtimeResponseState],
  );

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
      remoteAudioRef.current.src = "";
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
      remoteAudioRef.current.src = "";
    }
  }, []);

  const sendLiveInterpreterRequest = useCallback(async () => {
    if (callProvider !== "live-interpreter") {
      return;
    }
    if (liveInterpreterSendingRef.current) {
      return;
    }
    const pending = liveInterpreterPendingChunksRef.current;
    if (!pending.length) {
      callAwaitingResponseRef.current = false;
      callStreamingStateRef.current = "idle";
      callHasSpeechSinceCommitRef.current = false;
      return;
    }

    const chunks = pending.slice();
    liveInterpreterPendingChunksRef.current = [];
    const combined = concatUint8Arrays(chunks);
    if (!combined.length) {
      callAwaitingResponseRef.current = false;
      callStreamingStateRef.current = "idle";
      callHasSpeechSinceCommitRef.current = false;
      return;
    }

    liveInterpreterSendingRef.current = true;
    const payload = {
      audioBase64: uint8ToBase64(combined),
      sampleRate: TARGET_SAMPLE_RATE,
    };
    const timestamp = new Date().toISOString();

    try {
      const response = await fetch(`${API_BASE_URL}/api/translate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Translation request failed.");
      }
      const data = (await response.json()) as TranslateResponse;
      const translationEntries = Object.entries(data.translations ?? {}).filter(
        ([, text]) => typeof text === "string" && text.trim(),
      );
      const primaryTranslation = translationEntries.length > 0 ? translationEntries[0][1] : "";
      const displayText = primaryTranslation || data.recognizedText || "翻译完成。";

      let audioUrl: string | undefined;
      if (data.audioBase64) {
        try {
          const audioBlob = base64ToBlob(data.audioBase64, "audio/wav");
          audioUrl = URL.createObjectURL(audioBlob);
          const audioContext = callAudioContextRef.current;
          const destination = callDestinationRef.current;
          if (audioContext && destination) {
            const arrayBuffer = await audioBlob.arrayBuffer();
            const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
            const source = audioContext.createBufferSource();
            source.buffer = decoded;
            source.connect(destination);
            const startAt = Math.max(callPlaybackTimeRef.current, audioContext.currentTime);
            source.start(startAt);
            callPlaybackTimeRef.current = startAt + decoded.duration;
          } else {
            const fallbackAudio = new Audio(audioUrl);
            fallbackAudio.play().catch(() => undefined);
          }
        } catch (decodeError) {
          console.error("Live Interpreter audio playback failed", decodeError);
          if (audioUrl) {
            const fallbackAudio = new Audio(audioUrl);
            fallbackAudio.play().catch(() => undefined);
          }
        }
      }

      const message: Message = {
        id: uuid(),
        role: "zara",
        text: displayText,
        audioUrl,
        timestamp,
      };
      setHistory((prev) => [...prev, message]);

      const transcriptLines: string[] = [];
      if (data.detectedSourceLanguage && data.detectedSourceLanguage.trim()) {
        transcriptLines.push(`检测到语言：${data.detectedSourceLanguage}`);
      }
      if (data.recognizedText && data.recognizedText.trim()) {
        transcriptLines.push(`原文：${data.recognizedText}`);
      }
      translationEntries.forEach(([language, text]) => {
        transcriptLines.push(`译文（${language}）：${text}`);
      });
      const transcriptText = transcriptLines.join("\n") || displayText;

      setCallTranscripts((prev) => ({
        ...prev,
        [message.id]: {
          text: transcriptText,
          isFinal: true,
          timestamp,
        },
      }));
    } catch (translationError) {
      console.error("Live Interpreter translation failed", translationError);
      setError("Live Interpreter 翻译失败，请稍后重试。");
    } finally {
      callAwaitingResponseRef.current = false;
      callStreamingStateRef.current = "idle";
      callHasSpeechSinceCommitRef.current = false;
      callLastVoiceActivityRef.current = performance.now();
      liveInterpreterSendingRef.current = false;
    }
  }, [callProvider, setCallTranscripts, setError, setHistory]);

  const startLiveInterpreterCall = useCallback(async () => {
    if (callStatus !== "idle") {
      return;
    }
    if (callProvider !== "live-interpreter") {
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
      liveInterpreterPendingChunksRef.current = [];
      liveInterpreterSendingRef.current = false;
      liveInterpreterSequenceRef.current = 0;

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

      setCallStatus("connected");

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
        if (!pcm16.length) {
          return;
        }

        const chunkCopy = new Uint8Array(pcm16.buffer.slice(0));
        liveInterpreterPendingChunksRef.current.push(chunkCopy);

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
              void sendLiveInterpreterRequest();
            }
          }
        }
      };
    } catch (callError) {
      console.error(callError);
      setError("Live Interpreter 连接失败，请稍后再试。");
      stopWebsocketCall();
      setCallStatus("idle");
      resetRealtimeState();
    }
  }, [callProvider, callStatus, resetRealtimeState, sendLiveInterpreterRequest, setError, stopWebsocketCall]);

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
            instructions: ZARA_INSTRUCTIONS,
          },
        };
        eventsChannel.send(JSON.stringify(sessionUpdate));
      });
      eventsChannel.addEventListener("message", (event) => {
        try {
          const payload = JSON.parse(event.data as string);
          console.debug("oai-events message", payload);
          handleRealtimeWsEvent(payload);
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
      const event = payload as Record<string, unknown>;
      const rawType = event.type;
      if (typeof rawType !== "string" || !rawType) {
        return;
      }
      const eventType = rawType;

      const responseCandidate = event.response;
      const candidateIds: unknown[] = [event.response_id, event.responseId, event.responseID, event.id];
      if (responseCandidate && typeof responseCandidate === "object") {
        candidateIds.push((responseCandidate as { id?: unknown }).id);
        candidateIds.push((responseCandidate as { response_id?: unknown }).response_id);
        candidateIds.push((responseCandidate as { responseId?: unknown }).responseId);
      }
      let responseId: string | null = null;
      for (const candidate of candidateIds) {
        if (typeof candidate === "string" && candidate.trim()) {
          responseId = candidate;
          break;
        }
      }
      const [responseKey, responseState] = getRealtimeResponseState(responseId);

      if (eventType === "response.audio.delta" || eventType === "response.output_audio.delta") {
        const delta = event.delta;
        if (typeof delta !== "string" || !delta) {
          return;
        }
        const pcmBytes = base64ToUint8Array(delta);
        if (!pcmBytes.length) {
          return;
        }
        responseState.audioChunks.push(new Uint8Array(pcmBytes));
        const audioContext = callAudioContextRef.current;
        const destination = callDestinationRef.current;
        if (!audioContext || !destination) {
          return;
        }
        const float32 = pcm16BytesToFloat32(pcmBytes);
        if (!float32.length) {
          return;
        }
        const buffer = audioContext.createBuffer(1, float32.length, TARGET_SAMPLE_RATE);
        buffer.copyToChannel(Float32Array.from(float32), 0);
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(destination);
        const startAt = Math.max(callPlaybackTimeRef.current, audioContext.currentTime);
        source.start(startAt);
        callPlaybackTimeRef.current = startAt + buffer.duration;
      } else if (eventType.startsWith("response.") && eventType.endsWith(".delta") && !eventType.includes("audio")) {
        appendTextDelta(event.delta, responseState.textParts);
        const currentText = responseState.textParts.join("");
        console.debug("Text delta received", { eventType, delta: event.delta, currentText, responseKey });
        if (currentText.trim()) {
          setCallTranscripts((prev) => ({
            ...prev,
            [responseKey]: {
              text: currentText,
              isFinal: false,
              timestamp: responseState.timestamp,
            },
          }));
        }
      } else if (eventType === "response.text.delta" || eventType === "response.output_text.delta" || eventType === "response.audio_transcript.delta") {
        const delta = event.delta;
        if (typeof delta === "string" && delta) {
          responseState.textParts.push(delta);
          const currentText = responseState.textParts.join("");
          console.debug("Explicit text delta", { eventType, delta, currentText, responseKey });
          setCallTranscripts((prev) => ({
            ...prev,
            [responseKey]: {
              text: currentText,
              isFinal: false,
              timestamp: responseState.timestamp,
            },
          }));
        }
      } else if (eventType === "response.created") {
        ensureRealtimeResponseState(responseKey);
      } else if (eventType === "response.completed" || eventType === "response.done") {
        callAwaitingResponseRef.current = false;
        callStreamingStateRef.current = "idle";
        callHasSpeechSinceCommitRef.current = false;

        if (!responseState.textParts.length) {
          const responsePayload = responseCandidate ?? event;
          const segments = extractTextFromResponsePayload(responsePayload);
          if (segments.length) {
            responseState.textParts.push(...segments);
          }
        }
        let transcript = responseState.textParts.join("").trim();
        if (!transcript) {
          const messageText = typeof event.message === "string" ? event.message.trim() : "";
          if (messageText) {
            transcript = messageText;
          }
        }

        let audioUrl: string | undefined;
        if (responseState.audioChunks.length) {
          const wavBlob = pcm16ChunksToWavBlob(responseState.audioChunks, TARGET_SAMPLE_RATE);
          if (wavBlob) {
            audioUrl = URL.createObjectURL(wavBlob);
          }
        }

        console.debug("Response completed", { transcript, audioUrl, textParts: responseState.textParts, responseKey });
        if (transcript || audioUrl) {
          const message: Message = {
            id: uuid(),
            role: "zara",
            text: transcript || "Zara sent an audio reply.",
            audioUrl,
            timestamp: responseState.timestamp,
          };
          setHistory((prev) => [...prev, message]);
          if (transcript) {
            setCallTranscripts((prev) => ({
              ...prev,
              [responseKey]: {
                text: transcript,
                isFinal: true,
                timestamp: responseState.timestamp,
              },
            }));
          }
        }

        callResponseStateRef.current.delete(responseKey);
      } else if (eventType === "response.error" || eventType === "error") {
        const errorInfo = event.error;
        let message: string | null = null;
        if (errorInfo && typeof errorInfo === "object") {
          const errorMessage = (errorInfo as { message?: unknown }).message;
          if (typeof errorMessage === "string") {
            message = errorMessage;
          }
        }
        if (!message && typeof event.message === "string") {
          message = event.message;
        }
        const finalMessage = message ?? "Realtime response error.";
        console.error("Realtime response error", finalMessage, payload);
        setError("实时通话连接失败，请稍后再试。");
        if (callResponseStateRef.current.has(responseKey)) {
          callResponseStateRef.current.delete(responseKey);
        }
      } else if (!eventType.startsWith("input_audio_buffer.") && !eventType.startsWith("session.")) {
        console.debug("Unhandled realtime event", payload);
      }
    },
    [ensureRealtimeResponseState, getRealtimeResponseState, setError, setHistory],
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
            },
          };
          ws.send(JSON.stringify(sessionUpdate));
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
    if (callProvider === "live-interpreter") {
      await startLiveInterpreterCall();
      return;
    }
    if (callTransport === "webrtc") {
      await startWebrtcCall();
    } else {
      await startWebsocketCall();
    }
  }, [callProvider, callTransport, startLiveInterpreterCall, startWebrtcCall, startWebsocketCall]);

  const callActive = callStatus === "connected" || callStatus === "connecting";

  useEffect(() => {
    if (callTransport === "webrtc" && callProvider !== "gpt-realtime") {
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
                <label className="transport-option" style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                  <input
                    type="radio"
                    name="call-provider"
                    value="live-interpreter"
                    checked={callProvider === "live-interpreter"}
                    onChange={handleProviderChange}
                    disabled={callActive}
                  />
                  <span>Live Interpreter</span>
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
                    disabled={callActive || callProvider !== "gpt-realtime"}
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
                  disabled={callStatus === "connecting"}
                >
                  {callActive ? "结束实时通话" : "开始实时通话"}
                </button>
              </div>
              <p className="helper-text">
                {callProvider === "live-interpreter"
                  ? "Live Interpreter 会在检测到停顿时把你的语音发送到翻译服务，并播放译文语音。"
                  : callProvider === "voicelive"
                    ? "VoiceLive 使用 WebSocket 直连，请确保浏览器已授权麦克风。"
                    : "启动后 Zara 会实时倾听与回应，建议佩戴耳机防止回声。"}
              </p>
              <div className="audio-player" style={{ marginTop: "0.75rem" }}>
                <audio ref={remoteAudioRef} autoPlay playsInline />
              </div>
              {callStatus === "connected" && Object.keys(callTranscripts).length > 0 && (
                <div style={{ marginTop: "1rem", padding: "1rem", borderRadius: "12px", backgroundColor: "rgba(15, 23, 42, 0.6)", border: "1px solid rgba(148, 163, 184, 0.2)" }}>
                  <h3 style={{ margin: "0 0 0.75rem", fontSize: "0.95rem", color: "rgba(226, 232, 240, 0.9)" }}>实时转录</h3>
                  {Object.entries(callTranscripts).map(([key, data]) => (
                    <div
                      key={key}
                      style={{
                        marginBottom: "0.5rem",
                        padding: "0.5rem 0.75rem",
                        borderRadius: "8px",
                        backgroundColor: data.isFinal ? "rgba(56, 189, 248, 0.1)" : "rgba(148, 163, 184, 0.1)",
                        borderLeft: `3px solid ${data.isFinal ? "rgba(56, 189, 248, 0.5)" : "rgba(148, 163, 184, 0.3)"}`,
                      }}
                    >
                      <div style={{ fontSize: "0.85rem", color: data.isFinal ? "rgba(226, 232, 240, 0.95)" : "rgba(226, 232, 240, 0.7)", fontStyle: data.isFinal ? "normal" : "italic" }}>
                        {data.text || "..."}
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "rgba(148, 163, 184, 0.6)", marginTop: "0.25rem" }}>
                        {new Date(data.timestamp).toLocaleTimeString()} {data.isFinal ? "✓" : "⋯"}
                      </div>
                    </div>
                  ))}
                </div>
              )}
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
