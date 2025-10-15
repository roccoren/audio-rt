import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const TARGET_SAMPLE_RATE = 24000;

type ConversationMode = "text" | "voice" | "call";

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

export default function App(): JSX.Element {
  const [mode, setMode] = useState<ConversationMode>("voice");
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
  }, [callStatus, capturedAudio, error, isRecording, loading, mode, text]);

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

  const stopCall = useCallback(() => {
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
    }
    setCallStatus("idle");
    sessionInfoRef.current = null;
  }, []);

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

  const startCall = useCallback(async () => {
    if (callStatus !== "idle") {
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
            voice: "alloy",
            modalities: ["audio", "text"],
            input_audio_format: "pcm16",
            output_audio_format: "pcm16",
          },
        };
        eventsChannel.send(JSON.stringify(sessionUpdate));
        const responseCreate = {
          type: "response.create",
          response: {
            modalities: ["audio"],
            instructions: ZARA_INSTRUCTIONS,
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
          stopCall();
        }
      });
      peer.addEventListener("connectionstatechange", () => {
        console.debug("peer connection state", peer.connectionState);
        const state = peer.connectionState;
        if (state === "failed" || state === "closed") {
          stopCall();
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
      stopCall();
    }
  }, [callStatus, stopCall]);

  const callActive = callStatus === "connected" || callStatus === "connecting";

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
      sessionInfoRef.current = null;
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
              <p className="helper-text">启动后 Zara 会实时倾听与回应，建议佩戴耳机防止回声。</p>
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
