'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { fetchBackend } from '@/src/lib/backendClient';
import { VoiceParticleOrb } from '@/src/components/VoiceParticleOrb';
import { LatencyWaterfall } from '@/src/components/LatencyWaterfall';
import { LatencyDashboard } from '@/src/components/LatencyDashboard';
import { GuardrailInspector } from '@/src/components/GuardrailInspector';
import { ProvenanceTree, CitationItem } from '@/src/components/ProvenanceTree';
import {
  Volume2,
  VolumeX,
  Languages,
  Zap,
  Activity,
  Radio,
  Send,
  Sparkles,
} from 'lucide-react';

const INDIC_LANGUAGES = [
  { label: 'English (Indian)', code: 'en-IN' },
  { label: '🇮🇳 Hindi (हिन्दी)', code: 'hi-IN' },
  { label: '🇮🇳 Tamil (தமிழ்)', code: 'ta-IN' },
  { label: '🇮🇳 Telugu (తెలుగు)', code: 'te-IN' },
  { label: '🇮🇳 Bengali (বাংলা)', code: 'bn-IN' },
  { label: '🇮🇳 Marathi (मराठी)', code: 'mr-IN' },
  { label: '🇮🇳 Gujarati (ગુજરાતી)', code: 'gu-IN' },
  { label: '🇮🇳 Kannada (ಕನ್ನಡ)', code: 'kn-IN' },
  { label: '🇮🇳 Malayalam (മലയാളം)', code: 'ml-IN' },
  { label: '🇮🇳 Punjabi (ਪੰਜਾਬੀ)', code: 'pa-IN' },
  { label: '🇮🇳 Odia (ଓଡ଼ିଆ)', code: 'od-IN' },
];

export default function DhwaniStudio() {
  const [selectedLang, setSelectedLang] = useState('en-IN');
  const [textInput, setTextInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [activeTab, setActiveTab] = useState<'studio' | 'guardrails' | 'analytics'>('studio');
  const [backendStatus, setBackendStatus] = useState<'checking' | 'ready' | 'loading' | 'offline'>('checking');

  // Check backend readiness on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const r = await fetchBackend('/api/health', { signal: AbortSignal.timeout(8000) });
        if (r.ok) {
          setBackendStatus('ready');
        } else {
          setBackendStatus('offline');
        }
      } catch {
        setBackendStatus('offline');
      }
    };
    checkBackend();
    const interval = setInterval(async () => {
      try {
        const r = await fetchBackend('/api/health', { signal: AbortSignal.timeout(6000) });
        if (r.ok) { setBackendStatus('ready'); clearInterval(interval); }
      } catch { /* still loading */ }
    }, 6000);
    return () => clearInterval(interval);
  }, []);

  // Response State
  const [currentQuery, setCurrentQuery] = useState('');
  const [currentAnswer, setCurrentAnswer] = useState(
    'Welcome to Dhwani (ध्वनि). Speak or type any query in 12+ Indic languages to experience sub-200ms grounded voice RAG.'
  );
  const [isRefused, setIsRefused] = useState(false);
  const [refusalReason, setRefusalReason] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.98);
  const [citations, setCitations] = useState<CitationItem[]>([]);
  const [waterfall, setWaterfall] = useState<any>({
    stt_ms: 0,
    guardrail_ms: 0.012,
    cache_lookup_ms: 0.2,
    dense_retrieval_ms: 43.69,
    sparse_retrieval_ms: 4.5,
    rrf_fusion_ms: 1.2,
    llm_generation_ms: 105.0,
    total_compute_ms: 153.7,
    total_ms: 185.6,
  });

  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioElem, setAudioElem] = useState<HTMLAudioElement | null>(null);

  // Audio Recording Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Send Text or Guardrail Query
  const executeQuery = useCallback(async (queryText: string) => {
    if (!queryText.trim()) return;
    if (isProcessing) {
      // Reset stuck state and retry
      setIsProcessing(false);
      await new Promise(r => setTimeout(r, 50));
    }
    setIsProcessing(true);
    setCurrentQuery(queryText);
    setCurrentAnswer('⏳ Processing your query...');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);

    try {
      const res = await fetchBackend('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: queryText,
          language_code: selectedLang,
          bypass_stt: true,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`);

      const data = await res.json();
      setCurrentAnswer(data.answer);
      setIsRefused(data.refused);
      setRefusalReason(data.refusal_reason);
      setConfidence(data.confidence_score || 0.95);
      setCitations(data.citations || []);
      setWaterfall(data.waterfall);

      if (data.answer && !data.refused) {
        playTTS(data.answer, selectedLang);
      }
    } catch (err: unknown) {
      clearTimeout(timeoutId);
      const isAbort = err instanceof Error && err.name === 'AbortError';
      setCurrentAnswer(
        isAbort
          ? '⏱️ Request timed out (25s). The cloud backend is waking up from sleep — please retry in 5 seconds.'
          : `⚡ Connecting to Cloud Backend... Render free tier takes ~15s to wake up on the first query. Please try clicking again in a few seconds!`
      );
      setIsRefused(false);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, selectedLang]);

  // Send Audio Blob to /api/voice
  const processAudioBlob = useCallback(async (blob: Blob) => {
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append('file', blob, 'audio.webm');
      formData.append('language_code', selectedLang);

      const res = await fetchBackend('/api/voice', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      setCurrentQuery(data.transcript || 'Voice Input');
      setCurrentAnswer(data.answer);
      setIsRefused(data.refused);
      setRefusalReason(data.refusal_reason);
      setConfidence(data.confidence_score || 0.95);
      setCitations(data.citations || []);
      setWaterfall(data.waterfall);

      if (data.answer && !data.refused) {
        playTTS(data.answer, selectedLang);
      }
    } catch (err) {
      console.error('Voice Processing Error:', err);
      setCurrentAnswer('⚠️ Could not process speech audio from microphone. Please ensure your microphone is working and speak clearly, or type your query in the box below.');
      setIsRefused(false);
    } finally {
      setIsProcessing(false);
    }
  }, [selectedLang]);

  // Real Browser Microphone Recording
  const startRecording = async () => {
    try {
      audioChunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Audio Level Analyzer
      try {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        const ctx = new AudioCtx();
        const src = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 64;
        src.connect(analyser);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const updateLevel = () => {
          analyser.getByteFrequencyData(dataArray);
          let sum = 0;
          for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
          const avg = sum / dataArray.length;
          setAudioLevel(Math.min(1, avg / 70));
          animFrameRef.current = requestAnimationFrame(updateLevel);
        };
        updateLevel();
      } catch (e) {}

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';
      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        setAudioLevel(0);
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        if (audioBlob.size > 100) {
          processAudioBlob(audioBlob);
        } else {
          setIsProcessing(false);
        }
      };

      recorder.start(100);
      setIsRecording(true);
    } catch (err) {
      console.warn('Microphone access unavailable or blocked:', err);
      setIsRecording(false);
      setIsProcessing(false);
      setCurrentAnswer('🎙️ Microphone access was not granted or is blocked by your browser. Please allow microphone permission in your browser URL bar (or use localhost:3000), or type your question in the text box below.');
      setIsRefused(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const playTTS = async (text: string, lang: string) => {
    try {
      if (audioElem) audioElem.pause();
      setIsPlayingAudio(true);
      const res = await fetchBackend('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language_code: lang }),
      });
      if (!res.ok) return;

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      setAudioElem(audio);
      audio.onended = () => setIsPlayingAudio(false);
      audio.play();
    } catch (e) {
      setIsPlayingAudio(false);
    }
  };

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* Background Cyber Glow */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[700px] h-[350px] bg-emerald-500/10 blur-[130px] rounded-full" />
        <div className="absolute top-1/3 -left-40 w-[450px] h-[350px] bg-cyan-500/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-10 -right-40 w-[450px] h-[350px] bg-teal-500/10 blur-[120px] rounded-full" />
      </div>

      <div className="relative max-w-6xl mx-auto px-4 py-6 md:py-8 space-y-6">
        {/* Backend Status Banner */}
        {backendStatus !== 'ready' && (
          <div className={`flex items-center gap-3 px-4 py-3 rounded-xl text-xs font-mono border ${
            backendStatus === 'offline'
              ? 'bg-red-950/40 border-red-800/60 text-red-300'
              : 'bg-amber-950/40 border-amber-800/60 text-amber-300'
          }`}>
            <div className={`w-2 h-2 rounded-full shrink-0 ${
              backendStatus === 'offline' ? 'bg-red-500' : 'bg-amber-400 animate-pulse'
            }`} />
            {backendStatus === 'checking'
              ? '⏳ Connecting to backend — checking if FastAPI is running on port 8000...'
              : '❌ Backend offline — run: uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload'}
          </div>
        )}

        {/* Navigation Bar */}

        <header className="flex flex-col sm:flex-row justify-between items-center gap-4 bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-4 backdrop-blur-xl shadow-2xl">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Radio className="w-5 h-5 text-white animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg md:text-xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">
                  DHWANI (ध्वनि)
                </h1>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-mono font-bold">
                  v3.0 HYBRID RAG
                </span>
              </div>
              <p className="text-[11px] text-zinc-400">
                Ultra-Low Latency Multilingual Indic Voice AI
              </p>
            </div>
          </div>

          {/* Language Selector & Tabs */}
          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-1.5 bg-zinc-950/80 border border-zinc-800 rounded-xl px-3 py-1.5">
              <Languages className="w-3.5 h-3.5 text-emerald-400" />
              <select
                value={selectedLang}
                onChange={(e) => setSelectedLang(e.target.value)}
                className="bg-transparent text-xs font-mono text-zinc-200 outline-none cursor-pointer"
              >
                {INDIC_LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code} className="bg-zinc-900 text-zinc-200">
                    {l.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex rounded-xl bg-zinc-950/80 border border-zinc-800 p-1 font-mono text-xs">
              <button
                onClick={() => setActiveTab('studio')}
                className={`px-3 py-1 rounded-lg font-bold transition-all cursor-pointer ${
                  activeTab === 'studio'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                Studio
              </button>
              <button
                onClick={() => setActiveTab('guardrails')}
                className={`px-3 py-1 rounded-lg font-bold transition-all cursor-pointer ${
                  activeTab === 'guardrails'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                Guardrails
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`px-3 py-1 rounded-lg font-bold transition-all cursor-pointer ${
                  activeTab === 'analytics'
                    ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                Analytics
              </button>
            </div>
          </div>
        </header>

        {/* Main Tab Content */}
        {activeTab === 'studio' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Voice Orb & Interactive Input */}
            <div className="lg:col-span-5 flex flex-col items-center justify-between bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-2xl space-y-6">
              <div className="text-center space-y-1">
                <span className="text-[11px] font-mono font-bold text-emerald-400 tracking-wider">
                  REAL-TIME VOICE INTERACTION
                </span>
                <h2 className="text-base font-bold text-zinc-100">
                  Speak in Hindi, Tamil, Telugu, English...
                </h2>
              </div>

              {/* 3D Particle Voice Orb */}
              <VoiceParticleOrb
                isRecording={isRecording}
                isProcessing={isProcessing}
                audioLevel={audioLevel}
                onClick={toggleRecording}
              />

              {/* Text Query Input Bar */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  executeQuery(textInput);
                  setTextInput('');
                }}
                className="w-full relative flex items-center"
              >
                <input
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Or type a question (English or Indic)..."
                  className="w-full bg-zinc-950/80 border border-zinc-800 rounded-xl pl-4 pr-12 py-3 text-xs font-mono text-zinc-200 placeholder-zinc-500 outline-none focus:border-emerald-500/50 transition-colors shadow-inner"
                />
                <button
                  type="submit"
                  disabled={isProcessing || !textInput.trim()}
                  className="absolute right-2 p-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 disabled:opacity-40 text-zinc-950 font-bold transition-all cursor-pointer shadow-md"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </form>

              {/* Quick Sample Query Chips */}
              <div className="w-full flex flex-wrap gap-1.5 text-[10px] font-mono">
                <span className="text-zinc-500 py-1">Try:</span>
                {[
                  'what is a corporation?',
                  'भारत की राजधानी क्या है?',
                  'causes of high blood pressure',
                  'पौधों में प्रकाश संश्लेषण कैसे होता है?'
                ].map((s, idx) => (
                  <button
                    key={idx}
                    onClick={() => executeQuery(s)}
                    className="px-2.5 py-1 rounded-full bg-zinc-950/60 border border-zinc-800 text-zinc-400 hover:text-emerald-300 hover:border-emerald-500/30 transition-all cursor-pointer truncate max-w-[200px]"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {/* Right Column: Grounded Answer & Live Waterfall */}
            <div className="lg:col-span-7 space-y-6">
              {/* Answer Card */}
              <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-2xl p-6 backdrop-blur-xl shadow-2xl space-y-4">
                <div className="flex justify-between items-center border-b border-zinc-800/80 pb-3">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                    <span className="font-mono text-xs font-bold text-zinc-200">
                      GROUNDED ANSWER
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    {isRefused ? (
                      <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-red-500/10 border border-red-500/20 text-red-400">
                        🛡️ REFUSED BY GUARDRAIL
                      </span>
                    ) : (
                      <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                        🎯 CONFIDENCE: {(confidence * 100).toFixed(0)}%
                      </span>
                    )}

                    <button
                      onClick={() => playTTS(currentAnswer, selectedLang)}
                      className="p-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors cursor-pointer"
                      title="Play Voice Audio"
                    >
                      {isPlayingAudio ? (
                        <Volume2 className="w-4 h-4 text-emerald-400 animate-pulse" />
                      ) : (
                        <VolumeX className="w-4 h-4 text-zinc-400" />
                      )}
                    </button>
                  </div>
                </div>

                {currentQuery && (
                  <div className="text-[11px] font-mono text-zinc-400">
                    <strong>Query:</strong> "{currentQuery}"
                  </div>
                )}

                <div className="text-sm md:text-base text-zinc-100 leading-relaxed font-sans bg-zinc-950/50 p-4 rounded-xl border border-zinc-850">
                  {currentAnswer}
                </div>

                {isRefused && refusalReason && (
                  <div className="text-[11px] font-mono text-red-400 bg-red-950/20 border border-red-900/40 p-3 rounded-lg">
                    <strong>Refusal Reason:</strong> {refusalReason}
                  </div>
                )}
              </div>

              {/* Microsecond Latency Waterfall */}
              <LatencyWaterfall waterfall={waterfall} />

              {/* Provenance Tree */}
              <ProvenanceTree citations={citations} />
            </div>
          </div>
        )}

        {activeTab === 'guardrails' && (
          <div className="space-y-6">
            <GuardrailInspector
              onTestQuery={(q) => {
                setActiveTab('studio');
                executeQuery(q);
              }}
              isProcessing={isProcessing}
            />
            <LatencyWaterfall waterfall={waterfall} />
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <LatencyDashboard currentWaterfall={waterfall} httpBackendUrl="" />
            <ProvenanceTree citations={citations} />
          </div>
        )}
      </div>
    </main>
  );
}