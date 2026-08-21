'use client';

import React, { useEffect, useRef } from 'react';

interface VoiceParticleOrbProps {
  isRecording: boolean;
  isProcessing: boolean;
  audioLevel: number;
  onClick: () => void;
}

export function VoiceParticleOrb({
  isRecording,
  isProcessing,
  audioLevel,
  onClick,
}: VoiceParticleOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let angle = 0;

    const numParticles = 48;
    const baseRadius = 60;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;

      // Glow effect
      const gradient = ctx.createRadialGradient(
        centerX,
        centerY,
        10,
        centerX,
        centerY,
        baseRadius + (audioLevel * 45)
      );

      if (isRecording) {
        gradient.addColorStop(0, 'rgba(239, 68, 68, 0.8)'); // Red pulse
        gradient.addColorStop(0.5, 'rgba(249, 115, 22, 0.4)');
        gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
      } else if (isProcessing) {
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.8)'); // Blue pulse
        gradient.addColorStop(0.5, 'rgba(147, 51, 234, 0.4)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
      } else {
        gradient.addColorStop(0, 'rgba(16, 185, 129, 0.8)'); // Emerald pulse
        gradient.addColorStop(0.5, 'rgba(6, 182, 212, 0.3)');
        gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');
      }

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, baseRadius + (audioLevel * 40), 0, Math.PI * 2);
      ctx.fill();

      // Outer particle ring
      angle += isProcessing ? 0.04 : 0.015;
      for (let i = 0; i < numParticles; i++) {
        const pAngle = (i / numParticles) * Math.PI * 2 + angle;
        const waveOffset = isRecording ? Math.sin(pAngle * 4 + angle * 3) * (audioLevel * 30) : Math.sin(pAngle * 3 + angle * 2) * 5;
        const r = baseRadius + waveOffset + (audioLevel * 20);
        const x = centerX + Math.cos(pAngle) * r;
        const y = centerY + Math.sin(pAngle) * r;

        ctx.fillStyle = isRecording ? '#ef4444' : (isProcessing ? '#38bdf8' : '#34d399');
        ctx.beginPath();
        ctx.arc(x, y, 2.5 + (audioLevel * 3), 0, Math.PI * 2);
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isRecording, isProcessing, audioLevel]);

  return (
    <div
      onClick={onClick}
      className="relative flex items-center justify-center cursor-pointer group select-none"
      title={isRecording ? "Click to Stop & Send" : "Click to Speak"}
    >
      <canvas
        ref={canvasRef}
        width={240}
        height={240}
        className="w-48 h-48 md:w-56 md:h-56 transition-transform group-hover:scale-105"
      />

      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <div className="w-16 h-16 rounded-full bg-zinc-950/80 border border-zinc-700/80 backdrop-blur-md flex items-center justify-center shadow-xl group-hover:border-emerald-500/50 transition-colors">
          {isRecording ? (
            <div className="w-5 h-5 bg-red-500 rounded-sm animate-pulse" />
          ) : isProcessing ? (
            <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg
              className="w-7 h-7 text-emerald-400 group-hover:scale-110 transition-transform"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            </svg>
          )}
        </div>
        <span className="text-[11px] font-mono font-bold mt-2.5 px-2.5 py-0.5 rounded-full bg-zinc-900/90 border border-zinc-800 text-zinc-300">
          {isRecording ? "LISTENING..." : isProcessing ? "THINKING..." : "TAP TO SPEAK"}
        </span>
      </div>
    </div>
  );
}
