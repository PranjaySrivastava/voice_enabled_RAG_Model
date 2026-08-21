'use client';

import React from 'react';
import { Zap, ShieldCheck, Database, Cpu, Layers, HardDrive } from 'lucide-react';

export interface WaterfallProps {
  waterfall?: {
    stt_ms: number;
    guardrail_ms: number;
    cache_lookup_ms: number;
    dense_retrieval_ms: number;
    sparse_retrieval_ms: number;
    rrf_fusion_ms: number;
    llm_generation_ms: number;
    total_compute_ms: number;
    total_ms: number;
  };
}

export function LatencyWaterfall({ waterfall }: WaterfallProps) {
  if (!waterfall) {
    return (
      <div className="p-4 bg-zinc-950/70 border border-zinc-850 rounded-xl text-center text-zinc-500 font-mono text-xs">
        No query latency waterfall telemetry available.
      </div>
    );
  }

  const stages = [
    {
      name: '1. Speech-to-Text (Sarvam)',
      icon: <Zap className="w-3.5 h-3.5 text-indigo-400" />,
      ms: waterfall.stt_ms,
      color: 'bg-indigo-500',
      active: waterfall.stt_ms > 0
    },
    {
      name: '2. 4-Tier Guardrail Matrix',
      icon: <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />,
      ms: waterfall.guardrail_ms,
      color: 'bg-cyan-400',
      active: true
    },
    {
      name: '3. LanceDB Dense Vector Scan',
      icon: <Database className="w-3.5 h-3.5 text-emerald-400" />,
      ms: waterfall.dense_retrieval_ms,
      color: 'bg-emerald-500',
      active: true
    },
    {
      name: '4. BM25 Sparse Search + RRF',
      icon: <Layers className="w-3.5 h-3.5 text-teal-400" />,
      ms: waterfall.sparse_retrieval_ms + waterfall.rrf_fusion_ms,
      color: 'bg-teal-400',
      active: true
    },
    {
      name: '5. Groq LPU Token Generation',
      icon: <Cpu className="w-3.5 h-3.5 text-amber-400" />,
      ms: waterfall.llm_generation_ms,
      color: 'bg-amber-400',
      active: true
    }
  ];

  const total = waterfall.total_ms || 1;

  return (
    <div className="w-full bg-zinc-950/80 border border-zinc-800/80 rounded-xl p-4 font-mono text-xs space-y-3 shadow-xl">
      <div className="flex justify-between items-center border-b border-zinc-800/80 pb-2">
        <div className="flex items-center gap-2">
          <HardDrive className="w-4 h-4 text-emerald-400" />
          <span className="font-bold text-zinc-200 tracking-wider text-[11px]">
            STAGE-WISE LATENCY WATERFALL
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold">
            COMPUTE: {waterfall.total_compute_ms} ms
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-300 font-bold">
            TOTAL: {waterfall.total_ms} ms
          </span>
        </div>
      </div>

      {/* Visual Proportional Bar */}
      <div className="w-full h-2 rounded-full bg-zinc-900 overflow-hidden flex border border-zinc-800">
        {stages.filter(s => s.active).map((s, idx) => {
          const pct = Math.max(2, Math.round((s.ms / total) * 100));
          return (
            <div
              key={idx}
              className={`${s.color} h-full transition-all duration-300`}
              style={{ width: `${pct}%` }}
              title={`${s.name}: ${s.ms} ms (${pct}%)`}
            />
          );
        })}
      </div>

      {/* Stage Breakdown Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
        {stages.filter(s => s.active).map((s, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between p-2 rounded-lg bg-zinc-900/60 border border-zinc-850 hover:border-zinc-700 transition-colors"
          >
            <div className="flex items-center gap-2 text-zinc-400 text-[11px]">
              {s.icon}
              <span>{s.name}</span>
            </div>
            <span className="font-bold text-zinc-200">{s.ms} ms</span>
          </div>
        ))}
      </div>
    </div>
  );
}
