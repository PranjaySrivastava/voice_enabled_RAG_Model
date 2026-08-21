'use client';

import React, { useState } from 'react';
import { BookOpen, GitBranch, ChevronDown, ChevronUp, CheckCircle, ExternalLink } from 'lucide-react';

export interface CitationItem {
  chunk_text: string;
  parent_passage: string;
  translated_passage?: string;
  chunk_strategy: string;
  dense_distance: number;
  rrf_score: number;
  query_id: string;
}

interface ProvenanceTreeProps {
  citations?: CitationItem[];
}

export function ProvenanceTree({ citations = [] }: ProvenanceTreeProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(0);

  if (!citations || citations.length === 0) {
    return (
      <div className="p-4 bg-zinc-950/70 border border-zinc-850 rounded-xl text-center text-zinc-500 font-mono text-xs">
        No retrieved knowledge citations for this query.
      </div>
    );
  }

  return (
    <div className="w-full bg-zinc-950/80 border border-zinc-800/80 rounded-xl p-4 font-mono text-xs space-y-3 shadow-xl">
      <div className="flex justify-between items-center border-b border-zinc-800/80 pb-2">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-emerald-400" />
          <span className="font-bold text-zinc-200 tracking-wider text-[11px]">
            CITATION PROVENANCE TREE ({citations.length} UNITS)
          </span>
        </div>
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold">
          5-WAY CHUNKING HIERARCHY
        </span>
      </div>

      <div className="space-y-2">
        {citations.map((c, idx) => {
          const isExpanded = expandedIdx === idx;
          const strategyName = c.chunk_strategy.replace(/_/g, ' ').toUpperCase();

          return (
            <div
              key={idx}
              className="rounded-lg bg-zinc-900/70 border border-zinc-850 overflow-hidden transition-colors"
            >
              <div
                onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                className="flex items-center justify-between p-3 cursor-pointer hover:bg-zinc-850/60 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 font-bold flex items-center justify-center text-[10px]">
                    #{idx + 1}
                  </span>
                  <div>
                    <span className="font-bold text-zinc-200 text-[11px] block">
                      {strategyName}
                    </span>
                    <span className="text-[10px] text-zinc-500">
                      Cosine Distance: <strong className="text-emerald-400">{c.dense_distance}</strong> | RRF: <strong className="text-teal-400">{c.rrf_score.toFixed(4)}</strong>
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-zinc-400 hidden sm:inline">
                    {isExpanded ? 'Hide Passage' : 'View Full Parent Passage'}
                  </span>
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-zinc-400" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-zinc-400" />
                  )}
                </div>
              </div>

              {/* Matched Child Fragment */}
              <div className="px-3 pb-3 pt-0">
                <div className="p-2 rounded bg-zinc-950/80 border border-zinc-800 text-zinc-300 text-[11px] leading-relaxed">
                  <strong className="text-emerald-400 block text-[10px] uppercase mb-1">
                    🎯 Matched Vector Target (Micro-Unit):
                  </strong>
                  "{c.chunk_text}"
                </div>

                {/* Expanded Parent Passage */}
                {isExpanded && c.parent_passage && (
                  <div className="mt-2 p-2.5 rounded bg-zinc-950/90 border border-zinc-800/80 text-zinc-400 text-[11px] leading-relaxed animate-in fade-in">
                    <strong className="text-cyan-400 block text-[10px] uppercase mb-1">
                      📚 Full Parent Context Window (Fed to Groq LLM):
                    </strong>
                    {c.parent_passage}

                    {c.translated_passage && (
                      <div className="mt-2 pt-2 border-t border-zinc-850">
                        <strong className="text-amber-400 block text-[10px] uppercase mb-1">
                          🇮🇳 Parallel Indic Translation (MSMARCO-XI):
                        </strong>
                        {c.translated_passage}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
