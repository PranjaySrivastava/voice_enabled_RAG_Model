---
title: Dhwani (ध्वनि) - Indic Voice-RAG Engine
emoji: 🎙️
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: false
license: mit
---

# 🎙️ Dhwani (ध्वनि): Ultra-Low Latency Multilingual Indic Voice-RAG Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15.0-black)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![LanceDB](https://img.shields.io/badge/VectorDB-LanceDB-orange)](https://lancedb.github.io/lancedb/)
[![Sarvam AI](https://img.shields.io/badge/STT-Sarvam%20AI%20Saaras:v3-purple)](https://www.sarvam.ai/)
[![Groq LPU](https://img.shields.io/badge/LLM%20Inference-Groq%20LPU-red)](https://groq.com/)

> **Engineered for Hackathon Goa (HH Goa) 2026 Shortlisting Task 2: Build a Voice-Enabled RAG Model**  
> *Sub-200ms Core Compute SLA · 5-Way Hybrid Multi-Granularity Chunking · Dense-Sparse RRF Fusion · 4-Tier Guardrail Matrix · 11+ Indic Languages*

---

### 🌐 Live Production Deployments:
- 🎙️ **Interactive Voice Studio App**: [https://voice-enabled-rag-model-neon.vercel.app](https://voice-enabled-rag-model-neon.vercel.app)
- ⚡ **FastAPI Backend (Render)**: [https://dhwani-voice-backend.onrender.com](https://dhwani-voice-backend.onrender.com)
- 📖 **Interactive Swagger UI (API Docs)**: [https://dhwani-voice-backend.onrender.com/docs](https://dhwani-voice-backend.onrender.com/docs)
- 🩺 **System Health Check**: [https://dhwani-voice-backend.onrender.com/api/health](https://dhwani-voice-backend.onrender.com/api/health)
- 💻 **GitHub Repository**: [https://github.com/PranjaySrivastava/voice_enabled_RAG_Model](https://github.com/PranjaySrivastava/voice_enabled_RAG_Model)

---

## 📌 Architectural Blueprint

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      1. INGESTION & SPEECH-TO-TEXT                     │
 │  🎙️ Voice Audio ──► Web Audio Analyzer / VAD ──► Sarvam AI (saaras:v3) │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │               2. TIER-1 HEURISTIC SAFETY & INJECTION ARMOR             │
 │  🛡️ Regex Injection / Jailbreak / Toxicity Gating (< 0.05 ms latency)   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ (Pass)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 3. HYBRID DENSE-SPARSE RRF RETRIEVAL                   │
 │  ⚡ Semantic LRU Cache (< 1 ms)                                         │
 │  ├── LanceDB IVF-PQ Dense Cosine Search (BAAI/bge-small-en-v1.5)       │
 │  ├── In-Memory BM25 Sparse Keyword Inverted Index                      │
 │  └── Reciprocal Rank Fusion: RRF(d) = 1/(60+Rank_dense) + 1/(60+Rank_sp)│
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │              4. TIER-2 OUT-OF-DISTRIBUTION (OOD) SEMANTIC GATE         │
 │  📐 Cosine Distance Threshold Check (d > 0.62 ──► Refuse Speculation)  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ (In-Domain)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 5. REFLEXION HARNESS & GROQ LPU GENERATION             │
 │  🚀 Circuit Breakers + Exponential Jitter Retries                       │
 │  └── High-Speed Groq LPU Cascade (groq/compound-mini ──► gpt-oss-20b)  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │              6. TIER-3 NLI GROUNDING & TIER-4 SCRIPT VERIFICATION      │
 │  🔍 Concept Token Overlap Check + Native Script Verification           │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      7. SYNTHESIS & USER INTERFACE                     │
 │  🔊 Indic Text-to-Speech (gTTS) ──► Next.js 15 Holographic Studio UI   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Core Technical Innovations

### 1. 5-Way Hybrid Multi-Granularity Chunking Engine ([`backend/core/chunking.py`](backend/core/chunking.py))
Standard fixed-size chunking discards cross-sentence context and mangles grammatical structures. Dhwani processes the **ai4bharat/MSMARCO-XI** Indic dataset using 5 complementary chunking strategies:

1. **Atomic Proposition Extraction (Micro-Units)**: Breaks complex sentences into atomic factual propositions for pinpoint vector targeting.
2. **Entity-Preserving Hierarchical Parent-Child**: 45-word sliding child windows with 12-word overlap paired with full parent passage context for generation.
3. **Semantic Sentence Boundary**: Splits along natural Indic (`।`, `\n`) and English punctuation marks (`.`, `?`, `!`).
4. **Query-Conditioned Intent Augmentation**: Injects search intent keywords into chunks to maximize query-passage vector alignment.
5. **Cross-Lingual Syncretic Alignment**: Links translated Hindi/Indic passages directly with English source context.

#### Database Index Breakdown (LanceDB):
- **Total Indexed Chunks**: `6,363 chunks`
- **Hierarchical Parent-Child**: `2,365 chunks (37.2%)`
- **Cross-Lingual Aligned**: `2,242 chunks (35.2%)`
- **Semantic Sentence Boundary**: `1,489 chunks (23.4%)`
- **Query-Conditioned Metadata**: `267 chunks (4.2%)`
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors)

---

### 2. Hybrid Dense-Sparse RRF Retrieval ([`backend/core/retrieval.py`](backend/core/retrieval.py))
- **Dense Vector Search**: LanceDB IVF-PQ cosine index (`BAAI/bge-small-en-v1.5`).
- **Sparse Keyword Search**: In-memory BM25 index matching exact entities and terms.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse ranks:
  $$\text{RRF Score}(d) = \frac{1}{60 + \text{Rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{Rank}_{\text{sparse}}(d)}$$
- **Semantic LRU Cache**: Sub-millisecond ($< 1\text{ ms}$) retrieval for frequent queries.

---

### 3. 4-Tier Comprehensive Guardrail Matrix ([`backend/core/guardrails.py`](backend/core/guardrails.py))
- **Tier 1 (Prompt Injection & Safety Armor)**: Pre-retrieval heuristic regex filter blocking dangerous, toxic, or exploit queries ($< 0.05\text{ ms}$).
- **Tier 2 (Out-of-Distribution Gating)**: Rejects queries with cosine distance $> 0.62$ to prevent speculative hallucination.
- **Tier 3 (NLI Groundedness Gate)**: Evaluates substantive token and entity overlap between generated answers and retrieved context passages.
- **Tier 4 (Script & Language Consistency Gate)**: Verifies that answers match the user's requested Indic script.

---

### 4. Reflexion & Self-Correction Harness ([`backend/core/harness.py`](backend/core/harness.py))
- **Async Jitter Retries (`@async_jitter_retry`)**: Prevents thundering-herd issues during peak load with exponential backoff.
- **Circuit Breaker**: Isolates external API failures (threshold: 3 consecutive failures, recovery: 15s) and seamlessly falls back to extracted factual context.
- **High-Speed Groq LPU Cascade**: `groq/compound-mini` $\to$ `openai/gpt-oss-20b` $\to$ `openai/gpt-oss-120b` $\to$ `qwen/qwen3.6-27b`.

---

## 📊 Benchmark Telemetry & SLA Compliance

Measured across 25+ sequential queries using the automated test suite ([`backend/benchmark.py`](backend/benchmark.py)):

| Performance Metric | Measured Value | SLA Target | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Core Compute P50 (Median)** | **`153.70 ms`** | `< 200.0 ms` | 🎯 **100.0% PASS** |
| **P70 Latency** | **`167.74 ms`** | `< 200.0 ms` | 🎯 **100.0% PASS** |
| **P90 Latency** | **`175.09 ms`** | `< 200.0 ms` | 🎯 **100.0% PASS** |
| **P100 Latency (Worst-case)** | **`185.20 ms`** | Real-Time | 🎯 **100.0% PASS** |
| **LanceDB Hybrid Retrieval** | **`43.69 ms`** | `< 50.0 ms` | 🎯 **PASS** |
| **4-Tier Guardrail Overhead** | **`0.010 ms`** | `< 1.0 ms` | 🎯 **PASS** |
| **Safety Guardrail Accuracy** | **`100.0%`** | `> 95.0%` | 🎯 **PASS** |

---

## 🛠️ Quickstart & Local Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Active API keys for Sarvam AI and Groq

### 1. Clone & Install
```bash
# Clone repository
git clone https://github.com/PranjaySrivastava/voice_enabled_RAG_Model.git
cd voice_enabled_RAG_Model

# Install Python backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
npm install
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```ini
SARVAM_API_KEY=your_sarvam_api_key
GROQ_API_KEY=your_groq_api_key
GUARDRAIL_DISTANCE_THRESHOLD=0.62
DB_PATH=lancedb_msmarco
```

### 3. Start Backend Server
```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 4. Start Next.js Frontend Studio
In a second terminal:
```bash
npm run dev
```
Open **`http://localhost:3000`** in your browser.

### 5. Run Latency Benchmarks
```bash
python backend/benchmark.py --samples 25 --mode rag
```

---

## 📁 Repository Structure

```
.
├── backend/
│   ├── core/
│   │   ├── chunking.py         # 5-Way Hybrid Adaptive Chunker
│   │   ├── retrieval.py        # Hybrid Dense-Sparse RRF + Semantic Cache
│   │   ├── guardrails.py       # 4-Tier Guardrail Matrix & Refusal Taxonomy
│   │   └── harness.py          # Reflexion Harness & Circuit Breaker
│   ├── server.py               # FastAPI & Streaming WebSocket Server
│   ├── benchmark.py            # P50/P70/P100 Statistical Benchmark Runner
│   └── requirements.txt        # Python backend dependencies
├── dataset_prep/
│   └── build_vector_index.py   # MSMARCO-XI 5-way hybrid index builder
├── src/
│   ├── components/
│   │   ├── VoiceParticleOrb.tsx# Interactive 3D Audio Visualizer Canvas
│   │   ├── LatencyWaterfall.tsx# Microsecond Stage-Wise Waterfall Chart
│   │   ├── LatencyDashboard.tsx# Percentile Gauges & Live Benchmark Trigger
│   │   ├── GuardrailInspector.tsx# Interactive 4-Tier Guardrail Test Lab
│   │   └── ProvenanceTree.tsx  # Citation Hierarchy & Proposition Viewer
│   └── hooks/
│       └── useVoiceRAG.ts      # Web Audio Recording & Energy VAD Hook
├── app/
│   ├── layout.tsx              # Next.js Root Layout
│   └── page.tsx                # Dhwani Studio Main UI
├── app.py                      # Hugging Face Spaces Gradio app with ZeroGPU
├── BENCHMARK_REPORT.md         # Detailed latency benchmark report
├── DEPLOYMENT_GUIDE.md         # Hugging Face Spaces + Vercel deployment guide
├── SUBMISSION_KIT.md           # Official Task 2 Submission Form responses
└── README.md                   # Project documentation
```

---

## 🏆 Hackathon Goa 2026 Submission
Built for **HH Goa 2026 Shortlisting Task 2: Build a Voice-Enabled RAG Model** by **Pranjay Srivastava**.
