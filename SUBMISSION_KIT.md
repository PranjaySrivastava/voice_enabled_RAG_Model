# 🏆 Hackathon Goa 2026: Task 2 Official Submission Kit

**Project Name**: Dhwani (ध्वनि) — Ultra-Low Latency Multilingual Indic Voice-RAG Engine  
**Author / Submitter**: Pranjay Srivastava  
**Official Form Link**: [https://forms.gle/MNvCjcv23Hn2Eeu58](https://forms.gle/MNvCjcv23Hn2Eeu58)  
**Submission Deadline**: August 22, 2026, 11:59 PM  

---

## 📋 Direct Copy-Paste Form Responses

### 1. Project Title & Tagline
- **Project Title**: `Dhwani (ध्वनि) — Multilingual Voice AI Engine`
- **Short Tagline**: `Sub-200ms Voice-Enabled RAG with 5-Way Hybrid Chunking, Hybrid Dense-Sparse RRF Retrieval, and 4-Tier Guardrails across 12+ Indic Languages.`

---

### 2. Technical Requirements Breakdown

#### A. Speech-to-Text Implementation
> **Implementation**: Integrated with **Sarvam AI (`saaras:v3`)** via binary streaming WebSockets (`/ws/rag`) and direct HTTP endpoints (`/api/voice`), supported by browser Web Audio API with adaptive ambient noise calibration and energy-based Voice Activity Detection (VAD). Supports 12+ Indic languages (Hindi, Tamil, Telugu, Gujarati, Marathi, Bengali, Kannada, Malayalam, Punjabi, Odia, Indian English).

#### B. Chunking Strategy (5-Way Hybrid Multi-Granularity Pipeline)
> Rather than naive fixed-size chunking, Dhwani processes the **ai4bharat/MSMARCO-XI** Indic dataset using 5 complementary strategies:
> 1. **Atomic Proposition Extraction (Micro-Units)**: Deconstructs compound sentences into independent atomic assertions to create precise vector search targets.
> 2. **Entity-Preserving Hierarchical Parent-Child**: 45-word sliding windows with 12-word overlap paired with full parent passage context for generation.
> 3. **Semantic Sentence Boundary**: Splits along natural Indic punctuation (`।`, `\n`) and English boundary marks (`.`, `?`, `!`).
> 4. **Query-Conditioned Intent Augmentation**: Injects search query intent into chunks to align user question vectors.
> 5. **Cross-Lingual Syncretic Alignment**: Links translated Hindi passages directly with parent English context.

#### C. Hybrid Retrieval & Latency Performance (< 200ms SLA Target)
> - **Dense Retrieval**: LanceDB IVF-PQ cosine vector index (`BAAI/bge-small-en-v1.5`).
> - **Sparse Keyword Search**: In-memory BM25 inverted index.
> - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse ranks with $Score = \frac{1}{60 + R_{\text{dense}}} + \frac{1}{60 + R_{\text{sparse}}}$.
> - **Semantic In-Memory Cache**: Sub-1ms retrieval for frequent queries.
> - **Core Compute P50 Latency**: **`153.70 ms`** (100% compliant with the `< 200ms` SLA target).

#### D. Latency Analytics & Statistical Percentiles
> Measured across 25+ sequential queries via the automated benchmark suite:
> - **Core Compute P50 (Median)**: `153.70 ms`
> - **Core Compute P70**: `167.74 ms`
> - **Core Compute P90**: `175.09 ms`
> - **Core Compute P100 (Worst-case)**: `185.20 ms`
> - **LanceDB Hybrid Retrieval**: `43.69 ms`
> - **Guardrail Evaluation**: `0.010 ms`
> - **Sub-200ms Compliance Rate**: `100.0%`

#### E. Model Harness & Error Recovery
> - **Structured Orchestration**: Pydantic v2 schemas (`DhwaniRAGResponse`, `LatencyWaterfall`, `RetrievedCitation`).
> - **Resilience**: Async jitter retries (`@async_jitter_retry`) with exponential backoff.
> - **Circuit Breakers**: In-memory circuit breaker protecting against upstream provider rate limits.
> - **Multi-Model Failover Cascade**: Cascades across high-speed Groq LPU models (`groq/compound-mini` $\to$ `llama-3.1-8b-instant`).

#### F. 4-Tier Comprehensive Guardrail Matrix
> - **Tier 1 (Prompt Injection & Safety Armor)**: Pre-retrieval regex heuristic blocking dangerous, toxic, or exploit queries in `< 0.05 ms`.
> - **Tier 2 (Out-of-Domain Gating)**: Rejects English queries with cosine distance $> 0.85$ to prevent speculative hallucination while seamlessly enabling cross-lingual Indic generation.
> - **Tier 3 (NLI Groundedness Gate)**: Cross-references answer tokens against retrieved context passages.
> - **Tier 4 (Script & Language Consistency Gate)**: Verifies answers match requested Indic scripts.
> - **Guardrail Rejection Precision**: `100.0%` on adversarial prompts.

---

## 🔗 Links & Resources Checklist

- [x] **GitHub Repository**: `https://github.com/PranjaySrivastava/voice_enabled_RAG_Model`
- [x] **Live Interactive Web App (Vercel)**: `https://voice-enabled-rag-model-neon.vercel.app`
- [x] **Live FastAPI Backend (Render)**: `https://dhwani-voice-backend.onrender.com`
- [x] **Interactive Swagger API Docs**: `https://dhwani-voice-backend.onrender.com/docs`
- [ ] **Video 1 (90s Team/Process Video)**: Uploaded to Instagram & X with `#RAGInGoa`
- [ ] **Video 2 (Product Demo Video)**: Uploaded to Instagram & X with `#RAGInGoa`
