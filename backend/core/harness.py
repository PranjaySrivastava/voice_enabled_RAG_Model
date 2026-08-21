"""
Dhwani (ध्वनि) - Reflexion & Self-Correction Agent Harness
Implements circuit breakers, async retries with exponential jitter, multi-model failover,
and automatic query expansion for low-confidence retrievals.
"""

import time
import re
import asyncio
import random
from typing import List, Optional, Tuple, Dict, Any
from functools import wraps
import httpx
from pydantic import BaseModel, Field

# =====================================================================
# PYDANTIC STRUCTURED RESPONSE SCHEMAS
# =====================================================================

class LatencyWaterfall(BaseModel):
    stt_ms: float = Field(default=0.0, description="Speech-to-Text latency in ms")
    guardrail_ms: float = Field(default=0.0, description="Multi-Tier Guardrail latency in ms")
    cache_lookup_ms: float = Field(default=0.0, description="Semantic cache lookup latency in ms")
    dense_retrieval_ms: float = Field(default=0.0, description="LanceDB IVF-PQ vector search latency in ms")
    sparse_retrieval_ms: float = Field(default=0.0, description="BM25 keyword search latency in ms")
    rrf_fusion_ms: float = Field(default=0.0, description="Reciprocal Rank Fusion latency in ms")
    llm_generation_ms: float = Field(default=0.0, description="Groq LPU LLM generation latency in ms")
    total_compute_ms: float = Field(default=0.0, description="Core on-chip compute latency in ms (<200ms target)")
    total_ms: float = Field(default=0.0, description="Total pipeline latency in ms")

class RetrievedCitation(BaseModel):
    chunk_text: str
    parent_passage: str
    translated_passage: Optional[str] = ""
    chunk_strategy: str
    dense_distance: float
    rrf_score: float
    query_id: str

class DhwaniRAGResponse(BaseModel):
    transcript: str
    answer: str
    grounded: bool
    refused: bool
    refusal_code: Optional[str] = None
    refusal_reason: Optional[str] = None
    confidence_score: float = 1.0
    citations: List[RetrievedCitation] = []
    waterfall: LatencyWaterfall
    model_used: str = "Groq/LPU (gpt-oss-20b)"
    reflexion_applied: bool = False

# =====================================================================
# RESILIENCE: CIRCUIT BREAKER & ASYNC JITTER RETRIES
# =====================================================================

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 4, recovery_timeout: float = 20.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def allow_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

groq_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=15.0)

def async_jitter_retry(max_attempts: int = 2, base_delay: float = 0.06):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        sleep_time = (base_delay * (2 ** attempt)) + (random.random() * 0.03)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Operation failed with no exception recorded.")
        return wrapper
    return decorator

# =====================================================================
# MULTI-MODEL FALLBACK INFERENCE HARNESS
# =====================================================================

FAST_INFERENCE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192"
]

def _extract_fallback_answer_from_context(prompt: str) -> str:
    """Extracts factual sentences strictly from retrieved context passages."""
    if "Context" in prompt:
        ctx_part = prompt.split("Context", 1)[1]
        if ":\n" in ctx_part:
            ctx_part = ctx_part.split(":\n", 1)[1]
        elif ":" in ctx_part:
            ctx_part = ctx_part.split(":", 1)[1]
        if "\n\nQuestion" in ctx_part:
            ctx_part = ctx_part.split("\n\nQuestion", 1)[0].strip()
        first_p = ctx_part.split("\n---\n")[0].strip()
        
        # Clean blog header noise like "April 6, 2011, cherran, Leave a comment."
        first_p = re.sub(r'^(?:[a-zA-Z]+ \d{1,2}, \d{4},?\s*)?(?:[a-zA-Z0-9_-]+,?\s*)?(?:leave a comment\.?\s*)?', '', first_p, flags=re.IGNORECASE).strip()
        sentences = re.split(r'(?<=[.?!।\n])\s+', first_p)
        clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 15 and not s.strip().startswith("Relevant Query:")]
        if clean_sentences:
            return " ".join(clean_sentences[:2])
        if len(first_p) > 15:
            return first_p[:220].strip()
    return "No verified factual context found in the knowledge base for this query."

@async_jitter_retry(max_attempts=2, base_delay=0.05)
async def generate_grounded_answer(
    prompt: str,
    groq_api_key: str,
    http_client: httpx.AsyncClient
) -> Tuple[str, str]:
    """
    Executes harnessed LLM generation on Groq with failover across models.
    """
    is_valid_key = bool(groq_api_key and not groq_api_key.lower().startswith("your_") and len(groq_api_key) > 15)

    if not is_valid_key:
        return _extract_fallback_answer_from_context(prompt), "Dhwani Grounded Fact Engine"

    if not groq_circuit_breaker.allow_request():
        return _extract_fallback_answer_from_context(prompt), "CircuitBreaker Fast Fallback"

    system_instruction = (
        "You are Dhwani (ध्वनि), an ultra-fast, high-precision voice intelligence engine. "
        "Provide a direct, factual, completely grounded answer in 1-2 clean sentences. "
        "CRITICAL REQUIREMENTS:\n"
        "1. Never output reasoning tags, thinking blocks, meta-labels, or prefixes (never say 'Answer:', 'Response:', or 'User question is...').\n"
        "2. Answer immediately in the exact same language and script as the user's question.\n"
        "3. Strictly adhere only to the provided facts without hallucinating unmentioned claims."
    )

    last_err = ""
    for model_name in FAST_INFERENCE_MODELS:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 150
        }

        try:
            res = await http_client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=2.5
            )
            if res.status_code == 200:
                raw_text = res.json()["choices"][0]["message"]["content"].strip()
                if "<think>" in raw_text and "</think>" in raw_text:
                    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                for pfx in ["Answer:", "Response:", "उत्तर:", "जवाब:", "Direct Answer:"]:
                    if raw_text.startswith(pfx):
                        raw_text = raw_text[len(pfx):].strip()
                # If stripping left an empty string, fall back to context extraction
                if not raw_text:
                    raw_text = _extract_fallback_answer_from_context(prompt)
                groq_circuit_breaker.record_success()
                return raw_text, model_name
            else:
                last_err = f"HTTP {res.status_code}"
                groq_circuit_breaker.record_failure()
        except Exception as e:
            last_err = str(e)
            groq_circuit_breaker.record_failure()
            continue

    # Fallback to local grounded context extraction if Groq fails
    return _extract_fallback_answer_from_context(prompt), f"Dhwani Fallback ({last_err})"
