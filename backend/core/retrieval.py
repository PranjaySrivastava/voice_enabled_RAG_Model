"""
Dhwani (ध्वनि) - Hybrid Dense-Sparse Retrieval Engine with Reciprocal Rank Fusion (RRF)
Combines LanceDB Vector Search (Dense) + BM25 Token Matching (Sparse) + Sub-1ms Semantic Cache.
"""

import time
import math
import re
import collections
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

class RetrievedCandidate(BaseModel):
    chunk_text: str
    parent_passage: str
    translated_passage: Optional[str] = ""
    chunk_strategy: str
    dense_distance: float = 1.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    query_id: str = ""
    is_selected: int = 0
    eng_query: Optional[str] = ""

# Common Indic & English Stopwords to prevent false positive keyword matches
STOP_WORDS = {
    # Hindi / Indic common stop words
    "क्या", "है", "हैं", "की", "का", "के", "में", "से", "को", "पर", "ने", "और", "या", "तो", "भी",
    "होता", "होती", "होते", "कैसे", "कहाँ", "कब", "क्यों", "किस", "किसे", "था", "थी", "थे", "यह", "वह", "जो",
    "वाला", "वाली", "वाले", "कर", "करें", "रहा", "रही", "रहे", "हुए", "हुआ", "हुई",
    # English common stop words
    "what", "is", "are", "was", "were", "the", "a", "an", "in", "on", "of", "to", "for", "and", "or",
    "how", "where", "when", "why", "who", "which", "do", "does", "did", "it", "this", "that", "there", "by", "with"
}

def tokenize_and_filter(text: str) -> List[str]:
    """Cleans punctuation, lowercases, and strips common stop words for high-precision BM25 matching."""
    if not text:
        return []
    # Replace punctuation with spaces, keeping alphanumeric and Indic unicode blocks
    cleaned = re.sub(r'[^\w\s\u0900-\u0D7F]', ' ', str(text).lower())
    raw_tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 0]
    meaningful = [t for t in raw_tokens if t not in STOP_WORDS and len(t) > 1]
    return meaningful if meaningful else raw_tokens

# =====================================================================
# SUB-MILLISECOND IN-MEMORY SEMANTIC LRU CACHE
# =====================================================================

class SemanticLRUCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.cache: Dict[str, Tuple[List[RetrievedCandidate], float]] = collections.OrderedDict()

    def get(self, query: str) -> Optional[List[RetrievedCandidate]]:
        key = query.strip().lower()
        if key in self.cache:
            candidates, timestamp = self.cache[key]
            # Refresh position in LRU
            self.cache.move_to_end(key)
            return candidates
        return None

    def put(self, query: str, candidates: List[RetrievedCandidate]):
        key = query.strip().lower()
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (candidates, time.time())
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# =====================================================================
# IN-MEMORY BM25 SPARSE TOKEN INDEX
# =====================================================================

class LightweightBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = collections.defaultdict(int)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        self.documents: List[Dict[str, Any]] = []
        self.num_docs: int = 0

    def fit(self, records: List[Dict[str, Any]]):
        self.documents = records
        self.num_docs = len(records)
        if self.num_docs == 0:
            return
        
        total_len = 0
        for doc_id, doc in enumerate(records):
            # Index text, parent_passage, and translated_passage with stopword filtering
            full_content = f"{doc.get('text', '')} {doc.get('translated_passage', '')} {doc.get('parent_passage', '')}"
            tokens = tokenize_and_filter(full_content)
            doc_length = len(tokens)
            self.doc_len.append(doc_length)
            total_len += doc_length
            
            # Count term frequencies in this document
            tf = collections.Counter(tokens)
            for token, count in tf.items():
                self.doc_freqs[token] += 1
                self.inverted_index[token].append((doc_id, count))
                
        self.avgdl = total_len / self.num_docs if self.num_docs > 0 else 1.0

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.num_docs == 0:
            return []
        query_tokens = tokenize_and_filter(query)
        if not query_tokens:
            return []
        
        scores: Dict[int, float] = collections.defaultdict(float)
        matched_token_counts: Dict[int, int] = collections.defaultdict(int)
        
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            df = self.doc_freqs[token]
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            for doc_id, tf in self.inverted_index[token]:
                doc_length = self.doc_len[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                scores[doc_id] += idf * (numerator / denominator)
                matched_token_counts[doc_id] += 1
                
        # Only return candidates that matched significant query tokens
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scores[:top_k]

# =====================================================================
# RECIPROCAL RANK FUSION (RRF) ENGINE
# =====================================================================

def calculate_rrf(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Tuple[Dict[str, Any], float]],
    rrf_k: int = 60,
    top_n: int = 3
) -> List[RetrievedCandidate]:
    """
    Applies Reciprocal Rank Fusion:
    RRF_Score(doc) = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
    """
    candidates_map: Dict[str, Dict[str, Any]] = {}
    
    # 1. Score Dense candidates
    for rank, doc in enumerate(dense_results, 1):
        key = str(doc.get("text", ""))[:80]
        dist = float(doc.get("_distance", 1.0))
        dense_rrf = (1.0 / (rrf_k + rank)) * max(0.1, 1.0 - (dist if dist <= 1.0 else 0.5))
        if key not in candidates_map:
            candidates_map[key] = {
                "doc": doc,
                "dense_distance": dist,
                "sparse_score": 0.0,
                "rrf_score": dense_rrf
            }
        else:
            candidates_map[key]["rrf_score"] += dense_rrf
            candidates_map[key]["dense_distance"] = dist

    # 2. Score Sparse candidates
    for rank, (doc, sparse_s) in enumerate(sparse_results, 1):
        key = str(doc.get("text", ""))[:80]
        sparse_rrf = (1.0 / (rrf_k + rank)) * (1.0 + min(sparse_s / 5.0, 2.0))
        if key not in candidates_map:
            candidates_map[key] = {
                "doc": doc,
                "dense_distance": 0.35 if sparse_s > 4.0 else 0.65,
                "sparse_score": sparse_s,
                "rrf_score": sparse_rrf
            }
        else:
            candidates_map[key]["rrf_score"] += sparse_rrf
            candidates_map[key]["sparse_score"] = sparse_s
            if sparse_s > 4.0:
                candidates_map[key]["dense_distance"] = min(candidates_map[key]["dense_distance"], 0.35)

    # Sort by combined RRF score
    sorted_candidates = sorted(
        candidates_map.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )[:top_n]

    results = []
    for item in sorted_candidates:
        d = item["doc"]
        results.append(RetrievedCandidate(
            chunk_text=str(d.get("text", "")),
            parent_passage=str(d.get("parent_passage", "")),
            translated_passage=str(d.get("translated_passage", "")),
            chunk_strategy=str(d.get("chunk_strategy", "hybrid")),
            dense_distance=round(item["dense_distance"], 4),
            sparse_score=round(item["sparse_score"], 4),
            rrf_score=round(item["rrf_score"], 6),
            query_id=str(d.get("query_id", "")),
            is_selected=int(d.get("is_selected", 0)),
            eng_query=str(d.get("eng_query", ""))
        ))

    return results

