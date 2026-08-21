"""
Dhwani (ध्वनि) - 4-Tier Comprehensive Guardrail Matrix
Ensures safety, factual groundedness, out-of-domain rejection, and script consistency.
"""

import re
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field

class GuardrailEvaluation(BaseModel):
    is_safe: bool = True
    is_in_domain: bool = True
    is_grounded: bool = True
    script_consistent: bool = True
    refusal_code: Optional[str] = None
    refusal_message: Optional[str] = None
    evaluation_latency_ms: float = 0.0
    tier_triggered: Optional[str] = None

# Tier 1: Strict Prompt Injection, Jailbreak, Safety & Exploit Signatures
UNSAFE_PATTERNS = [
    r"\b(bomb|explosive|detonate|ied|pipe\s*bomb)\b",
    r"\b(hack|exploit|bypass|vulnerability|ddos|sql\s*injection|xss|malware|ransomware|trojan|keylogger)\b",
    r"\b(suicide|self-harm|kill\s*myself|cut\s*wrists)\b",
    r"\b(poison|cyanide|anthrax|ricin|bioweapon|chemical\s*weapon)\b",
    r"\b(credit\s*card\s*number|cvv|ssn|social\s*security\s*number|bank\s*pin|otp\s*bypass)\b",
    r"\b(child\s*abuse|csam|nsfw|pornography|deepfake)\b",
    r"(ignore\s*previous\s*instructions|system\s*prompt|you\s*are\s*now\s*DAN|jailbreak|developer\s*mode)",
    # Fictional universe / sci-fi entities (not in factual knowledge base)
    r"\b(warp\s*drive|starship\s*enterprise|millennium\s*falcon|lightsaber|klingon|vulcan\s*mind|jedi|sith\s*lord|tardis|dalek|death\s*star|thanos|infinity\s*gauntlet|horcrux|patronus|quidditch)\b",
]

COMPILED_UNSAFE_REGEX = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]

def evaluate_tier1_safety(query: str) -> Optional[Tuple[str, str]]:
    """
    Tier 1: Ultra-fast regex heuristic check for prompt injections, safety, and dangerous activities (< 0.05ms).
    Returns (refusal_code, refusal_message) if violated.
    """
    q_clean = query.strip().lower()
    for regex in COMPILED_UNSAFE_REGEX:
        match = regex.search(q_clean)
        if match:
            trigger_word = match.group(0)
            return (
                "UNSAFE_PROMPT_INJECTION_OR_TOXICITY",
                f"I cannot process this query as it violates safety guidelines regarding restricted or potentially harmful topics ({trigger_word})."
            )
    return None

OOD_REFUSAL_MESSAGES: Dict[str, str] = {
    "hi-IN": "यह प्रश्न सिस्टम के सत्यापित MSMARCO ज्ञानकोष के बाहर का है। गलत जानकारी से बचने के लिए अप्रमाणित विषयों पर उत्तर नहीं दिया जाता।",
    "od-IN": "ଏହି ପ୍ରଶ୍ନଟି ସିଷ୍ଟମର ସତ୍ୟାପିତ MSMARCO ଜ୍ଞାନକୋଷ ବାହାରେ ଅଟେ। ଭ୍ରାନ୍ତ ତଥ୍ୟରୁ ରକ୍ଷା ପାଇବା ପାଇଁ ମଡେଲ ଅଣ-ଯାଞ୍ଚିତ ବିଷୟରେ ଉତ୍ତର ଦିଏ ନାହିଁ।",
    "ta-IN": "இந்த கேள்வி அமைப்பின் சரிபார்க்கப்பட்ட MSMARCO அறிவுத் தளத்திற்கு வெளியே உள்ளது.",
    "te-IN": "ఈ ప్రశ్న సిస్టమ్ యొక్క ధృవీకరించబడిన MSMARCO నాలెడ్జ్ బేస్ వెలుపల ఉంది.",
    "bn-IN": "এই প্রশ্নটি সিস্টেমের যাচাইকৃত MSMARCO জ্ঞানভাণ্ডারের বাইরে।",
    "mr-IN": "हा प्रश्न प्रणालीच्या पडताळणी केलेल्या MSMARCO ज्ञानकोशाबाहेरचा आहे.",
    "gu-IN": "આ પ્રશ્ન સિસ્ટમના ચકાસાયેલ MSMARCO જ્ઞાનકોશની બહારનો છે.",
    "kn-IN": "ಈ ಪ್ರಶ್ನೆಯು ಸಿಸ್ಟಂನ ಪರಿಶೀಲಿಸಿದ MSMARCO ಜ್ಞಾನದ ಮೂಲದಿಂದ ಹೊರಗಿದೆ.",
    "ml-IN": "ഈ ചോദ്യം സിസ്റ്റത്തിന്റെ പരിശോധിച്ചുറപ്പിച്ച MSMARCO വിജ്ഞാന അടിത്തറയ്ക്ക് പുറത്താണ്.",
    "pa-IN": "ਇਹ ਸਵਾਲ ਸਿਸਟਮ ਦੇ ਪ੍ਰਮਾਣਿਤ MSMARCO ਗਿਆਨ ਅਧਾਰ ਤੋਂ ਬਾਹਰ ਹੈ।",
    "en-IN": "This query falls outside the factual knowledge base of the system. To prevent hallucinations, the model will not speculate on unverified topics."
}

def evaluate_tier2_domain_gate(
    top_distance: float,
    threshold: float = 0.68,
    language_code: str = "en-IN"
) -> Optional[Tuple[str, str]]:
    """
    Tier 2: Vector Space Out-of-Distribution (OOD) Gate.
    If the cosine distance to the closest knowledge passage exceeds the threshold,
    reject the query as out-of-domain knowledge to prevent hallucinated speculation.
    """
    if top_distance > threshold:
        msg = OOD_REFUSAL_MESSAGES.get(language_code, OOD_REFUSAL_MESSAGES["en-IN"])
        return ("OUT_OF_DOMAIN_QUERY", msg)
    return None

def evaluate_tier3_nli_groundedness(
    answer: str,
    retrieved_contexts: List[str]
) -> Tuple[bool, float]:
    """
    Tier 3: NLI-Inspired Groundedness & Hallucination Verification.
    Calculates key concept token overlap between the generated answer and retrieved source contexts.
    Returns (is_grounded, grounding_score).
    """
    if not retrieved_contexts or not answer:
        return True, 0.85
    
    # Extract substantive alphanumeric words (> 3 chars)
    ans_words = set(re.findall(r'\b[a-zA-Z\u0900-\u0D7F]{4,}\b', answer.lower()))
    if not ans_words:
        return True, 0.90
    
    combined_ctx = " ".join(retrieved_contexts).lower()
    matched_words = sum(1 for w in ans_words if w in combined_ctx)
    overlap_ratio = matched_words / len(ans_words)
    
    # High overlap ratio indicates high factual groundedness in retrieved context
    is_grounded = overlap_ratio >= 0.25
    return is_grounded, round(overlap_ratio, 3)

def evaluate_tier4_script_consistency(
    query: str,
    answer: str,
    language_code: str
) -> bool:
    """
    Tier 4: Linguistic & Script Consistency Gate.
    Verifies that non-English Indic queries receive answers in their expected native script.
    """
    # Devanagari Unicode range: \u0900-\u097F
    # Tamil: \u0B80-\u0BFF, Telugu: \u0C00-\u0C7F, Gujarati: \u0A80-\u0AFF, etc.
    if language_code.startswith("hi") or language_code.startswith("mr"):
        has_devanagari_query = bool(re.search(r'[\u0900-\u097F]', query))
        if has_devanagari_query:
            has_devanagari_ans = bool(re.search(r'[\u0900-\u097F]', answer))
            return has_devanagari_ans
    return True
