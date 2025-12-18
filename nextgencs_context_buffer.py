# nextgencs_context_buffer.py - Production context buffer → pyvegas
import asyncio
import json
import time
import random
import re
from collections import Counter
from typing import List, Dict, Any
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
nltk.download('punkt', quiet=True)

nlp = spacy.load("en_core_web_sm")

class NextGenCSContextBuffer:
    """Production context buffer for live Verizon transcripts."""
    
    def __init__(self, pyvegas_client, max_window=6, coherence_threshold=0.68):
        self.buffer: List[Dict] = []
        self.tfidf_vectors: List[np.ndarray] = []
        self.keyword_history = Counter()
        self.pyvegas_client = pyvegas_client
        self.max_window = max_window
        self.coherence_threshold = coherence_threshold
        self.last_activity = time.time()
        self.tfidf = TfidfVectorizer(max_features=500, stop_words='english', lowercase=True)
        
    def add_utterance(self, utterance: str, speaker: str = "unknown") -> None:
        """Live SSE ingestion point."""
        timestamp = time.time()
        utterance = self._clean_utterance(utterance)
        
        if len(self.buffer) >= self.max_window:
            asyncio.create_task(self._flush_to_pyvegas())
            return
        
        keywords = self._extract_keywords(utterance)
        self.keyword_history.update(keywords)
        
        entry = {
            "text": utterance,
            "speaker": speaker,
            "timestamp": timestamp,
            "keywords": keywords
        }
        
        self.buffer.append(entry)
        self.last_activity = timestamp
        
        self._update_tfidf()
        asyncio.create_task(self._check_flush())
    
    async def _flush_to_pyvegas(self) -> Dict[str, Any]:
        """Route complete context to pyvegas tool discovery."""
        if not self.buffer:
            return {}
        
        context_block = self._build_context_block()
        
        print(f"\n🚀 PYVEGAS ROUTING:\n{context_block}\n")
        
        # PRODUCTION: Route to pyvegas with tool discovery enabled
        result = await self.pyvegas_client.infer_with_tools(
            context=context_block,
            usecase_name="nextgencs_agent_assist",
            context_name="live_call_context"
        )
        
        print(f"✅ TOOL RESPONSE: {result.get('nudge', 'No nudge')}")
        print(f"🔧 Tools invoked: {result.get('tools_used', [])}")
        
        self._reset_buffer()
        return result
    
    def _build_context_block(self) -> str:
        """Smart context assembly."""
        top_keywords = [kw for kw, _ in self.keyword_history.most_common(5)]
        weighted_context = []
        
        for utt in self.buffer:
            relevance = len(set(utt["keywords"]) & set(top_keywords)) / max(len(utt["keywords"]), 1)
            if relevance > 0.3:
                weighted_context.append(utt["text"])
        
        return " ".join(weighted_context)
    
    # [Previous lightweight methods unchanged for brevity]
    def _clean_utterance(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip())
    
    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        domain_boost = {'bill', 'charge', 'call', 'phone', 'device', 'signal', 'promo', 'account'}
        return [w for w, _ in Counter(words).most_common(8) if w in domain_boost or len(words) > 3]
    
    def _update_tfidf(self):
        if len(self.buffer) >= 2:
            texts = [u["text"] for u in self.buffer[-3:]]
            self.tfidf_vectors = self.tfidf.transform(texts).toarray().tolist()
    
    async def _check_flush(self):
        await asyncio.sleep(0.3)
        reasons = self._get_flush_reasons()
        if reasons:
            await self._flush_to_pyvegas()
    
    def _get_flush_reasons(self) -> List[str]:
        reasons = []
        # [Simplified trigger logic from previous POC]
        if len(self.buffer) >= 3:
            reasons.append("context_complete")
        return reasons
    
    def _reset_buffer(self):
        self.buffer.clear()
        self.tfidf_vectors.clear()
        self.keyword_history.clear()

# Usage: Integrate with your SSE stream
async def main():
    from pyvegas_template import PyVegasAgent  # File 2
    client = PyVegasAgent(usecase_name="nextgencs", context_name="live_context", apiKey="your-key")
    buffer = NextGenCSContextBuffer(client)
    
    # Simulate SSE
    utterances = [
        "Customer: I want explanation why my december bill went up",
        "Agent: Sure let me pull up your account",
        "Customer: Yeah last month was only 89 dollars"
    ]
    
    for utt in utterances:
        speaker = "Customer" if "Customer" in utt else "Agent"
        buffer.add_utterance(utt, speaker)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
