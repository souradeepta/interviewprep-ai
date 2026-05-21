"""
Auto-generated from 06-context-distillation.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Context Distillation
# ## Learning Objectives
# 1. Understand extractive vs abstractive context compression techniques
# 2. Implement importance scoring and document selection from scratch
# 3. Build multi-document distillation pipelines with quality metrics
# 4. Apply context distillation to real-world LLM scenarios (long documents, transcripts, retrieval)
# ======================================================================

import numpy as np
import torch
import time
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: Basic Context Distillation (Extractive)
# Extract the most important sentences from a document using attention-based importance scoring.
# ======================================================================

# Level 1: Extractive distillation with importance scoring (from scratch)

class BasicContextDistiller:
    """Extract important sentences based on attention-like weights"""
    
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
    
    def compute_importance_scores(self, sentences: List[str]) -> np.ndarray:
        """
        Compute importance of each sentence using:
        1. Word frequency (TF-like)
        2. Position bias (earlier sentences weighted higher)
        3. Length normalization
        """
        n_sentences = len(sentences)
        scores = np.zeros(n_sentences)
        
        # Build word frequency
        word_freq = {}
        for sent in sentences:
            for word in sent.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score each sentence
        for i, sent in enumerate(sentences):
            # Word frequency component
            word_score = sum(word_freq.get(w.lower(), 0) for w in sent.split())
            # Position component (earlier = slightly higher)
            pos_score = 1.0 / (1.0 + 0.01 * i)
            # Length normalization
            length_norm = len(sent.split()) / (max(len(s.split()) for s in sentences) + 1)
            
            scores[i] = word_score * pos_score * length_norm
        
        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def distill(self, text: str) -> Tuple[str, float]:
        """Distill text to important sentences"""
        # Split into sentences (simple heuristic)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Compute importance
        scores = self.compute_importance_scores(sentences)
        
        # Select top-k sentences
        n_keep = max(1, int(len(sentences) * self.compression_ratio))
        top_indices = np.argsort(scores)[-n_keep:][::-1]
        top_indices.sort()  # Maintain original order
        
        # Reconstruct text
        distilled_text = '. '.join([sentences[i] for i in top_indices]) + '.'
        compression_ratio = len(distilled_text) / len(text)
        
        return distilled_text, compression_ratio

# Test with synthetic document
sample_doc = """
Machine learning is transforming industries. Deep learning models require massive datasets. 
Data preprocessing is crucial for model performance. Feature engineering helps models learn better. 
Transformers revolutionized NLP. Attention mechanisms are fundamental. Context matters for understanding. 
Large language models show emergent capabilities.
"""

distiller = BasicContextDistiller(compression_ratio=0.5)
distilled, ratio = distiller.distill(sample_doc)

print("Original document:")
print(sample_doc.strip())
print(f"\nDistilled document:")
print(distilled)
print(f"\nCompression ratio: {ratio:.2%}")


# Visualize importance scores
sentences = [s.strip() for s in sample_doc.split('.') if s.strip()]
scores = distiller.compute_importance_scores(sentences)

plt.figure(figsize=(12, 4))
plt.bar(range(len(sentences)), scores)
plt.xlabel('Sentence Index')
plt.ylabel('Importance Score')
plt.title('Sentence Importance Scores (Basic Distillation)')
plt.xticks(range(len(sentences)), [f'S{i}' for i in range(len(sentences))])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Selected {int(len(sentences) * 0.5)} out of {len(sentences)} sentences')


# ======================================================================
# ## Level 2: Advanced Multi-Document Distillation with Quality Metrics
# Add abstractive distillation, token pruning, and semantic similarity preservation metrics.
# ======================================================================

# Level 2: Advanced distillation with abstractive + extractive + quality metrics

class AdvancedContextDistiller:
    """Multi-document distillation with quality metrics"""
    
    def __init__(self, compression_ratio=0.3, abstractive_ratio=0.3):
        self.compression_ratio = compression_ratio
        self.abstractive_ratio = abstractive_ratio  # Fraction of output to abstract
    
    def compute_tf_idf_scores(self, sentences: List[str]) -> np.ndarray:
        """TF-IDF inspired scoring across documents"""
        n_sentences = len(sentences)
        scores = np.zeros(n_sentences)
        
        # Term frequency
        all_words = set()
        word_in_sent = {}  # Count sentences containing word
        
        for sent in sentences:
            words = set(w.lower() for w in sent.split())
            all_words.update(words)
            for w in words:
                word_in_sent[w] = word_in_sent.get(w, 0) + 1
        
        # Score sentences
        for i, sent in enumerate(sentences):
            tf_score = len(sent.split())  # Simple TF
            # IDF: weight words that appear in few sentences
            idf_score = sum(np.log(n_sentences / (word_in_sent.get(w.lower(), 1) + 1)) 
                           for w in sent.split() if w.lower() in word_in_sent)
            scores[i] = tf_score * idf_score
        
        return scores / (scores.max() + 1e-8)
    
    def semantic_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Simple bag-of-words similarity between sentences.
        In production, use transformers embeddings.
        """
        n = len(sentences)
        similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                words_i = set(w.lower() for w in sentences[i].split())
                words_j = set(w.lower() for w in sentences[j].split())
                
                if len(words_i) == 0 or len(words_j) == 0:
                    similarity[i, j] = 0
                else:
                    intersection = len(words_i & words_j)
                    union = len(words_i | words_j)
                    similarity[i, j] = intersection / union  # Jaccard similarity
        
        return similarity
    
    def distill_multi_document(self, documents: List[str]) -> Dict:
        """Distill multiple documents, select important ones"""
        all_sentences = []
        doc_boundaries = []
        
        # Flatten with boundaries
        for doc in documents:
            sents = [s.strip() for s in doc.split('.') if s.strip()]
            doc_boundaries.append(len(all_sentences))
            all_sentences.extend(sents)
        
        # Score sentences
        scores = self.compute_tf_idf_scores(all_sentences)
        
        # Compute similarity to avoid redundancy
        similarity = self.semantic_similarity_matrix(all_sentences)
        
        # Greedy selection: pick diverse, important sentences
        selected = []
        remaining = set(range(len(all_sentences)))
        n_select = max(1, int(len(all_sentences) * self.compression_ratio))
        
        while len(selected) < n_select and remaining:
            # Find highest score not in selected
            best_idx = max(remaining, key=lambda i: scores[i])
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            # Remove similar sentences
            redundant = [i for i in remaining if similarity[best_idx, i] > 0.7]
            remaining -= set(redundant)
        
        selected.sort()
        distilled_text = '. '.join([all_sentences[i] for i in selected]) + '.'
        
        # Compute metrics
        original_text = ' '.join(documents)
        orig_words = set(w.lower() for w in original_text.split())
        dist_words = set(w.lower() for w in distilled_text.split())
        overlap = len(orig_words & dist_words)
        coverage = overlap / len(orig_words) if orig_words else 0
        
        metrics = {
            'compression_ratio': len(distilled_text) / len(original_text),
            'word_coverage': coverage,
            'information_loss': 1.0 - coverage,
            'selected_sentences': len(selected),
            'total_sentences': len(all_sentences)
        }
        
        return {
            'distilled_text': distilled_text,
            'metrics': metrics,
            'selected_indices': selected
        }

# Test multi-document distillation
docs = [
    "Machine learning requires data. Feature engineering is critical. Preprocessing takes time.",
    "Transformers changed NLP. Attention mechanisms work well. Self-attention is powerful.",
    "Context windows are limited. Token limits affect models. Compression saves costs."
]

advanced_distiller = AdvancedContextDistiller(compression_ratio=0.35)
result = advanced_distiller.distill_multi_document(docs)

print("Distilled output:")
print(result['distilled_text'])
print("\nMetrics:")
for key, val in result['metrics'].items():
    if isinstance(val, float):
        print(f"  {key}: {val:.3f}")
    else:
        print(f"  {key}: {val}")


# Visualize similarity matrix
all_sents = []
for doc in docs:
    all_sents.extend([s.strip() for s in doc.split('.') if s.strip()])

similarity = advanced_distiller.semantic_similarity_matrix(all_sents)

plt.figure(figsize=(10, 8))
plt.imshow(similarity, cmap='YlOrRd')
plt.colorbar(label='Similarity (Jaccard)')
plt.xlabel('Sentence Index')
plt.ylabel('Sentence Index')
plt.title('Semantic Similarity Matrix (Multi-Document)')
plt.tight_layout()
plt.show()

print(f"Max similarity between different sentences: {similarity[similarity < 1.0].max():.3f}")
print(f"Selected {result['metrics']['selected_sentences']} out of {result['metrics']['total_sentences']} sentences")


# ======================================================================
# ## Real-World Example 1: Distill Customer Support Transcripts with Semantic Embeddings
# Use semantic embeddings to find important sentences in support conversations.
# ======================================================================

# Real-World Example 1: Support transcript distillation

class TranscriptDistiller:
    """Distill support transcripts using semantic embeddings"""
    
    def __init__(self, compression_ratio=0.4):
        self.compression_ratio = compression_ratio
        self.use_embeddings = False
    
    def mock_embeddings(self, texts: List[str]) -> np.ndarray:
        """Mock embeddings for demo (production: use real model)"""
        embeddings = []
        for text in texts:
            # Simple hash-based embedding for reproducibility
            vec = np.array([hash(w) % 100 for w in text.split()[:5]])
            vec = np.pad(vec, (0, max(0, 384 - len(vec))), mode='constant')
            embeddings.append(vec[:384] / (np.linalg.norm(vec[:384]) + 1e-8))
        return np.array(embeddings)
    
    def distill_transcript(self, transcript: str) -> Dict:
        """
        Distill transcript using:
        1. Semantic importance (distance from centroid)
        2. Temporal importance (later turns matter more in support)
        3. Diversity (don't select similar utterances)
        """
        turns = [t.strip() for t in transcript.split('\n') if t.strip()]
        
        # Get embeddings
        embeddings = self.mock_embeddings(turns)
        centroid = embeddings.mean(axis=0)
        
        # Compute scores
        scores = np.zeros(len(turns))
        
        for i, emb in enumerate(embeddings):
            # Distance from centroid (important = far from average)
            semantic_score = np.linalg.norm(emb - centroid)
            # Temporal weight (later turns important)
            temporal_score = 1.0 + (i / len(turns))
            scores[i] = semantic_score * temporal_score
        
        # Greedy selection with diversity constraint
        selected = []
        remaining = set(range(len(turns)))
        n_keep = max(1, int(len(turns) * self.compression_ratio))
        
        while len(selected) < n_keep and remaining:
            best_idx = max(remaining, key=lambda i: scores[i])
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            # Remove similar
            best_emb = embeddings[best_idx]
            similar = [i for i in remaining 
                       if np.dot(embeddings[i], best_emb) > 0.8]
            remaining -= set(similar)
        
        selected.sort()
        distilled = '\n'.join([turns[i] for i in selected])
        
        return {
            'distilled': distilled,
            'compression_ratio': len(distilled) / len(transcript),
            'turns_kept': len(selected),
            'turns_total': len(turns)
        }

# Test with mock support transcript
transcript = """Customer: Hi, I have an issue with my account
Agent: Hello! I'm here to help. What seems to be the problem?
Customer: I can't log in to my account
Agent: Let me help you with that. Can you confirm your email?
Customer: It's user@example.com
Agent: Thanks. I see your account. Have you tried resetting your password?
Customer: Yes, but I'm not receiving the reset email
Agent: Let me check your email settings. It might be in spam.
Customer: Found it! But the link is expired
Agent: I'll send a new reset link right away.
Customer: Great, I can log in now!
Agent: Excellent! Is there anything else I can help with?"""

distiller = TranscriptDistiller(compression_ratio=0.5)
result = distiller.distill_transcript(transcript)

print("Original transcript:")
print(transcript[:200] + "...")
print("\n" + "="*50)
print("Distilled transcript:")
print(result['distilled'])
print(f"\nCompression: {result['turns_kept']}/{result['turns_total']} turns kept")
print(f"Size reduction: {result['compression_ratio']:.1%}")


# ======================================================================
# ## Real-World Example 2: Abstractive Summarization via Key Sentences
# Extract key sentences to create abstractive-like summaries from long contexts.
# ======================================================================

# Real-World Example 2: Abstractive-style distillation

class AbstractiveSummarizer:
    """Create abstractive-style summaries via key sentence extraction"""
    
    def __init__(self, max_length=100):
        self.max_length = max_length
    
    def extract_key_sentences(self, text: str, n_sentences=3) -> str:
        """Extract key sentences using word frequency"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Score by word frequency
        word_freq = {}
        for sent in sentences:
            for word in sent.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        scores = [sum(word_freq.get(w.lower(), 0) for w in s.split()) 
                  for s in sentences]
        
        top_indices = np.argsort(scores)[-n_sentences:][::-1]
        top_indices.sort()
        
        return '. '.join([sentences[i] for i in top_indices]) + '.'
    
    def summarize(self, text: str) -> Dict:
        """Create summary via key sentence extraction"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 3:
            return {
                'summary': text,
                'compression_ratio': 1.0,
                'method': 'original (too short to compress)'
            }
        
        # Extract key sentences
        n_key = max(1, len(sentences) // 2)
        summary = self.extract_key_sentences(text, n_sentences=n_key)
        
        return {
            'summary': summary,
            'compression_ratio': len(summary) / len(text),
            'method': 'extractive key-sentences',
            'original_length': len(text),
            'summary_length': len(summary),
            'sentences_kept': n_key
        }

# Test abstractive summarization
long_text = """Deep learning has revolutionized artificial intelligence over the past decade. 
The transformer architecture, introduced in 2017, became the foundation for large language models. 
These models require massive amounts of training data and computational resources. 
Context windows in transformers are limited, typically ranging from 2K to 200K tokens. 
This limitation makes context compression crucial for long-document tasks. 
Context distillation reduces computational costs and latency in production systems. 
Extractive methods preserve original text while abstractive methods generate new summaries. 
Both approaches have trade-offs in terms of information preservation and compression ratio."""

summarizer = AbstractiveSummarizer(max_length=80)
result = summarizer.summarize(long_text)

print("Original text:")
print(long_text)
print(f"\nSummary ({result['method']}):")
print(result['summary'])
print(f"\nCompression: {result['original_length']} → {result['summary_length']} chars")
print(f"Ratio: {result['compression_ratio']:.1%}")
print(f"Sentences: {result['sentences_kept']} kept")


# ======================================================================
# ## Real-World Example 3: Context Window Optimizer for Long-Context Models
# Compress context for long-context models to fit within token limits while preserving critical information.
# ======================================================================

# Real-World Example 3: Context window optimization

class ContextWindowOptimizer:
    """
    Fit contexts into token limits while preserving information.
    Useful for: RAG systems, long-document QA, multi-turn conversations.
    """
    
    def __init__(self, max_tokens=512, chars_per_token=4):
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token  # Rough estimate
        self.max_chars = max_tokens * chars_per_token
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: ~4 chars per token)"""
        return len(text) // self.chars_per_token
    
    def optimize(self, context: str, query: str = None) -> Dict:
        """
        Optimize context to fit token limit while maintaining relevance to query.
        Strategy:
        1. Score sentences by relevance to query (if provided)
        2. Greedily select until token limit
        3. Maintain order for coherence
        """
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        current_tokens = self.estimate_tokens(context)
        
        if current_tokens <= self.max_tokens:
            return {
                'optimized_context': context,
                'compression_needed': False,
                'original_tokens': current_tokens,
                'optimized_tokens': current_tokens,
                'compression_ratio': 1.0
            }
        
        # Scoring function
        scores = np.ones(len(sentences))
        
        if query:
            # Score by relevance to query
            query_words = set(w.lower() for w in query.split())
            for i, sent in enumerate(sentences):
                sent_words = set(w.lower() for w in sent.split())
                overlap = len(query_words & sent_words)
                scores[i] = overlap + 0.1  # Small baseline
        
        # Add position bonus (context order matters)
        position_bonus = np.linspace(0, 0.5, len(sentences))
        scores += position_bonus
        
        # Greedy selection by score, respecting order
        selected_indices = []
        current_length = 0
        
        # Select high-scoring sentences in original order
        scored_sents = [(i, sentences[i], scores[i]) for i in range(len(sentences))]
        scored_sents.sort(key=lambda x: -x[2])  # Sort by score
        
        for idx, sent, score in scored_sents:
            sent_tokens = self.estimate_tokens(sent)
            if current_length + sent_tokens <= self.max_tokens:
                selected_indices.append(idx)
                current_length += sent_tokens
        
        selected_indices.sort()  # Restore original order
        optimized = '. '.join([sentences[i] for i in selected_indices]) + '.'
        
        return {
            'optimized_context': optimized,
            'compression_needed': True,
            'original_tokens': current_tokens,
            'optimized_tokens': self.estimate_tokens(optimized),
            'compression_ratio': self.estimate_tokens(optimized) / current_tokens,
            'sentences_selected': len(selected_indices),
            'sentences_total': len(sentences)
        }

# Test with long document
long_doc = """Transformer models fundamentally changed NLP. The self-attention mechanism allows the model to weight different parts of the input differently. Unlike RNNs, transformers can process sequences in parallel, making training much faster. 
GPT models are autoregressive transformers trained on next-token prediction. They learn to generate coherent text by predicting one token at a time. 
BERT uses bidirectional attention and masked language modeling for pretraining. These architectural differences lead to different strengths: GPT excels at generation, BERT at understanding. 
Context windows limit how much information a model can consider at once. Typical limits range from 2K to 128K tokens. 
Long context windows enable better performance on document understanding and complex reasoning tasks."""

query = "how do transformers use attention?"

optimizer = ContextWindowOptimizer(max_tokens=100, chars_per_token=4)
result = optimizer.optimize(long_doc, query)

print(f"Query: {query}")
print(f"\nOriginal context: {result['original_tokens']} tokens")
print(f"Optimized context: {result['optimized_tokens']} tokens")
print(f"Compression: {result['compression_ratio']:.1%}")
print(f"Sentences: {result['sentences_selected']}/{result['sentences_total']}")
print(f"\nOptimized context:")
print(result['optimized_context'])


# ======================================================================
# ## Comparison: Compression Ratio vs Information Loss
# Benchmark different distillation approaches on compression-quality trade-off.
# ======================================================================

# Comparison: Different distillation methods

test_doc = """Artificial intelligence is transforming industries worldwide. Machine learning models power recommendation systems. Deep learning drives computer vision applications. Natural language processing enables chatbots and translation services. Reinforcement learning trains agents for game playing and robotics. Transfer learning allows models to leverage pretraining. Fine-tuning adapts models to specific tasks. Quantization reduces model sizes for deployment. Pruning removes unnecessary parameters. Knowledge distillation compresses large models into smaller ones."""

ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
compressions = []
information_losses = []

for ratio in ratios:
    # Level 1 (basic)
    basic = BasicContextDistiller(compression_ratio=ratio)
    dist, comp_ratio = basic.distill(test_doc)
    compressions.append(comp_ratio)
    
    # Info loss: 1 - word_coverage
    orig_words = set(w.lower() for w in test_doc.split())
    dist_words = set(w.lower() for w in dist.split())
    coverage = len(orig_words & dist_words) / len(orig_words)
    information_losses.append(1.0 - coverage)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Compression vs ratio
ax1.plot(ratios, compressions, 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.set_xlabel('Desired Compression Ratio')
ax1.set_ylabel('Actual Compression Ratio')
ax1.set_title('Compression Effectiveness')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.15, 0.85)

# Information loss
ax2.plot(compressions, information_losses, 'o-', linewidth=2, markersize=8, color='coral')
ax2.set_xlabel('Compression Ratio')
ax2.set_ylabel('Information Loss (1 - Word Coverage)')
ax2.set_title('Compression-Quality Trade-off')
ax2.grid(True, alpha=0.3)
ax2.fill_between(compressions, information_losses, alpha=0.2, color='coral')

plt.tight_layout()
plt.show()

# Summary table
print("\nDistillation Methods Comparison:")
print(f"{'Ratio':<8} {'Compression':<15} {'Info Loss':<15}")
print("-" * 40)
for r, c, l in zip(ratios, compressions, information_losses):
    print(f"{r:<8.1f} {c:<15.1%} {l:<15.1%}")


# ======================================================================
# ## Key Takeaways
# **Core Idea:** Context distillation reduces token usage while preserving critical information through extractive (sentence selection) or abstractive (summary generation) methods.
# **Approaches and When to Use:**
# | Method | Speed | Compression | Info Preservation | Best For |
# |--------|-------|-------------|------------------|----------|
# | Extractive (TF-IDF) | Very Fast | 30-50% | 80-95% | Real-time, bounded latency |
# | Semantic (Embeddings) | Medium | 40-60% | 85-95% | Quality-focused, offline |
# | Abstractive (Models) | Slow | 50-80% | 70-85% | Content generation, summarization |
# | Token Pruning | Fast | 20-40% | 90-98% | Light compression, quick wins |
# **Common Failure Modes:**
# - **Information Bottleneck:** Distilling too aggressively (>70%) loses crucial details. Fix: Validate compression ratio empirically on downstream tasks.
# - **Redundancy Overfitting:** Selecting similar sentences wastes tokens. Fix: Use semantic diversity constraints (MMR, MMD).
# - **Order Sensitivity:** Extractive methods break coherence if context order matters. Fix: Preserve original sentence order when reconstructing.
# **Related Concepts:**
# - [Retrieval-Augmented Generation (RAG)](./05-retrieval-augmented-generation.ipynb) – Uses distillation in retrieval pipelines
# - [Quantization & Pruning](./12-quantization-pruning.ipynb) – Compress models similarly
# - [Long-Context Modeling](./04-long-context-modeling.ipynb) – Handles larger context windows efficiently
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Experiment with compression ratios:** Run the Advanced Distiller with ratios 0.2, 0.4, 0.6 and compare information loss.
# 2. **Compare extractive vs abstractive:** Apply both BasicContextDistiller and AbstractiveSummarizer to the same document and compare outputs.
# 3. **Optimize for your context window:** Use ContextWindowOptimizer with different token limits and observe quality trade-offs.
# ======================================================================
