#!/usr/bin/env python3
"""
Automatic Semantic Shift Detection System

This system automatically detects potential semantic shifts by analyzing:
1. Frequency changes over time periods
2. Context vector divergence using word embeddings
3. Co-occurrence pattern changes
4. Statistical significance testing

Based on research methods from:
- Hamilton et al. (2016): Statistical laws of semantic change
- Kulkarni et al. (2014): Statistical approaches to historical linguistics
- Kim et al. (2014): Temporal analysis of semantic change
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy import stats
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AutoDetectedShift:
    """Represents an automatically detected semantic shift candidate."""
    word: str
    shift_score: float
    significance: float
    time_periods: Tuple[str, str]
    context_change: float
    frequency_change: float
    cooccurrence_change: float
    evidence_contexts: List[str]
    confidence: float

class AutomaticShiftDetector:
    """
    Automatic detection of semantic shifts using multiple statistical methods.
    
    Methods implemented:
    1. Frequency-based detection (sudden usage changes)
    2. Context vector analysis (semantic drift)
    3. Co-occurrence pattern analysis (collocational changes)
    4. Combined scoring with statistical significance
    """
    
    def __init__(self, 
                 min_frequency: int = 10,
                 context_window: int = 50,
                 significance_threshold: float = 0.05,
                 shift_threshold: float = 0.3):
        self.min_frequency = min_frequency
        self.context_window = context_window
        self.significance_threshold = significance_threshold
        self.shift_threshold = shift_threshold
        
        # Storage for analysis
        self.word_frequencies = defaultdict(lambda: defaultdict(int))
        self.word_contexts = defaultdict(lambda: defaultdict(list))
        self.word_cooccurrences = defaultdict(lambda: defaultdict(Counter))
        
    def analyze_corpus_for_shifts(self, 
                                decade_texts: Dict[str, List[str]], 
                                candidate_words: Optional[List[str]] = None) -> List[AutoDetectedShift]:
        """
        Analyze corpus to automatically detect semantic shifts.
        
        Args:
            decade_texts: Dictionary mapping decades to text lists
            candidate_words: Optional list of words to analyze (if None, extract high-frequency words)
            
        Returns:
            List of detected semantic shift candidates
        """
        logger.info("=== AUTOMATIC SEMANTIC SHIFT DETECTION ===")
        
        # Step 1: Extract word usage data
        if candidate_words is None:
            candidate_words = self._extract_candidate_words(decade_texts)
        
        logger.info(f"Analyzing {len(candidate_words)} candidate words across {len(decade_texts)} decades")
        
        # Step 2: Compute usage statistics
        self._compute_word_statistics(decade_texts, candidate_words)
        
        # Step 3: Detect shifts using multiple methods
        detected_shifts = []
        
        for word in candidate_words:
            shift_candidate = self._detect_word_shift(word, decade_texts)
            if shift_candidate and shift_candidate.confidence > 0.5:
                detected_shifts.append(shift_candidate)
        
        # Step 4: Rank by confidence and return top candidates
        detected_shifts.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Detected {len(detected_shifts)} potential semantic shifts")
        return detected_shifts
    
    def _extract_candidate_words(self, decade_texts: Dict[str, List[str]], 
                                top_n: int = 500) -> List[str]:
        """
        Extract candidate words for shift analysis based on frequency patterns.
        Focus on words with interesting temporal patterns.
        """
        logger.info("Extracting candidate words from corpus...")
        
        # Count word frequencies per decade
        decade_word_counts = {}
        total_counts = Counter()
        
        for decade, texts in decade_texts.items():
            decade_counts = Counter()
            for text in texts[:100]:  # Sample for efficiency
                words = re.findall(r'\\b[a-z]{3,15}\\b', text.lower())
                decade_counts.update(words)
            
            decade_word_counts[decade] = decade_counts
            total_counts.update(decade_counts)
        
        # Find words with interesting temporal patterns
        candidates = []
        decades = sorted(decade_texts.keys())
        
        for word, total_freq in total_counts.most_common(top_n * 2):
            if total_freq < self.min_frequency * len(decades):
                continue
                
            # Check if word has temporal variation
            decade_freqs = []
            for decade in decades:
                freq = decade_word_counts[decade].get(word, 0)
                decade_freqs.append(freq)
            
            # Calculate coefficient of variation
            if np.mean(decade_freqs) > 0:
                cv = np.std(decade_freqs) / np.mean(decade_freqs)
                if cv > 0.5:  # High temporal variation
                    candidates.append(word)
        
        logger.info(f"Selected {len(candidates)} candidate words with temporal variation")
        return candidates[:top_n]
    
    def _compute_word_statistics(self, decade_texts: Dict[str, List[str]], 
                               candidate_words: List[str]):
        """Compute comprehensive statistics for candidate words."""
        logger.info("Computing word usage statistics...")
        
        for decade, texts in decade_texts.items():
            decade_text = ' '.join(texts[:50])  # Sample for efficiency
            
            for word in candidate_words:
                # Count frequencies
                freq = len(re.findall(rf'\\b{re.escape(word)}\\b', decade_text.lower()))
                self.word_frequencies[word][decade] = freq
                
                # Extract contexts
                contexts = self._extract_contexts(decade_text, word)
                self.word_contexts[word][decade] = contexts[:20]  # Limit contexts
                
                # Compute co-occurrences
                coocs = self._compute_cooccurrences(decade_text, word)
                self.word_cooccurrences[word][decade] = coocs
    
    def _extract_contexts(self, text: str, word: str) -> List[str]:
        """Extract contexts around word occurrences."""
        contexts = []
        text_lower = text.lower()
        word_lower = word.lower()
        
        start = 0
        while True:
            pos = text_lower.find(word_lower, start)
            if pos == -1:
                break
                
            # Extract context window
            context_start = max(0, pos - self.context_window)
            context_end = min(len(text), pos + len(word) + self.context_window)
            context = text[context_start:context_end].strip()
            
            if len(context) > 20:  # Minimum context length
                contexts.append(context)
            
            start = pos + 1
            if len(contexts) >= 50:  # Limit contexts per decade
                break
        
        return contexts
    
    def _compute_cooccurrences(self, text: str, word: str, window: int = 10) -> Counter:
        """Compute word co-occurrences within a window."""
        coocs = Counter()
        words = re.findall(r'\\b[a-z]+\\b', text.lower())
        
        for i, w in enumerate(words):
            if w == word.lower():
                # Get surrounding words
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                for j in range(start, end):
                    if j != i and len(words[j]) > 2:
                        coocs[words[j]] += 1
        
        return coocs
    
    def _detect_word_shift(self, word: str, decade_texts: Dict[str, List[str]]) -> Optional[AutoDetectedShift]:
        """
        Detect semantic shift for a specific word using multiple methods.
        """
        decades = sorted(decade_texts.keys())
        if len(decades) < 3:
            return None
            
        # Method 1: Frequency-based change detection
        freq_change = self._detect_frequency_change(word, decades)
        
        # Method 2: Context vector change detection
        context_change = self._detect_context_change(word, decades)
        
        # Method 3: Co-occurrence pattern change
        cooc_change = self._detect_cooccurrence_change(word, decades)
        
        # Combine scores
        combined_score = np.mean([freq_change, context_change, cooc_change])
        
        if combined_score < self.shift_threshold:
            return None
        
        # Statistical significance testing
        significance = self._test_statistical_significance(word, decades)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            freq_change, context_change, cooc_change, significance, word, decades
        )
        
        # Get evidence contexts
        evidence_contexts = self._get_evidence_contexts(word, decades)
        
        # Determine time periods of change
        change_periods = self._identify_change_periods(word, decades)
        
        return AutoDetectedShift(
            word=word,
            shift_score=combined_score,
            significance=significance,
            time_periods=change_periods,
            context_change=context_change,
            frequency_change=freq_change,
            cooccurrence_change=cooc_change,
            evidence_contexts=evidence_contexts,
            confidence=confidence
        )
    
    def _detect_frequency_change(self, word: str, decades: List[str]) -> float:
        """Detect significant frequency changes over time."""
        frequencies = [self.word_frequencies[word][decade] for decade in decades]
        
        if sum(frequencies) == 0:
            return 0.0
        
        # Normalize by decade text length (approximation)
        normalized_freqs = np.array(frequencies) / (np.array(frequencies).sum() + 1)
        
        # Calculate change score using coefficient of variation
        if np.mean(normalized_freqs) > 0:
            cv = np.std(normalized_freqs) / np.mean(normalized_freqs)
            return min(cv, 1.0)
        
        return 0.0
    
    def _detect_context_change(self, word: str, decades: List[str]) -> float:
        """Detect semantic change using context vector analysis."""
        try:
            # Collect all contexts
            all_contexts = []
            decade_labels = []
            
            for decade in decades:
                contexts = self.word_contexts[word][decade]
                if len(contexts) < 5:  # Need minimum contexts
                    continue
                    
                all_contexts.extend(contexts)
                decade_labels.extend([decade] * len(contexts))
            
            if len(all_contexts) < 10:
                return 0.0
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            context_vectors = vectorizer.fit_transform(all_contexts)
            
            # Group by decades and compute centroids
            decade_centroids = {}
            for decade in decades:
                decade_indices = [i for i, label in enumerate(decade_labels) if label == decade]
                if len(decade_indices) < 3:
                    continue
                    
                decade_matrix = context_vectors[decade_indices]
                centroid = np.mean(decade_matrix.toarray(), axis=0)
                decade_centroids[decade] = centroid
            
            if len(decade_centroids) < 2:
                return 0.0
            
            # Calculate semantic drift between early and late periods
            early_decades = decades[:len(decades)//2]
            late_decades = decades[len(decades)//2:]
            
            early_centroid = np.mean([decade_centroids[d] for d in early_decades 
                                    if d in decade_centroids], axis=0)
            late_centroid = np.mean([decade_centroids[d] for d in late_decades 
                                   if d in decade_centroids], axis=0)
            
            # Calculate cosine distance (1 - similarity)
            similarity = cosine_similarity([early_centroid], [late_centroid])[0][0]
            semantic_distance = 1 - similarity
            
            return min(semantic_distance, 1.0)
            
        except Exception as e:
            logger.warning(f"Context analysis failed for '{word}': {e}")
            return 0.0
    
    def _detect_cooccurrence_change(self, word: str, decades: List[str]) -> float:
        """Detect change in co-occurrence patterns."""
        try:
            # Get co-occurrence vectors for each decade
            decade_coocs = {}
            all_cooc_words = set()
            
            for decade in decades:
                coocs = self.word_cooccurrences[word][decade]
                if len(coocs) < 5:
                    continue
                decade_coocs[decade] = coocs
                all_cooc_words.update(coocs.keys())
            
            if len(decade_coocs) < 2:
                return 0.0
            
            # Create co-occurrence vectors
            cooc_words = list(all_cooc_words)[:100]  # Limit vocabulary
            decade_vectors = {}
            
            for decade, coocs in decade_coocs.items():
                vector = np.array([coocs.get(word, 0) for word in cooc_words])
                if vector.sum() > 0:
                    vector = vector / vector.sum()  # Normalize
                decade_vectors[decade] = vector
            
            # Calculate change between early and late periods
            early_decades = [d for d in decades[:len(decades)//2] if d in decade_vectors]
            late_decades = [d for d in decades[len(decades)//2:] if d in decade_vectors]
            
            if not early_decades or not late_decades:
                return 0.0
            
            early_vector = np.mean([decade_vectors[d] for d in early_decades], axis=0)
            late_vector = np.mean([decade_vectors[d] for d in late_decades], axis=0)
            
            # Calculate Jensen-Shannon divergence
            js_divergence = self._jensen_shannon_divergence(early_vector, late_vector)
            
            return min(js_divergence, 1.0)
            
        except Exception as e:
            logger.warning(f"Co-occurrence analysis failed for '{word}': {e}")
            return 0.0
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two probability distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize to ensure they're probability distributions
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate KL divergences
        m = 0.5 * (p + q)
        kl_pm = stats.entropy(p, m)
        kl_qm = stats.entropy(q, m)
        
        # Jensen-Shannon divergence
        js = 0.5 * kl_pm + 0.5 * kl_qm
        return js
    
    def _test_statistical_significance(self, word: str, decades: List[str]) -> float:
        """Test statistical significance of observed changes."""
        try:
            frequencies = [self.word_frequencies[word][decade] for decade in decades]
            
            if len(frequencies) < 3 or sum(frequencies) < 10:
                return 1.0  # Not significant
            
            # Kolmogorov-Smirnov test for distribution change
            mid_point = len(frequencies) // 2
            early_freqs = frequencies[:mid_point]
            late_freqs = frequencies[mid_point:]
            
            if len(early_freqs) < 2 or len(late_freqs) < 2:
                return 1.0
            
            # Use Mann-Whitney U test for difference in distributions
            statistic, p_value = stats.mannwhitneyu(early_freqs, late_freqs, alternative='two-sided')
            
            return p_value
            
        except Exception as e:
            logger.warning(f"Significance testing failed for '{word}': {e}")
            return 1.0
    
    def _calculate_confidence(self, freq_change: float, context_change: float, 
                            cooc_change: float, significance: float, 
                            word: str, decades: List[str]) -> float:
        """Calculate overall confidence in the detected shift."""
        # Base confidence from combined scores
        base_confidence = np.mean([freq_change, context_change, cooc_change])
        
        # Adjust for statistical significance
        sig_adjustment = 1.0 if significance < self.significance_threshold else 0.5
        
        # Adjust for data quality
        total_frequency = sum(self.word_frequencies[word][d] for d in decades)
        freq_adjustment = min(1.0, total_frequency / 50.0)  # More data = higher confidence
        
        # Adjust for consistency across methods
        method_scores = [freq_change, context_change, cooc_change]
        consistency = 1.0 - (np.std(method_scores) / (np.mean(method_scores) + 0.1))
        
        # Combine adjustments
        confidence = base_confidence * sig_adjustment * freq_adjustment * consistency
        
        return min(confidence, 1.0)
    
    def _get_evidence_contexts(self, word: str, decades: List[str], n: int = 5) -> List[str]:
        """Get representative contexts showing the semantic change."""
        evidence = []
        
        # Get contexts from early and late periods
        early_decades = decades[:len(decades)//2]
        late_decades = decades[len(decades)//2:]
        
        for period_name, period_decades in [("Early", early_decades), ("Late", late_decades)]:
            period_contexts = []
            for decade in period_decades:
                period_contexts.extend(self.word_contexts[word][decade][:3])
            
            # Sample representative contexts
            if period_contexts:
                sampled = period_contexts[:n//2] if len(period_contexts) >= n//2 else period_contexts
                evidence.extend([f"[{period_name}] {ctx[:200]}..." for ctx in sampled])
        
        return evidence
    
    def _identify_change_periods(self, word: str, decades: List[str]) -> Tuple[str, str]:
        """Identify the time periods when the semantic change likely occurred."""
        frequencies = [self.word_frequencies[word][decade] for decade in decades]
        
        # Find the period with maximum change
        max_change_idx = 0
        max_change = 0
        
        for i in range(len(frequencies) - 1):
            if frequencies[i] > 0 and frequencies[i+1] > 0:
                change = abs(frequencies[i+1] - frequencies[i]) / (frequencies[i] + 1)
                if change > max_change:
                    max_change = change
                    max_change_idx = i
        
        # Return the transition period
        start_decade = decades[max_change_idx]
        end_decade = decades[min(max_change_idx + 1, len(decades) - 1)]
        
        return (start_decade, end_decade)

def run_automatic_detection(decade_texts: Dict[str, List[str]], 
                          candidate_words: Optional[List[str]] = None,
                          top_n: int = 50) -> List[AutoDetectedShift]:
    """
    Convenience function to run automatic semantic shift detection.
    
    Args:
        decade_texts: Dictionary mapping decades to text lists
        candidate_words: Optional list of specific words to analyze
        top_n: Number of top candidates to return
        
    Returns:
        List of detected semantic shift candidates
    """
    detector = AutomaticShiftDetector()
    detected_shifts = detector.analyze_corpus_for_shifts(decade_texts, candidate_words)
    
    return detected_shifts[:top_n]

if __name__ == "__main__":
    # Demo automatic detection
    print("=== AUTOMATIC SEMANTIC SHIFT DETECTION DEMO ===")
    
    # This would normally use real corpus data
    demo_texts = {
        "1900s": ["The computer calculated the figures by hand using mathematical tables."],
        "1950s": ["The electronic computer processed data using vacuum tubes and circuits."],
        "1980s": ["The personal computer had a mouse for clicking and a screen display."],
        "2000s": ["The computer network connected to the internet and cloud services."]
    }
    
    detector = AutomaticShiftDetector()
    shifts = detector.analyze_corpus_for_shifts(demo_texts, ["computer"])
    
    print(f"Detected {len(shifts)} potential shifts")
    for shift in shifts:
        print(f"  {shift.word}: confidence={shift.confidence:.2f}, score={shift.shift_score:.2f}") 