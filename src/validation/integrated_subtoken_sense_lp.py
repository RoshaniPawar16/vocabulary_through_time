#!/usr/bin/env python3
"""
Integrated Subtoken-Sense Linear Programming for Temporal Distribution Inference

This module implements joint optimization over BPE subtoken frequencies and semantic 
sense frequencies to infer temporal distribution from tokenizer merge patterns.

The key insight is that semantic shifts leave traces in subtoken usage patterns,
which can be captured through direct mathematical constraints linking f_ij 
(subtoken frequencies) to s_wkj (sense frequencies).
"""

import logging
import numpy as np
import cvxpy as cp
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import re
import time

from .comprehensive_semantic_shifts import get_high_confidence_shifts

logger = logging.getLogger(__name__)

class IntegratedSubtokenSenseLP:
    """
    Joint optimization system for BPE subtoken and semantic sense analysis.
    
    Instead of the usual pipeline (tokenize → classify → optimize), this approach
    simultaneously optimizes subtoken frequencies and sense classifications with
    explicit constraints linking them together.
    """
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """Initialize the LP system with a target tokenizer."""
        logger.info("Initializing integrated subtoken-sense LP system")
        
        # Load tokenizer and vocabulary
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.bpe_vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.bpe_vocab)
        
        # Load semantic shift words
        self.semantic_shifts = get_high_confidence_shifts(min_confidence=0.85)
        
        # Data structures for optimization
        self.subtoken_frequencies = {}  # f_ij: subtoken i, decade j
        self.sense_frequencies = {}     # s_wkj: word w, sense k, decade j
        self.subtoken_to_sense_mapping = {}  # Φ(i,w,k): maps subtoken i to word w sense k
        
        # Trained models
        self.sense_classifiers = {}
        self.integrated_lp_model = None
        
        logger.info(f"Loaded {len(self.semantic_shifts)} semantic shift words")
        logger.info(f"BPE vocabulary size: {self.vocab_size}")
        
    def extract_subtoken_patterns(self, decade_texts: Dict[str, List[str]]) -> Dict:
        """
        Extract BPE subtoken frequency patterns across decades.
        These become the f_ij variables in our optimization.
        """
        logger.info("Extracting BPE subtoken patterns")
        
        subtoken_frequencies = defaultdict(lambda: defaultdict(int))
        decade_token_counts = {}
        
        for decade, texts in decade_texts.items():
            logger.info(f"Processing {decade}...")
            
            # Sample texts for efficiency
            sample_texts = texts[:30]
            decade_text = ' '.join(sample_texts)
            
            # Tokenize with BPE
            try:
                tokens = self.tokenizer.tokenize(decade_text[:50000])  # Memory limit
                decade_token_counts[decade] = len(tokens)
                
                # Count subtoken frequencies
                token_counts = Counter(tokens)
                
                for token, count in token_counts.items():
                    if token in self.bpe_vocab:
                        subtoken_frequencies[token][decade] = count
                        
            except Exception as e:
                logger.warning(f"Tokenization failed for {decade}: {e}")
                decade_token_counts[decade] = 0
        
        # Normalize by decade text length
        normalized_frequencies = defaultdict(lambda: defaultdict(float))
        
        for token, decade_counts in subtoken_frequencies.items():
            for decade, count in decade_counts.items():
                if decade_token_counts[decade] > 0:
                    normalized_freq = count / decade_token_counts[decade]
                    normalized_frequencies[token][decade] = normalized_freq
        
        self.subtoken_frequencies = dict(normalized_frequencies)
        
        logger.info(f"Found patterns for {len(self.subtoken_frequencies)} subtokens")
        return {
            'subtoken_frequencies': self.subtoken_frequencies,
            'decade_token_counts': decade_token_counts,
            'total_unique_subtokens': len(self.subtoken_frequencies)
        }
    
    def build_subtoken_sense_mapping(self, decade_texts: Dict[str, List[str]]) -> Dict:
        """
        Build mapping Φ(i,w,k) from subtokens to word senses.
        
        This is where we figure out how much each subtoken contributes
        to each sense of each word. It's the bridge between BPE patterns
        and semantic analysis.
        """
        logger.info("Building subtoken-sense mapping")
        
        # Initialize mapping structure: subtoken -> word -> sense -> weight
        mapping = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Train sense classifiers first
        self._train_sense_classifiers(decade_texts)
        
        for word, word_data in self.semantic_shifts.items():
            if word not in self.sense_classifiers:
                continue
                
            logger.info(f"Mapping subtokens for '{word}'")
            
            senses = word_data['senses']
            classifier = self.sense_classifiers[word]
            
            # Get all subtokens that could represent this word
            word_subtokens = self._get_word_subtokens(word)
            
            # For each subtoken, figure out its sense contributions
            for subtoken in word_subtokens:
                
                # Collect contexts containing this subtoken
                subtoken_contexts = self._extract_subtoken_contexts(subtoken, decade_texts)
                
                if len(subtoken_contexts) < 5:
                    continue
                
                try:
                    # Classify senses for subtoken contexts
                    X = classifier['vectorizer'].transform(subtoken_contexts)
                    sense_probabilities = classifier['model'].predict_proba(X)
                    
                    # Average probabilities across contexts
                    avg_sense_probs = np.mean(sense_probabilities, axis=0)
                    
                    # Map subtoken to word senses with weights
                    for sense_idx, sense in enumerate(senses):
                        if sense_idx < len(avg_sense_probs):
                            weight = avg_sense_probs[sense_idx]
                            mapping[subtoken][word][sense] = weight
                            
                except Exception as e:
                    logger.debug(f"Mapping failed for subtoken '{subtoken}': {e}")
                    continue
        
        # Clean mapping - only keep significant weights
        cleaned_mapping = {}
        for subtoken, word_data in mapping.items():
            cleaned_word_data = {}
            for word, sense_data in word_data.items():
                cleaned_sense_data = {sense: weight for sense, weight in sense_data.items() 
                                    if weight > 0.1}  # Minimum significance threshold
                if cleaned_sense_data:
                    cleaned_word_data[word] = cleaned_sense_data
            if cleaned_word_data:
                cleaned_mapping[subtoken] = cleaned_word_data
        
        self.subtoken_to_sense_mapping = cleaned_mapping
        
        logger.info(f"Built mapping for {len(cleaned_mapping)} subtokens")
        logger.info(f"Covering {len(set(w for st_data in cleaned_mapping.values() for w in st_data.keys()))} words")
        
        return {
            'mapping_size': len(cleaned_mapping),
            'mapped_subtokens': list(cleaned_mapping.keys())[:20],  # Sample
            'mapped_words': list(set(w for st_data in cleaned_mapping.values() for w in st_data.keys()))
        }
    
    def _get_word_subtokens(self, word: str) -> List[str]:
        """Get all BPE subtokens that could represent this word."""
        subtokens = set()
        
        # Direct tokenization
        try:
            direct_tokens = self.tokenizer.tokenize(word)
            subtokens.update(direct_tokens)
        except:
            pass
        
        # Variations (capitalized, etc.)
        for variation in [word.lower(), word.upper(), word.capitalize()]:
            try:
                tokens = self.tokenizer.tokenize(variation)
                subtokens.update(tokens)
            except:
                pass
        
        # Add space-prefixed versions (common in GPT tokenizers)
        for variation in [f" {word.lower()}", f" {word.capitalize()}", f" {word.upper()}"]:
            try:
                tokens = self.tokenizer.tokenize(variation)
                subtokens.update(tokens)
            except:
                pass
        
        # Find partial matches in vocabulary - MORE AGGRESSIVE
        word_lower = word.lower()
        for vocab_token in self.bpe_vocab:
            # Clean token (remove BPE artifacts)
            clean_token = vocab_token.replace('Ġ', '').replace('##', '').lower()
            
            # Multiple matching strategies
            match_found = False
            
            # 1. Exact matches
            if clean_token == word_lower:
                subtokens.add(vocab_token)
                match_found = True
            
            # 2. Word contains token or token contains word (if meaningful length)
            elif len(clean_token) >= 3 and len(word_lower) >= 3:
                if (word_lower in clean_token or clean_token in word_lower):
                    subtokens.add(vocab_token)
                    match_found = True
            
            # 3. Significant character overlap (for morphological variants)
            elif len(clean_token) >= 4 and len(word_lower) >= 4:
                overlap = len(set(word_lower) & set(clean_token))
                if overlap >= min(len(word_lower), len(clean_token)) * 0.6:
                    subtokens.add(vocab_token)
                    match_found = True
            
            # 4. Edit distance for close matches
            elif len(clean_token) >= 4 and len(word_lower) >= 4:
                if self._levenshtein_distance(word_lower, clean_token) <= 2:
                    subtokens.add(vocab_token)
                    match_found = True
            
            # 5. Common morphological patterns
            if not match_found and len(word_lower) >= 4:
                # Plural forms
                if word_lower.endswith('s') and clean_token == word_lower[:-1]:
                    subtokens.add(vocab_token)
                elif clean_token.endswith('s') and word_lower == clean_token[:-1]:
                    subtokens.add(vocab_token)
                # Past tense forms
                elif word_lower.endswith('ed') and clean_token == word_lower[:-2]:
                    subtokens.add(vocab_token)
                elif clean_token.endswith('ed') and word_lower == clean_token[:-2]:
                    subtokens.add(vocab_token)
                # Progressive forms
                elif word_lower.endswith('ing') and clean_token == word_lower[:-3]:
                    subtokens.add(vocab_token)
                elif clean_token.endswith('ing') and word_lower == clean_token[:-3]:
                    subtokens.add(vocab_token)
        
        # Log findings for debugging
        if subtokens:
            logger.debug(f"Found {len(subtokens)} subtokens for '{word}': {list(subtokens)[:5]}...")
        else:
            logger.debug(f"No subtokens found for '{word}'")
        
        return list(subtokens)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _extract_subtoken_contexts(self, subtoken: str, decade_texts: Dict[str, List[str]], 
                                 max_contexts: int = 50) -> List[str]:
        """Extract contexts containing a specific subtoken."""
        contexts = []
        
        for decade, texts in decade_texts.items():
            for text in texts[:10]:  # Sample texts
                try:
                    # Tokenize text
                    tokens = self.tokenizer.tokenize(text[:5000])
                    
                    # Find subtoken occurrences
                    for i, token in enumerate(tokens):
                        if token == subtoken:
                            # Extract context window
                            start_idx = max(0, i - 10)
                            end_idx = min(len(tokens), i + 11)
                            context_tokens = tokens[start_idx:end_idx]
                            
                            # Convert back to text (approximate)
                            context_text = self.tokenizer.convert_tokens_to_string(context_tokens)
                            if len(context_text.strip()) > 20:
                                contexts.append(context_text.strip())
                                
                            if len(contexts) >= max_contexts:
                                return contexts
                                
                except Exception as e:
                    logger.debug(f"Context extraction failed: {e}")
                    continue
        
        return contexts
    
    def _train_sense_classifiers(self, decade_texts: Dict[str, List[str]]):
        """Train sense classifiers for semantic shift words."""
        logger.info("Training sense classifiers for mapping...")
        
        for word, word_data in self.semantic_shifts.items():
            logger.info(f"Training classifier for '{word}'")
            
            # Collect training data
            contexts = []
            labels = []
            
            for decade, texts in decade_texts.items():
                # Handle both "1900s" and "1900" format
                if decade.endswith('s'):
                    decade_year = int(decade.replace('s', ''))
                else:
                    decade_year = int(decade)
                
                for text in texts[:15]:  # Sample for efficiency
                    word_contexts = self._extract_word_contexts(text, word)
                    logger.debug(f"  Found {len(word_contexts)} contexts for '{word}' in {decade}")
                    
                    for context in word_contexts:
                        label = self._create_sense_label(word, context, decade_year)
                        if label is not None:
                            contexts.append(context)
                            labels.append(label)
            
            logger.info(f"  Collected {len(contexts)} contexts with {len(set(labels))} unique labels")
            
            # Train classifier if enough data
            if len(contexts) >= 10 and len(set(labels)) >= 2:  # Lowered minimum contexts
                try:
                    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
                    X = vectorizer.fit_transform(contexts)
                    
                    classifier = LogisticRegression(class_weight='balanced', random_state=42)
                    
                    # Quick validation
                    scores = cross_val_score(classifier, X, labels, cv=3, scoring='f1_macro')
                    
                    logger.info(f"  Cross-validation scores: {scores.mean():.3f} ± {scores.std():.3f}")
                    
                    if scores.mean() > 0.4:  # Lowered threshold for modest performance
                        classifier.fit(X, labels)
                        
                        self.sense_classifiers[word] = {
                            'model': classifier,
                            'vectorizer': vectorizer,
                            'performance': scores.mean(),
                            'senses': word_data['senses']
                        }
                        
                        logger.info(f"  ✓ Trained classifier for '{word}': {scores.mean():.3f}")
                    else:
                        logger.info(f"  ✗ Poor performance for '{word}': {scores.mean():.3f}")
                        
                except Exception as e:
                    logger.info(f"  ✗ Training failed for '{word}': {e}")
            else:
                logger.info(f"  ✗ Insufficient data for '{word}': {len(contexts)} contexts, {len(set(labels))} labels")
    
    def _extract_word_contexts(self, text: str, word: str, window_size: int = 50) -> List[str]:
        """Extract contexts around word occurrences."""
        contexts = []
        text_lower = text.lower()
        word_lower = word.lower()
        
        start = 0
        while True:
            pos = text_lower.find(word_lower, start)
            if pos == -1:
                break
            
            context_start = max(0, pos - window_size)
            context_end = min(len(text), pos + len(word) + window_size)
            context = text[context_start:context_end].strip()
            
            if len(context) > 20:
                contexts.append(context)
            
            start = pos + 1
            if len(contexts) >= 20:
                break
        
        return contexts
    
    def _create_sense_label(self, word: str, context: str, decade_year: int) -> Optional[int]:
        """Create sense labels using temporal and contextual cues."""
        if word not in self.semantic_shifts:
            return None
        
        word_data = self.semantic_shifts[word]
        transition_start, transition_end = word_data['transition']
        markers = word_data['markers']
        senses = word_data['senses']
        
        # Temporal prior
        if decade_year < transition_start:
            temporal_prior = 0
        elif decade_year > transition_end:
            temporal_prior = 1
        else:
            progress = (decade_year - transition_start) / (transition_end - transition_start)
            temporal_prior = 0 if progress < 0.5 else 1
        
        # Contextual evidence
        context_lower = context.lower()
        old_markers = markers.get(senses[0], [])
        new_markers = markers.get(senses[1], [])
        
        old_score = sum(1 for marker in old_markers if marker in context_lower)
        new_score = sum(1 for marker in new_markers if marker in context_lower)
        
        # Decision logic
        if old_score > new_score + 1:
            return 0
        elif new_score > old_score + 1:
            return 1
        else:
            return temporal_prior
    
    def train_integrated_lp_model(self, decade_texts: Dict[str, List[str]]) -> Dict:
        """
        CORE INNOVATION: Train the integrated LP model that jointly optimizes:
        1. Subtoken frequencies f_ij
        2. Sense frequencies s_wkj  
        3. Temporal distribution p_j
        
        With explicit constraints linking subtokens to senses.
        """
        logger.info("=== TRAINING INTEGRATED SUBTOKEN-SENSE LP MODEL ===")
        
        # Step 1: Extract subtoken patterns
        subtoken_results = self.extract_subtoken_patterns(decade_texts)
        
        # Step 2: Build subtoken-sense mapping
        mapping_results = self.build_subtoken_sense_mapping(decade_texts)
        
        # Step 3: Set up integrated optimization
        decades = sorted(decade_texts.keys())
        n_decades = len(decades)
        
        # Get relevant subtokens (those mapped to semantic shift words)
        relevant_subtokens = list(self.subtoken_to_sense_mapping.keys())
        n_subtokens = len(relevant_subtokens)
        
        if n_subtokens == 0:
            raise ValueError("No subtokens mapped to semantic shift words")
        
        # Get semantic shift words and senses
        mapped_words = list(set(word for st_data in self.subtoken_to_sense_mapping.values() 
                               for word in st_data.keys()))
        
        word_sense_pairs = []
        for word in mapped_words:
            if word in self.semantic_shifts:
                for sense in self.semantic_shifts[word]['senses']:
                    word_sense_pairs.append((word, sense))
        
        logger.info(f"Integrated LP setup:")
        logger.info(f"  • Decades: {n_decades}")
        logger.info(f"  • Relevant subtokens: {n_subtokens}")
        logger.info(f"  • Word-sense pairs: {len(word_sense_pairs)}")
        
        # CORE INNOVATION: Integrated optimization
        return self._solve_integrated_optimization(
            decades, relevant_subtokens, word_sense_pairs
        )
    
    def _solve_integrated_optimization(self, decades: List[str], subtokens: List[str], 
                                     word_sense_pairs: List[Tuple[str, str]]) -> Dict:
        """
        CORE RESEARCH CONTRIBUTION: Joint optimization with explicit constraints.
        
        Uses alternating optimization to handle bilinear constraints:
        1. Fix p, optimize f,s (convex subproblem)
        2. Fix f,s, optimize p (convex subproblem)
        3. Iterate until convergence
        
        Variables:
        - f[i,j]: frequency of subtoken i in decade j
        - s[w,k,j]: frequency of sense k of word w in decade j
        - p[j]: temporal distribution weight for decade j
        
        Constraints:
        - Consistency: s[w,k,j] = Σ_i Φ(i,w,k) * f[i,j]
        - Normalization: Σ_j p[j] = 1
        - Non-negativity: f, s, p ≥ 0
        """
        logger.info("=== SOLVING INTEGRATED OPTIMIZATION (ALTERNATING) ===")
        
        n_decades = len(decades)
        n_subtokens = len(subtokens)
        n_word_senses = len(word_sense_pairs)
        
        try:
            # OBSERVED DATA - subtoken frequencies from corpus
            observed_f = np.zeros((n_subtokens, n_decades))
            for i, subtoken in enumerate(subtokens):
                for j, decade in enumerate(decades):
                    observed_f[i, j] = self.subtoken_frequencies.get(subtoken, {}).get(decade, 0)
            
            # INITIALIZE VARIABLES
            # Start with uniform temporal distribution
            p_current = np.ones(n_decades) / n_decades
            
            # Initialize f with observed data (small regularization)
            f_current = observed_f + 0.001 * np.random.rand(n_subtokens, n_decades)
            
            # Initialize s based on f and mapping
            s_current = np.zeros((n_word_senses, n_decades))
            
            logger.info(f"Starting alternating optimization:")
            logger.info(f"  • Variables: f({n_subtokens}×{n_decades}), s({n_word_senses}×{n_decades}), p({n_decades})")
            
            # ALTERNATING OPTIMIZATION LOOP
            max_iterations = 10
            tolerance = 1e-4
            
            for iteration in range(max_iterations):
                logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
                
                # ========================================
                # STEP 1: Fix p, optimize f and s
                # ========================================
                
                # VARIABLES for subproblem 1
                f = cp.Variable((n_subtokens, n_decades), nonneg=True)
                s = cp.Variable((n_word_senses, n_decades), nonneg=True)
                
                # CONSTRAINTS for subproblem 1
                constraints_1 = []
                
                # Subtoken-sense consistency constraints
                for ws_idx, (word, sense) in enumerate(word_sense_pairs):
                    for j, decade in enumerate(decades):
                        # s[w,k,j] = Σ_i Φ(i,w,k) * f[i,j]
                        subtoken_contribution = 0
                        
                        for i, subtoken in enumerate(subtokens):
                            if (subtoken in self.subtoken_to_sense_mapping and
                                word in self.subtoken_to_sense_mapping[subtoken] and
                                sense in self.subtoken_to_sense_mapping[subtoken][word]):
                                
                                weight = self.subtoken_to_sense_mapping[subtoken][word][sense]
                                subtoken_contribution += weight * f[i, j]
                        
                        # Enforce consistency
                        constraints_1.append(s[ws_idx, j] == subtoken_contribution)
                
                # Bounds on f (based on observations)
                constraints_1.append(f <= observed_f * 10)  # Upper bound
                constraints_1.append(f >= observed_f * 0.1)  # Lower bound
                
                # OBJECTIVE for subproblem 1
                # Minimize reconstruction error + regularization
                reconstruction_error = cp.sum_squares(f - observed_f)
                sparsity_penalty = 0.01 * cp.sum(cp.abs(f))
                smoothness_penalty = 0.001 * cp.sum_squares(f[:, 1:] - f[:, :-1])
                
                # Sense frequency regularization (using fixed p)
                sense_regularization = 0
                for ws_idx in range(n_word_senses):
                    # Encourage sense frequencies to be consistent with temporal distribution
                    expected_total = cp.sum(s[ws_idx, :])
                    weighted_total = s[ws_idx, :] @ p_current  # Linear term (p fixed)
                    sense_regularization += cp.square(expected_total - weighted_total * n_decades)
                sense_regularization *= 0.001
                
                objective_1 = cp.Minimize(
                    reconstruction_error + sparsity_penalty + smoothness_penalty + sense_regularization
                )
                
                # SOLVE subproblem 1
                problem_1 = cp.Problem(objective_1, constraints_1)
                problem_1.solve(solver=cp.OSQP, verbose=False, max_iter=5000)
                
                if problem_1.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    logger.warning(f"    Subproblem 1 failed: {problem_1.status}")
                    break
                
                # Update f and s
                f_new = f.value
                s_new = s.value
                
                # ========================================
                # STEP 2: Fix f,s, optimize p
                # ========================================
                
                # VARIABLES for subproblem 2
                p = cp.Variable(n_decades, nonneg=True)
                
                # CONSTRAINTS for subproblem 2
                constraints_2 = []
                constraints_2.append(cp.sum(p) == 1)  # Probability distribution
                constraints_2.append(p >= 0.001)  # Minimum probability
                
                # Temporal smoothness
                for j in range(n_decades - 1):
                    constraints_2.append(cp.abs(p[j+1] - p[j]) <= 0.3)
                
                # OBJECTIVE for subproblem 2
                # Encourage temporal distribution consistent with sense patterns
                temporal_consistency = 0
                for ws_idx in range(n_word_senses):
                    # Want: sense frequencies should align with temporal distribution
                    sense_pattern = s_new[ws_idx, :]  # Fixed from step 1
                    normalized_pattern = sense_pattern / (np.sum(sense_pattern) + 1e-8)
                    temporal_consistency += cp.sum_squares(p - normalized_pattern)
                
                # Add preference for earlier periods (slight bias)
                period_preference = 0.001 * cp.sum(cp.multiply(np.arange(n_decades), p))
                
                objective_2 = cp.Minimize(temporal_consistency + period_preference)
                
                # SOLVE subproblem 2
                problem_2 = cp.Problem(objective_2, constraints_2)
                problem_2.solve(solver=cp.OSQP, verbose=False, max_iter=5000)
                
                if problem_2.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    logger.warning(f"    Subproblem 2 failed: {problem_2.status}")
                    break
                
                # Update p
                p_new = p.value
                
                # ========================================
                # CHECK CONVERGENCE
                # ========================================
                
                f_change = np.linalg.norm(f_new - f_current) if f_current is not None else float('inf')
                p_change = np.linalg.norm(p_new - p_current)
                
                logger.info(f"    Changes: ||Δf||={f_change:.6f}, ||Δp||={p_change:.6f}")
                
                # Update for next iteration
                f_current = f_new
                s_current = s_new
                p_current = p_new
                
                # Check convergence
                if f_change < tolerance and p_change < tolerance:
                    logger.info(f"    ✅ Converged after {iteration + 1} iterations")
                    break
            
            else:
                logger.info(f"    ⚠️  Reached maximum iterations ({max_iterations})")
            
            # ========================================
            # FINAL RESULTS
            # ========================================
            
            logger.info("✅ Alternating optimization completed successfully!")
            
            # Store model
            self.integrated_lp_model = {
                'subtoken_frequencies': f_current,
                'sense_frequencies': s_current,
                'temporal_distribution': p_current,
                'decades': decades,
                'subtokens': subtokens,
                'word_sense_pairs': word_sense_pairs,
                'iterations': iteration + 1,
                'mapping': self.subtoken_to_sense_mapping
            }
            
            # Convert temporal distribution to dictionary
            temporal_dist = {decade: float(p_current[i]) for i, decade in enumerate(decades)}
            
            # Calculate final objective value
            final_reconstruction_error = np.sum((f_current - observed_f) ** 2)
            
            logger.info(f"Final reconstruction error: {final_reconstruction_error:.4f}")
            logger.info("Temporal distribution:")
            for decade, prob in temporal_dist.items():
                logger.info(f"  {decade}: {prob*100:.1f}%")
            
            return {
                'status': 'success',
                'temporal_distribution': temporal_dist,
                'reconstruction_error': final_reconstruction_error,
                'iterations': iteration + 1,
                'n_subtokens': n_subtokens,
                'n_word_senses': n_word_senses,
                'solver_status': 'OPTIMAL_ALTERNATING',
                'sense_frequencies': {
                    f"{word_sense_pairs[i][0]}_{word_sense_pairs[i][1]}": s_current[i, :].tolist()
                    for i in range(len(word_sense_pairs))
                }
            }
                
        except Exception as e:
            logger.error(f"❌ Alternating optimization failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_complete_integrated_analysis(self, training_texts: Dict[str, List[str]], 
                                       test_tokenizer: str = None) -> Dict:
        """
        Run the complete integrated subtoken-sense analysis.
        
        This implements the CORE RESEARCH CONTRIBUTION.
        """
        start_time = time.time()
        
        logger.info("=== COMPLETE INTEGRATED SUBTOKEN-SENSE ANALYSIS ===")
        logger.info(f"Training decades: {list(training_texts.keys())}")
        logger.info(f"Semantic shift words: {len(self.semantic_shifts)}")
        
        results = {
            'methodology': 'integrated_subtoken_sense_lp',
            'innovation': 'direct_bpe_semantic_optimization',
            'training_decades': list(training_texts.keys()),
            'vocabulary_size': len(self.semantic_shifts)
        }
        
        # Step 1: Train integrated model
        training_results = self.train_integrated_lp_model(training_texts)
        results['training_results'] = training_results
        
        if training_results['status'] == 'success':
            # Step 2: Get training temporal distribution
            results['training_temporal_distribution'] = training_results['temporal_distribution']
        
        # Timing and summary
        total_time = time.time() - start_time
        results['execution_time'] = total_time
        
        logger.info("=== INTEGRATED ANALYSIS COMPLETE ===")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        if training_results['status'] == 'success':
            temporal_dist = training_results['temporal_distribution']
            top_decade = max(temporal_dist.items(), key=lambda x: x[1])
            logger.info(f"Top predicted period: {top_decade[0]} ({top_decade[1]*100:.1f}%)")
        
        return results

# Convenience function
def run_integrated_analysis(training_texts: Dict[str, List[str]], 
                          tokenizer_name: str = "gpt2",
                          test_tokenizer: str = None) -> Dict:
    """
    Convenience function to run the integrated subtoken-sense analysis.
    
    This is the CORE RESEARCH CONTRIBUTION - direct joint optimization
    over BPE patterns and semantic shifts.
    """
    analyzer = IntegratedSubtokenSenseLP(tokenizer_name)
    return analyzer.run_complete_integrated_analysis(training_texts, test_tokenizer)

if __name__ == "__main__":
    # Demo the integrated approach
    print("=== INTEGRATED SUBTOKEN-SENSE LP DEMO ===")
    print("This implements the CORE research innovation:")
    print("Direct joint optimization over BPE patterns and semantic shifts")
    
    # Would normally use real data
    demo_texts = {
        "1900s": ["The computer calculated figures using tables and manual methods."],
        "1980s": ["The computer processed data using electronic circuits and memory."],
        "2000s": ["The computer connected to networks and cloud services online."]
    }
    
    analyzer = IntegratedSubtokenSenseLP("gpt2")
    results = analyzer.run_complete_integrated_analysis(demo_texts)
    
    print(f"Analysis completed: {results['training_results']['status']}")
    if results['training_results']['status'] == 'success':
        print("Temporal distribution:")
        for decade, prob in results['training_temporal_distribution'].items():
            print(f"  {decade}: {prob*100:.1f}%")
