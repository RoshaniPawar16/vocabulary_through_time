import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer
from scipy.stats import entropy
import re
from typing import Dict, List, Tuple, Optional, Set
import warnings
import matplotlib.pyplot as plt
import os
import torch

logger = logging.getLogger(__name__)

class SemanticShiftDetector:
    """
    STRICT IMPLEMENTATION: Semantic shift detection following exact research equations
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", load_model: bool = True):
        self.semantic_model_name = model_name
        self.device_name = "mps" if torch.backends.mps.is_available() else "cpu"
        
        if load_model:
            self.semantic_model = SentenceTransformer(model_name, device=self.device_name)
            logger.info(f"✅ MPS backend is available. Using Apple Silicon GPU.")
        else:
            self.semantic_model = None
        
        self.regression_model = TemporalRegressionModel()
        
        # EXACT SPECIFICATION: 18 decades for historical analysis
        self.decades = [
            '1850s', '1860s', '1870s', '1880s', '1890s', '1900s', '1910s', '1920s', '1930s',
            '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'
        ]
        
        self.semantic_shift_words = [
            'the', 'and', 'of', 'to', 'in', 'is', 'it', 'you', 'that', 'he',
            'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'at', 'be',
            'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not',
            'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'use',
            'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other'
        ]

    def batch_extract_and_embed_contexts(self, decade_texts: Dict, max_per_decade: int = 50):
        """
        Batch extract all contexts for all words and embed them in a single GPU batch.
        Returns:
            word_to_contexts: {word: [context, ...]}
            context_to_embedding: {context: embedding}
        """
        import re
        import random
        from collections import defaultdict
        all_contexts = []
        word_to_contexts = defaultdict(list)
        # 1. Extract all contexts for all words
        for word in self.semantic_shift_words:
            for decade, texts in decade_texts.items():
                decade_contexts = []
                for text in texts:
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if re.search(r'\b' + re.escape(word) + r'\b', sentence, re.IGNORECASE):
                            clean_sentence = re.sub(r'\s+', ' ', sentence.strip())
                            if 20 <= len(clean_sentence) <= 300:
                                decade_contexts.append(clean_sentence)
                # Sample contexts if too many
                if len(decade_contexts) > max_per_decade:
                    decade_contexts = random.sample(decade_contexts, max_per_decade)
                word_to_contexts[word].extend(decade_contexts)
                all_contexts.extend(decade_contexts)
        # 2. Remove duplicates for embedding
        all_contexts = list(set(all_contexts))
        # 3. Batch embed all contexts
        logger.info(f"Batch embedding {len(all_contexts)} unique contexts on device {self.device_name}...")
        context_embeddings = self.semantic_model.encode(all_contexts, show_progress_bar=True, device=self.device_name)
        # 4. Map context to embedding
        context_to_embedding = {ctx: emb for ctx, emb in zip(all_contexts, context_embeddings)}
        return word_to_contexts, context_to_embedding

    def train_semantic_model(self, decade_texts: Dict) -> Dict:
        """
        STRICT IMPLEMENTATION: Multi-Sample Regression Training with Comprehensive Optimization
        """
        logger.info("STRICT IMPLEMENTATION: Training with comprehensive optimization...")
        decades = self.decades
        all_word_features = []
        all_word_targets = []
        successful_words = 0
        self.word_clustering_results = {}

        # STEP 1: Comprehensive pre-computation (single pass through corpus)
        self._precompute_corpus_statistics(decade_texts)

        # STEP 2: Batch extract and embed all contexts for all words
        word_to_contexts, context_to_embedding = self.batch_extract_and_embed_contexts(decade_texts)

        # STEP 3: Process each word using pre-computed statistics
        for word in self.semantic_shift_words:
            try:
                word_contexts = word_to_contexts[word]
                if len(word_contexts) < 4:
                    logger.warning(f"Insufficient contexts for '{word}': {len(word_contexts)}")
                    continue

                # Use precomputed embeddings for clustering
                embeddings = np.array([context_to_embedding[ctx] for ctx in word_contexts])
                clustering_results = self.discover_senses_with_silhouette(word_contexts, word, precomputed_embeddings=embeddings)
                self.word_clustering_results[word] = clustering_results

                # Extract FIXED-LENGTH features (49 features) - NOW FAST
                features = self.extract_fixed_features(word, clustering_results, decade_texts)
                feature_vector = self._features_dict_to_vector(features, decades)
                all_word_features.append(feature_vector)

                # Create individual ground truth for this word
                word_ground_truth = self._create_word_temporal_ground_truth(word, decade_texts)
                target_vector = [word_ground_truth[decade] for decade in decades]
                all_word_targets.append(target_vector)

                successful_words += 1
                logger.info(f"Word '{word}': {clustering_results['optimal_k']} senses, silhouette={clustering_results['best_silhouette']:.3f}")
            except Exception as e:
                logger.error(f"Failed to process word '{word}': {e}")
                continue

        if successful_words == 0:
            return {'error': 'No words successfully processed'}

        X_features = np.array(all_word_features)
        Y_targets = np.array(all_word_targets)
        training_performance = self.regression_model.train_multi_output_model(X_features, Y_targets, decades)

        return {
            'X_features': X_features,
            'Y_targets': Y_targets,
            'training_performance': training_performance,
            'successful_words': successful_words,
            'word_clustering_results': self.word_clustering_results
        }

    def discover_senses_with_silhouette(self, word_contexts: List[str], word: str, precomputed_embeddings=None) -> Dict:
        """
        EXACT EQUATION 3: Adaptive clustering using silhouette scores
        s(K; θ) = (1/n)Σᵢ(bᵢ-aᵢ)/max(aᵢ,bᵢ)
        
        EXACT HYPERPARAMETERS θ:
        - K_range: [2, min(8, len(contexts)//3)]
        - random_state: 42
        - n_init: 10
        - max_iter: 300
        """
        logger.info(f"Discovering senses for '{word}' using EXACT Equation 3...")
        
        if len(word_contexts) < 4:
            logger.warning(f"Insufficient contexts for '{word}': {len(word_contexts)}")
            return self._create_fallback_clustering(word_contexts, word)
        
        # 1. Generate embeddings for word contexts
        try:
            if precomputed_embeddings is not None:
                embeddings = precomputed_embeddings
            else:
                embeddings = self.semantic_model.encode(word_contexts)
        except Exception as e:
            logger.error(f"Error encoding contexts for '{word}': {e}")
            return self._create_fallback_clustering(word_contexts, word)
        
        # 2. Test different cluster numbers with EXACT hyperparameters θ
        silhouette_scores = {}
        cluster_range = range(2, min(8, len(word_contexts)//3))  # θ_K_range
        
        for k in cluster_range:
            try:
                # Apply K-means clustering with EXACT hyperparameters θ
                kmeans = KMeans(
                    n_clusters=k, 
                    random_state=42,    # θ_random_state
                    n_init=10,          # θ_n_init  
                    max_iter=300        # θ_max_iter
                )
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score for this K (Equation 3)
                if len(np.unique(cluster_labels)) > 1:
                    sil_score = silhouette_score(embeddings, cluster_labels)
                    silhouette_scores[k] = sil_score
                    logger.info(f"  K={k}: s(K;θ) = {sil_score:.3f}")
            except Exception as e:
                logger.debug(f"Error clustering with K={k}: {e}")
        
        # 3. Find optimal K* = arg maxₖ s(K; θ)
        if silhouette_scores:
            optimal_k = max(silhouette_scores, key=silhouette_scores.get)
            best_silhouette = silhouette_scores[optimal_k]
        else:
            optimal_k = 2  # fallback
            best_silhouette = 0.0
        
        # 4. Final clustering with optimal K*
        try:
            final_kmeans = KMeans(
                n_clusters=optimal_k, 
                random_state=42, 
                n_init=10,
                max_iter=300
            )
            final_labels = final_kmeans.fit_predict(embeddings)
        except Exception as e:
            logger.error(f"Final clustering failed for '{word}': {e}")
            return self._create_fallback_clustering(word_contexts, word)
        
        return {
            'word': word,
            'optimal_k': optimal_k,
            'best_silhouette': best_silhouette,
            'cluster_labels': final_labels,
            'all_silhouette_scores': silhouette_scores,
            'contexts': word_contexts,
            'embeddings': embeddings,
            'kmeans_model': final_kmeans
        }

    def _create_fallback_clustering(self, contexts: List[str], word: str) -> Dict:
        """Create fallback clustering for insufficient data"""
        n_contexts = len(contexts)
        fallback_labels = [0] * n_contexts  # Single cluster
        
        return {
            'word': word,
            'optimal_k': 1,
            'best_silhouette': 0.0,
            'cluster_labels': fallback_labels,
            'all_silhouette_scores': {},
            'contexts': contexts,
            'embeddings': np.zeros((n_contexts, 384)),  # Dummy embeddings
            'kmeans_model': None
        }

    def extract_fixed_features(self, word: str, clustering_results: Dict, decade_texts: Dict) -> Dict:
        """
        EXACT EQUATION 5: Fixed-Dimension Feature Vector
        
        EXACT FEATURE BREAKDOWN (Total: 49 features):
        - Basic clustering features: 5 features (silhouette_score, num_senses, cluster_separation, temporal_variance, normalized_inertia)
        - Word frequencies per decade: 18 features  
        - Average sense frequencies per decade: 18 features (using Equation 4)
        - New semantic features: 8 features
        TOTAL: 5 + 18 + 18 + 8 = 49 features (EXACT as per Equation 5)
        """
        
        features = {}
        decades = self.decades  # Use consistent decade list (18 decades)
        
        # EXACT EQUATION 4: Sense Frequency Averaging
        # avg_sense_freqᵈᵉᶜᵃᵈᵉ = (1/K*)Σₖ senseₖ_freqᵈᵉᶜᵃᵈᵉ
        sense_frequencies = self._calculate_sense_frequencies_per_decade(word, clustering_results, decade_texts)
        
        # EXACT IMPLEMENTATION of Equation 4: Average across ALL senses for each decade
        for decade in decades:
            decade_sense_freqs = sense_frequencies.get(decade, {})
            if decade_sense_freqs:
                # EXACT: "average frequency of all the sensors" using Equation 4
                K_star = len(decade_sense_freqs)  # K* = optimal number of senses
                avg_sense_freq = (1/K_star) * sum(decade_sense_freqs.values()) if K_star > 0 else 0.0
            else:
                avg_sense_freq = 0.0
            features[f'avg_sense_freq_{decade}'] = avg_sense_freq
        
        # Word frequency per decade (FIXED LENGTH = 18)
        word_frequencies = self._calculate_word_frequencies_per_decade(word, decade_texts)
        for decade in decades:
            features[f'word_freq_{decade}'] = word_frequencies.get(decade, 0.0)
        
        # Basic clustering quality metrics (FIXED LENGTH = 5)
        features['silhouette_score'] = clustering_results.get('best_silhouette', 0.0)
        features['num_senses'] = clustering_results.get('optimal_k', 1)
        features['cluster_separation'] = self._calculate_cluster_separation(clustering_results)
        features['temporal_variance'] = self._calculate_temporal_variance(sense_frequencies)
        features['normalized_inertia'] = self._calculate_normalized_inertia(clustering_results)
        
        # New semantic features (FIXED LENGTH = 8)
        features['entropy_change'] = self._calculate_entropy_change(sense_frequencies)
        features['context_diversity'] = self._calculate_context_diversity(clustering_results)
        features['syntactic_stability'] = self._calculate_syntactic_stability(word, clustering_results['contexts'])
        features['frequency_acceleration'] = self._calculate_frequency_acceleration(word_frequencies, decades)
        features['co_occurrence_shift'] = self._calculate_co_occurrence_shift(word, clustering_results, decade_texts)
        features['semantic_drift_velocity'] = self._calculate_semantic_drift_velocity(sense_frequencies, decades)
        features['cross_decade_overlap'] = self._calculate_cross_decade_overlap(clustering_results, decade_texts)
        features['cluster_stability'] = self._calculate_cluster_stability(clustering_results)
        
        # EXACT VERIFICATION: Correct feature count as per Equation 5
        expected_features = 5 + len(decades) * 2 + 8  # 5 + 18*2 + 8 = 49
        actual_features = len(features)
        
        if actual_features != expected_features:
            logger.error(f"EQUATION 5 VIOLATION: expected {expected_features} features, got {actual_features}")
            raise ValueError(f"Feature count mismatch violates Equation 5: expected {expected_features}, got {actual_features}")
        else:
            logger.info(f"Word '{word}': EQUATION 5 VERIFIED = {actual_features} features")
        
        return features

    def _features_dict_to_vector(self, features: Dict, decades: List[str]) -> np.ndarray:
        """Convert features dictionary to numpy vector with consistent ordering"""
        feature_vector = []
        
        # Basic clustering features (5 features)
        feature_vector.extend([
            features.get('silhouette_score', 0.0),
            features.get('num_senses', 1),
            features.get('cluster_separation', 0.0),
            features.get('temporal_variance', 0.0),
            features.get('normalized_inertia', 0.0)
        ])
        
        # Word frequencies per decade (18 features)
        for decade in decades:
            feature_vector.append(features.get(f'word_freq_{decade}', 0.0))
        
        # Average sense frequencies per decade (18 features)
        for decade in decades:
            feature_vector.append(features.get(f'avg_sense_freq_{decade}', 0.0))
        
        # Additional semantic features (8 features)
        feature_vector.extend([
            features.get('entropy_change', 0.0),
            features.get('context_diversity', 0.0),
            features.get('syntactic_stability', 0.0),
            features.get('frequency_acceleration', 0.0),
            features.get('co_occurrence_shift', 0.0),
            features.get('semantic_drift_velocity', 0.0),
            features.get('cross_decade_overlap', 0.0),
            features.get('cluster_stability', 0.0)
        ])
        
        return np.array(feature_vector)

    def _create_word_temporal_ground_truth(self, word: str, decade_texts: Dict) -> Dict[str, float]:
        """Create individual word temporal ground truth distribution"""
        word_frequencies = self._calculate_word_frequencies_per_decade(word, decade_texts)
        
        # Normalize to create probability distribution
        total_freq = sum(word_frequencies.values())
        if total_freq > 0:
            return {decade: freq / total_freq for decade, freq in word_frequencies.items()}
        else:
            # Uniform distribution if no data
            return {decade: 1.0 / len(self.decades) for decade in self.decades}

    def _calculate_sense_frequencies_per_decade(self, word: str, clustering_results: Dict, decade_texts: Dict) -> Dict:
        """Calculate frequency of each sense in each decade"""
        
        sense_frequencies = {}
        contexts = clustering_results['contexts']
        labels = clustering_results['cluster_labels']
        
        # Group contexts by sense
        sense_contexts = defaultdict(list)
        for context, label in zip(contexts, labels):
            sense_contexts[label].append(context)
        
        # Calculate frequencies per decade per sense
        for decade, texts in decade_texts.items():
            decade_sense_freqs = {}
            total_word_occurrences = 0
            
            # Count total word occurrences in this decade
            for text in texts:
                total_word_occurrences += len(re.findall(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE))
            
            # Count sense occurrences per decade
            for sense_id, sense_ctxs in sense_contexts.items():
                sense_count = 0
                for context in sense_ctxs:
                    # Check if this context appears in this decade's texts
                    for text in texts:
                        if context.strip().lower() in text.lower():
                            sense_count += 1
                            break
                
                # Normalize by total word occurrences
                decade_sense_freqs[sense_id] = sense_count / max(total_word_occurrences, 1)
            
            sense_frequencies[decade] = decade_sense_freqs
        
        return sense_frequencies

    def _precompute_corpus_statistics(self, decade_texts: Dict):
        """
        COMPREHENSIVE PRE-COMPUTATION: Build all necessary statistics in single corpus pass
        """
        logger.info("COMPREHENSIVE PRE-COMPUTATION: Building corpus statistics...")
        
        # Initialize data structures
        self.word_counts_per_decade = defaultdict(lambda: defaultdict(int))
        self.total_words_per_decade = defaultdict(int)
        self.word_cooccurrences_per_decade = defaultdict(lambda: defaultdict(set))
        self.context_to_decade_mapping = {}
        self.decade_text_lengths = defaultdict(int)
        
        # Single pass through entire corpus
        for decade, texts in decade_texts.items():
            logger.info(f"Pre-computing statistics for {decade} ({len(texts)} texts)...")
            
            for text_idx, text in enumerate(texts):
                # Count total words in this decade
                words = re.findall(r'\b\w+\b', text.lower())
                self.total_words_per_decade[decade] += len(words)
                self.decade_text_lengths[decade] += len(text)
                
                # Count target words and build co-occurrence sets
                for word in self.semantic_shift_words:
                    word_matches = re.findall(r'\b' + re.escape(word.lower()) + r'\b', text.lower())
                    self.word_counts_per_decade[word][decade] += len(word_matches)
                    
                    # Build co-occurrence set for this word in this decade
                    if word_matches:
                        # Extract all words from sentences containing target word
                        sentences = re.split(r'[.!?]+', text)
                        for sentence in sentences:
                            if re.search(r'\b' + re.escape(word.lower()) + r'\b', sentence.lower()):
                                sentence_words = re.findall(r'\b\w+\b', sentence.lower())
                                sentence_words = [w for w in sentence_words if w != word.lower() and len(w) > 2]
                                self.word_cooccurrences_per_decade[word][decade].update(sentence_words)
                
                # Build context-to-decade mapping for fast lookup
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    clean_sentence = re.sub(r'\s+', ' ', sentence.strip())
                    if 20 <= len(clean_sentence) <= 300:
                        self.context_to_decade_mapping[clean_sentence] = decade
        
        logger.info("COMPREHENSIVE PRE-COMPUTATION COMPLETE")

    def _calculate_word_frequencies_per_decade(self, word: str, decade_texts: Dict) -> Dict[str, float]:
        """Calculate frequency of word occurrences per decade"""
        word_frequencies = {}
        
        # Use pre-computed statistics if available
        if hasattr(self, 'word_counts_per_decade') and word in self.word_counts_per_decade:
            for decade in decade_texts.keys():
                word_count = self.word_counts_per_decade[word][decade]
                total_words = self.total_words_per_decade[decade]
                word_frequencies[decade] = word_count / max(total_words, 1)
        else:
            # Fallback to direct calculation
            for decade, texts in decade_texts.items():
                total_words = 0
                word_count = 0
                
                for text in texts:
                    # Count total words (approximate)
                    words = re.findall(r'\b\w+\b', text.lower())
                    total_words += len(words)
                    
                    # Count target word occurrences
                    word_matches = re.findall(r'\b' + re.escape(word.lower()) + r'\b', text.lower())
                    word_count += len(word_matches)
                
                # Calculate frequency
                word_frequencies[decade] = word_count / max(total_words, 1)
        
        return word_frequencies

    def _calculate_co_occurrence_shift(self, word: str, clustering_results: Dict, decade_texts: Dict) -> float:
        """Calculate how co-occurrence patterns change across decades"""
        
        # Use pre-computed co-occurrence sets if available
        if hasattr(self, 'word_cooccurrences_per_decade') and word in self.word_cooccurrences_per_decade:
            decade_cooccurrences = self.word_cooccurrences_per_decade[word]
        else:
            # Fallback to direct calculation
            decade_cooccurrences = {}
            for decade, texts in decade_texts.items():
                cooccurrence_words = set()
                for text in texts:
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if re.search(r'\b' + re.escape(word.lower()) + r'\b', sentence.lower()):
                            sentence_words = re.findall(r'\b\w+\b', sentence.lower())
                            sentence_words = [w for w in sentence_words if w != word.lower() and len(w) > 2]
                            cooccurrence_words.update(sentence_words)
                decade_cooccurrences[decade] = cooccurrence_words
        
        if not decade_cooccurrences:
            return 0.0
        
        # Calculate Jaccard similarity between consecutive decades
        similarities = []
        decade_list = sorted(decade_cooccurrences.keys())
        
        for i in range(1, len(decade_list)):
            prev_decade = decade_list[i-1]
            curr_decade = decade_list[i]
            
            prev_words = decade_cooccurrences[prev_decade]
            curr_words = decade_cooccurrences[curr_decade]
            
            if len(prev_words) == 0 and len(curr_words) == 0:
                similarity = 1.0
            elif len(prev_words) == 0 or len(curr_words) == 0:
                similarity = 0.0
            else:
                intersection = len(prev_words & curr_words)
                union = len(prev_words | curr_words)
                similarity = intersection / union if union > 0 else 0.0
            
            similarities.append(similarity)
        
        # Return inverse of mean similarity (higher shift = lower similarity)
        mean_similarity = np.mean(similarities) if similarities else 1.0
        return 1.0 - mean_similarity

    def _calculate_cross_decade_overlap(self, clustering_results: Dict, decade_texts: Dict) -> float:
        """Calculate how much contexts overlap across decades"""
        contexts = clustering_results.get('contexts', [])
        if not contexts:
            return 0.0
        
        # Use pre-computed context-to-decade mapping if available
        if hasattr(self, 'context_to_decade_mapping'):
            context_decades = {}
            for context in contexts:
                context_decades[context] = self.context_to_decade_mapping.get(context, None)
        else:
            # Fallback to direct calculation
            context_decades = {}
            for context in contexts:
                for decade, texts in decade_texts.items():
                    for text in texts:
                        if context.strip().lower() in text.lower():
                            context_decades[context] = decade
                            break
                    if context in context_decades:
                        break
        
        # Calculate overlap ratio
        if not contexts:
            return 0.0
        
        unique_decades = len(set(d for d in context_decades.values() if d is not None))
        overlap_ratio = unique_decades / len(decade_texts) if decade_texts else 0.0
        
        return min(overlap_ratio, 1.0)  # Cap at 1.0

    # Helper methods for mathematical feature calculation
    def _calculate_cluster_separation(self, clustering_results: Dict) -> float:
        """Calculate cluster separation metric"""
        embeddings = clustering_results.get('embeddings', np.array([]))
        labels = clustering_results.get('cluster_labels', [])
        
        if len(embeddings) == 0 or len(np.unique(labels)) <= 1:
            return 0.0
        
        # Calculate inter-cluster distances
        unique_labels = np.unique(labels)
        cluster_centers = []
        
        for label in unique_labels:
            cluster_points = embeddings[np.array(labels) == label]
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
        
        if len(cluster_centers) < 2:
            return 0.0
        
        # Calculate average pairwise distance between cluster centers
        distances = []
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0

    def _calculate_normalized_inertia(self, clustering_results: Dict) -> float:
        """Calculate normalized inertia"""
        embeddings = clustering_results.get('embeddings', np.array([]))
        labels = clustering_results.get('cluster_labels', [])
        
        if len(embeddings) == 0:
            return 0.0
        
        # Calculate within-cluster sum of squares
        inertia = 0.0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_points = embeddings[np.array(labels) == label]
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                inertia += np.sum((cluster_points - center) ** 2)
        
        # Normalize by number of points and dimensions
        n_points = len(embeddings)
        n_dims = embeddings.shape[1] if len(embeddings.shape) > 1 else 1
        
        return inertia / (n_points * n_dims) if n_points > 0 and n_dims > 0 else 0.0

    def _calculate_temporal_variance(self, sense_frequencies: Dict) -> float:
        """Calculate temporal variance of sense frequencies"""
        if not sense_frequencies:
            return 0.0
        
        # Calculate variance across decades for each sense
        variances = []
        all_senses = set()
        
        for decade_freqs in sense_frequencies.values():
            all_senses.update(decade_freqs.keys())
        
        for sense in all_senses:
            sense_values = []
            for decade_freqs in sense_frequencies.values():
                sense_values.append(decade_freqs.get(sense, 0.0))
            
            if len(sense_values) > 1:
                variances.append(np.var(sense_values))
        
        return np.mean(variances) if variances else 0.0

    def _calculate_entropy_change(self, sense_frequencies: Dict) -> float:
        """Calculate entropy change over time"""
        if not sense_frequencies:
            return 0.0
        
        decades = sorted(sense_frequencies.keys())
        if len(decades) < 2:
            return 0.0
        
        # Calculate entropy for first and last decades
        first_decade = decades[0]
        last_decade = decades[-1]
        
        def calculate_entropy(freq_dict):
            if not freq_dict:
                return 0.0
            values = list(freq_dict.values())
            total = sum(values)
            if total == 0:
                return 0.0
            probs = [v / total for v in values]
            return entropy(probs)
        
        first_entropy = calculate_entropy(sense_frequencies[first_decade])
        last_entropy = calculate_entropy(sense_frequencies[last_decade])
        
        return last_entropy - first_entropy

    def _calculate_context_diversity(self, clustering_results: Dict) -> float:
        """Calculate context diversity metric"""
        contexts = clustering_results.get('contexts', [])
        
        if not contexts:
            return 0.0
        
        # Simple diversity metric: average sentence length variation
        lengths = [len(ctx.split()) for ctx in contexts]
        
        if len(lengths) < 2:
            return 0.0
        
        return np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0.0

    def _calculate_syntactic_stability(self, word: str, contexts: List[str]) -> float:
        """Calculate syntactic stability of word usage"""
        if not contexts:
            return 0.0
        
        # Simple POS pattern analysis
        pos_patterns = []
        
        for context in contexts:
            # Extract words around the target word
            words = context.lower().split()
            try:
                word_idx = words.index(word.lower())
                
                # Get surrounding context
                start = max(0, word_idx - 2)
                end = min(len(words), word_idx + 3)
                pattern = ' '.join(words[start:end])
                pos_patterns.append(pattern)
            except ValueError:
                continue
        
        if not pos_patterns:
            return 0.0
        
        # Calculate pattern diversity (inverse of stability)
        unique_patterns = len(set(pos_patterns))
        total_patterns = len(pos_patterns)
        
        return 1.0 - (unique_patterns / total_patterns) if total_patterns > 0 else 0.0

    def _calculate_frequency_acceleration(self, word_frequencies: Dict[str, float], decades: List[str]) -> float:
        """Calculate frequency acceleration over time"""
        if len(decades) < 3:
            return 0.0
        
        # Calculate second derivative of frequency over time
        freq_values = [word_frequencies.get(decade, 0.0) for decade in decades]
        
        # Simple numerical second derivative
        accelerations = []
        for i in range(1, len(freq_values) - 1):
            accel = freq_values[i+1] - 2*freq_values[i] + freq_values[i-1]
            accelerations.append(accel)
        
        return np.mean(accelerations) if accelerations else 0.0

    def _calculate_semantic_drift_velocity(self, sense_frequencies: Dict, decades: List[str]) -> float:
        """Calculate semantic drift velocity"""
        if len(decades) < 2:
            return 0.0
        
        # Calculate rate of change in sense distribution
        drift_rates = []
        
        for i in range(len(decades) - 1):
            curr_decade = decades[i]
            next_decade = decades[i + 1]
            
            curr_freqs = sense_frequencies.get(curr_decade, {})
            next_freqs = sense_frequencies.get(next_decade, {})
            
            # Calculate distribution change
            all_senses = set(curr_freqs.keys()) | set(next_freqs.keys())
            
            if all_senses:
                changes = []
                for sense in all_senses:
                    curr_val = curr_freqs.get(sense, 0.0)
                    next_val = next_freqs.get(sense, 0.0)
                    changes.append(abs(next_val - curr_val))
                
                drift_rates.append(np.mean(changes))
        
        return np.mean(drift_rates) if drift_rates else 0.0

    def _calculate_cluster_stability(self, clustering_results: Dict) -> float:
        """Calculate cluster stability metric"""
        embeddings = clustering_results.get('embeddings', np.array([]))
        labels = clustering_results.get('cluster_labels', [])
        
        if len(embeddings) == 0 or len(np.unique(labels)) <= 1:
            return 0.0
        
        # Calculate average intra-cluster distance
        unique_labels = np.unique(labels)
        intra_distances = []
        
        for label in unique_labels:
            cluster_points = embeddings[np.array(labels) == label]
            if len(cluster_points) > 1:
                center = np.mean(cluster_points, axis=0)
                distances = [np.linalg.norm(point - center) for point in cluster_points]
                intra_distances.extend(distances)
        
        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0.0
        
        # Stability is inverse of average intra-cluster distance
        return 1.0 / (1.0 + avg_intra_distance)


class TemporalRegressionModel:
    """
    A wrapper for a regression model to predict temporal distributions.
    """
    def __init__(self, model_type: str = "linear"):
        if model_type == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError("Unsupported model type")
        self.is_trained = False

    def train(self, X_features: np.ndarray, Y_targets: np.ndarray):
        """Trains the regression model."""
        self.model.fit(X_features, Y_targets)
        self.is_trained = True
        logger.info("Temporal regression model trained successfully.")

    def predict(self, X_features: np.ndarray) -> np.ndarray:
        """Makes predictions with the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(X_features)

    def evaluate(self, X_features: np.ndarray, Y_targets: np.ndarray) -> dict:
        """Evaluates the model and returns performance metrics."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        predictions = self.predict(X_features)
        mse = mean_squared_error(Y_targets, predictions)
        r2 = r2_score(Y_targets, predictions)
        
        return {
            'mean_squared_error': mse,
            'r_squared': r2
        }

    def train_multi_output_model(self, X_features: np.ndarray, Y_targets: np.ndarray, decades: List[str]) -> Dict:
        """Train multi-output regression model"""
        try:
            self.model.fit(X_features, Y_targets)
            self.is_trained = True
            
            # Evaluate training performance
            y_pred = self.model.predict(X_features)
            mse = mean_squared_error(Y_targets, y_pred)
            r2 = r2_score(Y_targets, y_pred)
            
            return {
                'mse': mse,
                'r2': r2,
                'model_coefficients': self.model.coef_.tolist() if hasattr(self.model, 'coef_') else None,
                'model_intercept': self.model.intercept_.tolist() if hasattr(self.model, 'intercept_') else None,
                'training_samples': X_features.shape[0],
                'feature_dimensions': X_features.shape[1],
                'target_dimensions': Y_targets.shape[1] if len(Y_targets.shape) > 1 else 1
            }
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    print("STEP 3: Training Semantic Enhancement Model...")
    print("This may take several minutes depending on dataset size...")