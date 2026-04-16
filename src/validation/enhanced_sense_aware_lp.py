#!/usr/bin/env python3
"""
Enhanced Sense-Aware Linear Programming for Temporal Distribution Inference

FULLY ENHANCED IMPLEMENTATION with:
1. Comprehensive vocabulary from legitimate research sources (60+ words)
2. Automatic semantic shift detection capabilities
3. Proper ML-based sense classifiers (Logistic Regression + SVM)
4. Full CVXPY linear programming optimization 

This implements the COMPLETE approach with research-backed vocabulary.
"""

import logging
import numpy as np
import cvxpy as cp
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re
import time

# Import our comprehensive semantic shifts database
from .comprehensive_semantic_shifts import get_all_semantic_shifts, get_high_confidence_shifts
from .automatic_shift_detection import run_automatic_detection, AutoDetectedShift

logger = logging.getLogger(__name__)

class EnhancedSenseAwareTemporalInference:
    """
    Enhanced sense-aware temporal distribution inference using:
    - Comprehensive research-backed vocabulary (60+ semantic shift words)
    - Automatic semantic shift detection
    - ML-based sense classification (LR + SVM)
    - Full linear programming optimization with CVXPY
    """
    
    def __init__(self, tokenizer_name: str = "gpt2", use_automatic_detection: bool = True):
        """
        Initialize the enhanced sense-aware temporal inference system.
        
        Args:
            tokenizer_name: Name of tokenizer to analyze
            use_automatic_detection: Whether to include automatically detected shifts
        """
        logger.info("=== INITIALIZING ENHANCED SENSE-AWARE TEMPORAL INFERENCE ===")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name
        
        # Load comprehensive semantic shift words from research sources
        self.semantic_shift_words = self._load_comprehensive_vocabulary()
        
        # Storage for models and results
        self.trained_classifiers = {}
        self.lp_model = None
        self.sense_frequencies = {}
        self.automatic_shifts = []
        self.use_automatic_detection = use_automatic_detection
        
        logger.info(f"Loaded {len(self.semantic_shift_words)} research-backed semantic shift words")
        
    def _load_comprehensive_vocabulary(self) -> Dict[str, Dict]:
        """Load comprehensive semantic shift vocabulary from research sources."""
        # Get all high-confidence shifts from research
        research_shifts = get_high_confidence_shifts(min_confidence=0.8)
        
        # Convert to our internal format
        vocabulary = {}
        for word, data in research_shifts.items():
            vocabulary[word] = {
                "senses": data["senses"],
                "transition": data["transition"],
                "markers": data["markers"],
                "source": data.get("source", "Research"),
                "shift_type": data.get("shift_type", "unknown"),
                "confidence": data.get("confidence", 0.8)
            }
        
        logger.info(f"Loaded {len(vocabulary)} high-confidence research-backed semantic shifts")
        return vocabulary
    
    def add_automatic_shifts(self, decade_texts: Dict[str, List[str]], top_n: int = 20):
        """
        Add automatically detected semantic shifts to the vocabulary.
        
        Args:
            decade_texts: Training decade texts for automatic detection
            top_n: Number of top automatic detections to include
        """
        if not self.use_automatic_detection:
            return
            
        logger.info("=== RUNNING AUTOMATIC SEMANTIC SHIFT DETECTION ===")
        
        # Run automatic detection
        detected_shifts = run_automatic_detection(decade_texts, top_n=top_n)
        
        # Convert high-confidence automatic detections to our format
        for shift in detected_shifts:
            if shift.confidence > 0.7:  # High confidence threshold
                # Create sense definitions based on evidence
                early_contexts = [ctx for ctx in shift.evidence_contexts if "[Early]" in ctx]
                late_contexts = [ctx for ctx in shift.evidence_contexts if "[Late]" in ctx]
                
                # Extract marker words from contexts (simplified)
                early_markers = self._extract_markers_from_contexts(early_contexts)
                late_markers = self._extract_markers_from_contexts(late_contexts)
                
                self.semantic_shift_words[shift.word] = {
                    "senses": ["early_sense", "late_sense"],
                    "transition": shift.time_periods,
                    "markers": {
                        "early_sense": early_markers[:7],
                        "late_sense": late_markers[:7]
                    },
                    "source": "Automatic Detection",
                    "shift_type": "detected",
                    "confidence": shift.confidence,
                    "automatic": True
                }
                
                self.automatic_shifts.append(shift)
        
        logger.info(f"Added {len(self.automatic_shifts)} automatically detected shifts")
        logger.info(f"Total vocabulary size: {len(self.semantic_shift_words)} words")
    
    def _extract_markers_from_contexts(self, contexts: List[str]) -> List[str]:
        """Extract characteristic words from context examples."""
        if not contexts:
            return ["context", "usage", "meaning"]
            
        # Simple extraction of frequent words from contexts
        all_words = []
        for context in contexts[:5]:  # Sample contexts
            words = re.findall(r'\\b[a-z]{4,12}\\b', context.lower())
            all_words.extend(words)
        
        # Get most frequent non-stopwords
        word_counts = Counter(all_words)
        stopwords = {"this", "that", "with", "from", "they", "were", "been", "have", "had", "will", "would", "could", "should", "early", "late"}
        
        markers = []
        for word, count in word_counts.most_common(10):
            if word not in stopwords and len(word) > 3:
                markers.append(word)
                if len(markers) >= 7:
                    break
        
        return markers if markers else ["usage", "context", "meaning"]

    def train_enhanced_ml_classifiers(self, training_decade_texts: Dict[str, List[str]]) -> Dict:
        """
        Train proper ML classifiers for sense disambiguation.
        Uses TF-IDF features + Logistic Regression/SVM with cross-validation.
        
        Returns:
            Training results and classifier performance metrics
        """
        logger.info("=== TRAINING ENHANCED ML CLASSIFIERS ===")
        
        # Extract training data for all semantic shift words
        training_data = self._extract_classifier_training_data(training_decade_texts)
        
        classifier_results = {}
        successful_words = 0
        
        for word, word_data in training_data.items():
            if len(word_data['contexts']) < 30:  # Minimum training examples
                logger.warning(f"Insufficient training data for '{word}': {len(word_data['contexts'])} examples")
                continue
            
            logger.info(f"Training classifiers for word '{word}' with {len(word_data['contexts'])} examples")
            
            # Train multiple models and select best
            best_model, best_score, model_comparison = self._train_and_select_best_classifier(
                word, word_data['contexts'], word_data['labels']
            )
            
            if best_model is not None and best_score > 0.6:
                self.trained_classifiers[word] = {
                    'model': best_model,
                    'performance': {
                        'cv_score': best_score,
                        'model_comparison': model_comparison,
                        'training_size': len(word_data['contexts'])
                    },
                    'senses': self.semantic_shift_words[word]['senses'],
                    'source': self.semantic_shift_words[word].get('source', 'Unknown')
                }
                successful_words += 1
                logger.info(f"  ✓ Trained classifier for '{word}': {best_score:.3f} CV score")
            else:
                logger.warning(f"  ✗ Failed to train reliable classifier for '{word}'")
        
        classifier_results = {
            'total_words': len(training_data),
            'successful_classifiers': successful_words,
            'success_rate': successful_words / len(training_data) if training_data else 0,
            'classifiers': self.trained_classifiers
        }
        
        logger.info(f"Successfully trained {successful_words}/{len(training_data)} classifiers")
        return classifier_results
    
    def _extract_classifier_training_data(self, decade_texts: Dict[str, List[str]]) -> Dict:
        """Extract training contexts and labels for all semantic shift words."""
        training_data = {}
        
        for word in self.semantic_shift_words:
            logger.debug(f"Extracting training data for '{word}'")
            
            contexts = []
            labels = []
            
            for decade, texts in decade_texts.items():
                decade_year = int(decade.replace('s', ''))
                
                # Extract contexts for this word in this decade
                for text in texts[:20]:  # Sample texts for efficiency
                    word_contexts = self._extract_word_contexts(text, word, window_size=50)
                    
                    for context in word_contexts:
                        # Create enhanced label using multiple signals
                        label = self._create_enhanced_label(word, context, decade_year)
                        if label is not None:
                            contexts.append(context)
                            labels.append(label)
            
            if len(contexts) >= 20:  # Minimum contexts needed
                training_data[word] = {
                    'contexts': contexts,
                    'labels': labels
                }
        
        return training_data
    
    def _create_enhanced_label(self, word: str, context: str, decade_year: int) -> Optional[int]:
        """
        Create enhanced sense labels using multiple signals:
        1. Temporal priors based on transition periods
        2. Contextual markers
        3. Transition likelihood
        """
        if word not in self.semantic_shift_words:
            return None
        
        word_info = self.semantic_shift_words[word]
        transition_start, transition_end = word_info["transition"]
        senses = word_info["senses"]
        markers = word_info["markers"]
        
        # Signal 1: Temporal prior
        if decade_year < transition_start:
            temporal_prior = 0  # Old sense
        elif decade_year > transition_end:
            temporal_prior = 1  # New sense
        else:
            # During transition - use smooth interpolation
            progress = (decade_year - transition_start) / (transition_end - transition_start)
            temporal_prior = 0 if progress < 0.5 else 1
        
        # Signal 2: Contextual markers
        context_lower = context.lower()
        old_markers = markers.get(senses[0], [])
        new_markers = markers.get(senses[1], [])
        
        old_marker_score = sum(1 for marker in old_markers if marker in context_lower)
        new_marker_score = sum(1 for marker in new_markers if marker in context_lower)
        
        # Combine temporal and contextual signals
        if old_marker_score > new_marker_score + 1:
            return 0  # Strong old sense signal
        elif new_marker_score > old_marker_score + 1:
            return 1  # Strong new sense signal
        else:
            return temporal_prior  # Fall back to temporal prior
    
    def _train_and_select_best_classifier(self, word: str, contexts: List[str], labels: List[int]) -> Tuple:
        """
        Train multiple classifier models and select the best one via cross-validation.
        """
        try:
            # Create TF-IDF features
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                max_df=0.95,
                min_df=2
            )
            
            X = vectorizer.fit_transform(contexts)
            y = np.array(labels)
            
            # Check for class balance
            unique_labels, counts = np.unique(y, return_counts=True)
            if len(unique_labels) < 2 or min(counts) < 5:
                logger.warning(f"Insufficient class balance for '{word}': {dict(zip(unique_labels, counts))}")
                return None, 0.0, {}
            
            # Define candidate models
            models = {
                'logistic_regression': Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
                ]),
                'svm': Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('classifier', SVC(class_weight='balanced', random_state=42, probability=True))
                ])
            }
            
            # Cross-validation comparison
            model_scores = {}
            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=min(5, len(y)//3), scoring='f1_macro')
                    model_scores[name] = {
                        'mean_score': scores.mean(),
                        'std_score': scores.std(),
                        'scores': scores.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Model {name} failed for word '{word}': {e}")
                    model_scores[name] = {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}
            
            # Select best model
            best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['mean_score'])
            best_score = model_scores[best_model_name]['mean_score']
            
            if best_score > 0.5:  # Minimum acceptable performance
                best_model = models[best_model_name]
                best_model.fit(X, y)
                
                # Store vectorizer with model
                best_model.vectorizer = vectorizer
                
                return best_model, best_score, model_scores
            else:
                return None, 0.0, model_scores
                
        except Exception as e:
            logger.error(f"Classifier training failed for word '{word}': {e}")
            return None, 0.0, {}

    def _extract_word_contexts(self, text: str, word: str, window_size: int = 50) -> List[str]:
        """Extract contexts around word occurrences."""
        contexts = []
        text_lower = text.lower()
        word_lower = word.lower()
        
        # Find all occurrences
        start = 0
        while True:
            pos = text_lower.find(word_lower, start)
            if pos == -1:
                break
            
            # Extract context window
            context_start = max(0, pos - window_size)
            context_end = min(len(text), pos + len(word) + window_size)
            context = text[context_start:context_end].strip()
            
            if len(context) > 20:  # Minimum context length
                contexts.append(context)
            
            start = pos + 1
            
            if len(contexts) >= 30:  # Limit contexts per text
                break
        
        return contexts

    def train_enhanced_lp_model(self, training_decade_texts: Dict[str, List[str]]) -> Dict:
        """
        Train the enhanced linear programming model using sense frequencies.
        """
        logger.info("=== TRAINING ENHANCED LP MODEL ===")
        
        # Compute sense frequencies using ML classifiers
        ml_sense_frequencies = self._compute_ml_sense_frequencies(training_decade_texts)
        
        # Store sense frequencies
        self.sense_frequencies = ml_sense_frequencies
        
        # Train LP model using CVXPY optimization
        training_decades = sorted(training_decade_texts.keys())
        lp_results = self._train_enhanced_lp_optimization(training_decades)
        
        return {
            'sense_frequencies': ml_sense_frequencies,
            'lp_model': lp_results,
            'training_decades': training_decades,
            'vocabulary_size': len(self.semantic_shift_words)
        }

    def _compute_ml_sense_frequencies(self, decade_texts: Dict[str, List[str]]) -> Dict:
        """
        Compute sense frequencies using trained ML classifiers.
        """
        logger.info("Computing ML-based sense frequencies...")
        
        sense_frequencies = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        for decade, texts in decade_texts.items():
            logger.debug(f"Processing decade {decade}")
            
            for word, classifier_data in self.trained_classifiers.items():
                classifier = classifier_data['model']
                senses = classifier_data['senses']
                
                # Extract contexts for this word
                all_contexts = []
                for text in texts[:30]:  # Sample texts
                    contexts = self._extract_word_contexts(text, word)
                    all_contexts.extend(contexts)
                
                if not all_contexts:
                    continue
                
                # Classify senses using trained model
                try:
                    X = classifier.vectorizer.transform(all_contexts)
                    predictions = classifier.predict(X)
                    probabilities = classifier.predict_proba(X)
                    
                    # Compute sense frequencies
                    for sense_idx, sense in enumerate(senses):
                        # Use probability-weighted frequencies
                        sense_freq = np.sum(probabilities[:, sense_idx])
                        sense_frequencies[word][sense][decade] = sense_freq
                        
                except Exception as e:
                    logger.warning(f"Sense classification failed for '{word}' in {decade}: {e}")
        
        return dict(sense_frequencies)

    def _train_enhanced_lp_optimization(self, training_decades: List[str]) -> Dict:
        """
        Train enhanced LP model using CVXPY for proper convex optimization.
        """
        logger.info("Training enhanced LP optimization model...")
        
        try:
            # Prepare data matrices
            word_sense_pairs = []
            frequency_matrix = []
            
            for word, sense_data in self.sense_frequencies.items():
                for sense, decade_freqs in sense_data.items():
                    word_sense_pairs.append((word, sense))
                    
                    # Create frequency vector for this word-sense pair
                    freq_vector = []
                    for decade in training_decades:
                        freq = decade_freqs.get(decade, 0.0)
                        freq_vector.append(freq)
                    
                    frequency_matrix.append(freq_vector)
            
            if not frequency_matrix:
                logger.error("No frequency data available for LP training")
                return {'status': 'failed', 'error': 'No data'}
            
            F = np.array(frequency_matrix)  # Feature matrix (word-senses × decades)
            n_decades = len(training_decades)
            n_word_senses = len(word_sense_pairs)
            
            logger.info(f"Training LP with {n_word_senses} word-sense pairs across {n_decades} decades")
            
            # Enhanced CVXPY optimization
            # Variables: temporal distribution weights
            w = cp.Variable(n_decades, nonneg=True)
            
            # Objective: Minimize reconstruction error with L2 regularization
            reconstruction_error = cp.sum_squares(F @ w - 1.0)  # Target unit reconstruction
            regularization = 0.01 * cp.sum_squares(w)  # L2 regularization
            objective = cp.Minimize(reconstruction_error + regularization)
            
            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1 (probability distribution)
                w >= 0.001,      # Minimum positive weights
                w <= 0.8         # Maximum weight per decade (avoid over-concentration)
            ]
            
            # Additional smoothness constraint (optional)
            if n_decades > 2:
                # Penalize large differences between adjacent decades
                smoothness_penalty = 0.001 * cp.sum_squares(w[1:] - w[:-1])
                objective = cp.Minimize(reconstruction_error + regularization + smoothness_penalty)
            
            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = w.value
                
                # Store model
                self.lp_model = {
                    'weights': optimal_weights,
                    'training_decades': training_decades,
                    'word_sense_pairs': word_sense_pairs,
                    'frequency_matrix': F,
                    'reconstruction_error': reconstruction_error.value,
                    'objective_value': problem.value
                }
                
                logger.info(f"LP optimization completed successfully")
                logger.info(f"  Reconstruction error: {reconstruction_error.value:.4f}")
                logger.info(f"  Objective value: {problem.value:.4f}")
                
                return {
                    'status': 'success',
                    'weights': optimal_weights,
                    'reconstruction_error': reconstruction_error.value,
                    'training_decades': training_decades,
                    'n_word_senses': n_word_senses
                }
            else:
                logger.error(f"LP optimization failed with status: {problem.status}")
                return {'status': 'failed', 'error': problem.status}
                
        except Exception as e:
            logger.error(f"LP training failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def apply_to_test_data(self, test_decade_texts: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Apply the trained enhanced model to test data.
        """
        logger.info("=== APPLYING ENHANCED MODEL TO TEST DATA ===")
        
        if self.lp_model is None:
            raise ValueError("Model must be trained before applying to test data")
        
        # Compute sense frequencies for test data
        test_sense_frequencies = self._compute_ml_sense_frequencies(test_decade_texts)
        
        # Apply LP model
        temporal_distribution = self._apply_enhanced_lp_model(test_sense_frequencies)
        
        return temporal_distribution

    def _apply_enhanced_lp_model(self, test_sense_frequencies: Dict) -> Dict[str, float]:
        """
        Apply the trained LP model to compute temporal distribution from test sense frequencies.
        """
        logger.info("Applying enhanced LP model...")
        
        try:
            # Reconstruct test frequency matrix
            word_sense_pairs = self.lp_model['word_sense_pairs']
            training_decades = self.lp_model['training_decades']
            
            test_frequency_matrix = []
            
            for word, sense in word_sense_pairs:
                freq_vector = []
                
                if word in test_sense_frequencies and sense in test_sense_frequencies[word]:
                    sense_data = test_sense_frequencies[word][sense]
                    
                    for decade in training_decades:
                        freq = sense_data.get(decade, 0.0)
                        freq_vector.append(freq)
                else:
                    # Word/sense not found in test data
                    freq_vector = [0.0] * len(training_decades)
                
                test_frequency_matrix.append(freq_vector)
            
            F_test = np.array(test_frequency_matrix)
            
            # Use trained weights to predict temporal distribution
            if F_test.shape[0] > 0 and np.sum(F_test) > 0:
                # Solve for temporal distribution that best explains test frequencies
                w_train = self.lp_model['weights']
                
                # Compute weighted contribution of each decade
                temporal_scores = F_test.T @ w_train  # Decade-wise scores
                
                # Normalize to get distribution
                if np.sum(temporal_scores) > 0:
                    temporal_distribution = temporal_scores / np.sum(temporal_scores)
                else:
                    # Uniform fallback
                    temporal_distribution = np.ones(len(training_decades)) / len(training_decades)
            else:
                # No test data - use uniform distribution
                temporal_distribution = np.ones(len(training_decades)) / len(training_decades)
            
            # Convert to dictionary
            result = {}
            for i, decade in enumerate(training_decades):
                result[decade] = float(temporal_distribution[i])
            
            logger.info("Enhanced LP model applied successfully")
            return result
            
        except Exception as e:
            logger.error(f"LP model application failed: {e}")
            # Return uniform distribution as fallback
            n_decades = len(self.lp_model['training_decades'])
            return {decade: 1.0/n_decades for decade in self.lp_model['training_decades']}

    def _map_subtokens_to_words(self, decade_texts: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Map BPE subtokens to semantic shift words for integration.
        """
        logger.info("Mapping BPE subtokens to semantic shift words...")
        
        subtoken_word_mapping = defaultdict(lambda: defaultdict(float))
        
        # Get BPE vocabulary
        vocab = self.tokenizer.get_vocab()
        
        # For each semantic shift word, find related subtokens
        for word in self.semantic_shift_words:
            # Tokenize the word
            word_tokens = self.tokenizer.tokenize(word)
            
            # Direct mapping for word tokens
            for token in word_tokens:
                if token in vocab:
                    subtoken_word_mapping[token][word] = 1.0
            
            # Also check for partial matches in vocabulary
            for vocab_token in vocab:
                if word.lower() in vocab_token.lower() or vocab_token.lower() in word.lower():
                    if len(vocab_token) > 2:  # Avoid too short tokens
                        subtoken_word_mapping[vocab_token][word] += 0.5
        
        return dict(subtoken_word_mapping)

    def run_complete_enhanced_analysis(self, training_texts: Dict[str, List[str]], 
                                     test_texts: Dict[str, List[str]]) -> Dict:
        """
        Run the complete enhanced analysis pipeline.
        """
        start_time = time.time()
        
        logger.info("=== STARTING COMPLETE ENHANCED ANALYSIS ===")
        logger.info(f"Training decades: {list(training_texts.keys())}")
        logger.info(f"Test decades: {list(test_texts.keys())}")
        logger.info(f"Vocabulary size: {len(self.semantic_shift_words)} words")
        
        results = {
            'metadata': {
                'tokenizer': self.tokenizer_name,
                'vocabulary_size': len(self.semantic_shift_words),
                'training_decades': list(training_texts.keys()),
                'test_decades': list(test_texts.keys()),
                'automatic_detection': self.use_automatic_detection
            }
        }
        
        # Step 1: Add automatic shifts if enabled
        if self.use_automatic_detection:
            self.add_automatic_shifts(training_texts)
            results['automatic_shifts'] = len(self.automatic_shifts)
        
        # Step 2: Train ML classifiers
        classifier_results = self.train_enhanced_ml_classifiers(training_texts)
        results['classifier_training'] = classifier_results
        
        # Step 3: Train LP model
        lp_results = self.train_enhanced_lp_model(training_texts)
        results['lp_training'] = lp_results
        
        # Step 4: Apply to test data
        temporal_distribution = self.apply_to_test_data(test_texts)
        results['temporal_distribution'] = temporal_distribution
        
        # Step 5: Additional analysis
        results['subtoken_mapping'] = self._map_subtokens_to_words(training_texts)
        
        # Timing and summary
        total_time = time.time() - start_time
        results['execution_time'] = total_time
        
        logger.info("=== ENHANCED ANALYSIS COMPLETE ===")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Successful classifiers: {classifier_results.get('successful_classifiers', 0)}")
        logger.info(f"Temporal distribution: {temporal_distribution}")
        
        return results

# Convenience function for external use
def run_enhanced_temporal_analysis(training_texts: Dict[str, List[str]], 
                                 test_texts: Dict[str, List[str]], 
                                 tokenizer_name: str = "gpt2",
                                 use_automatic_detection: bool = True) -> Dict:
    """
    Convenience function to run the complete enhanced temporal analysis.
    
    Args:
        training_texts: Training decade texts
        test_texts: Test decade texts  
        tokenizer_name: Tokenizer to analyze
        use_automatic_detection: Whether to include automatic shift detection
        
    Returns:
        Complete analysis results
    """
    analyzer = EnhancedSenseAwareTemporalInference(
        tokenizer_name=tokenizer_name,
        use_automatic_detection=use_automatic_detection
    )
    
    return analyzer.run_complete_enhanced_analysis(training_texts, test_texts) 