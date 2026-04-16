#!/usr/bin/env python3
"""
Final, Stable Temporal Inference Pipeline using N-grams and Subtokens

This pipeline implements a robust, stable methodology for temporal inference
by combining two distinct and computationally stable signals in a Linear
Programming framework:
1.  Subtoken Frequencies: A signal based on how words are constructed by a
    modern tokenizer.
2.  Character N-gram Frequencies: A stylistic signal that captures common
    character patterns (e.g., suffixes, common letter combinations).

This approach avoids the unstable dependencies that caused crashes while
maintaining the core principle of combining multiple linguistic signals.

The pipeline flow is as follows:
- DATA SPLIT: The data is split into a TRAINING set (past) and a TEST set (future).
- FEATURE EXTRACTION (TRAINING SET):
    1. A master feature list of the most important subtokens and n-grams is created.
    2. For each decade, a combined feature profile is computed.
- LP MODELING (TEST SET):
    3. A combined feature profile is computed for the test data.
    4. An LP problem is formulated to find the temporal distribution that best
       reconstructs the test set's profile from the training profiles.
"""

import sys
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import argparse

from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import cvxpy as cp

# Add project root to path and import local modules
sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import load_all_temporal_data

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MAIN PIPELINE CLASS ---

class FinalStablePipeline:
    """
    Implements the final, stable temporal inference pipeline.
    """
    
    def __init__(self, all_temporal_data: Dict[str, List[str]]):
        self.all_data = all_temporal_data
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.ngram_vectorizer = None
        
        # Data structures
        self.master_feature_list: List[str] = []
        self.training_profiles: Dict[str, np.ndarray] = {}

    def run_complete_pipeline(self, test_file_path: str):
        """Orchestrates the entire pipeline from discovery to inference."""
        print("=" * 80)
        print("RUNNING FINAL, STABLE TEMPORAL INFERENCE PIPELINE")
        print(f"Test File: {test_file_path}")
        print("=" * 80)
        
        logger.info("Splitting data for training and testing...")
        # Training data is now all historical data loaded. Test data comes from the file.
        train_data = {d: self.all_data[d] for d in self.all_data if d != 'test'}

        self._run_feature_extraction(train_data)
        
        # Load test data from the specified file
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_texts = [f.read()]
        except FileNotFoundError:
            logger.error(f"Test file not found: {test_file_path}")
            return None

        results = self._run_test_and_solve(test_texts)
        
        return results
    
    def _run_feature_extraction(self, train_data: Dict[str, List[str]]):
        """
        Extracts all necessary features (subtoken and n-gram profiles)
        from the historical training data.
        """
        print("\n" + "=" * 60)
        print("PHASE 1: FEATURE EXTRACTION (Training Data)")
        print("=" * 60)

        # Step 1: Establish the master feature list
        self._create_master_feature_lists(train_data)

        # Step 2: Build the combined profile for each training decade
        for decade, texts in train_data.items():
            logger.info(f"Building feature profile for {decade}...")
            
            combined_profile = self._compute_combined_profile(texts)
            self.training_profiles[decade] = combined_profile
            logger.info(f"  - Profile for {decade} created with {len(combined_profile)} features.")

    def _create_master_feature_lists(self, train_data):
        """Creates the master lists of all features."""
        # N-gram features
        self.ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=500)
        all_train_text = [" ".join(texts) for texts in train_data.values()]
        self.ngram_vectorizer.fit(all_train_text)
        ngram_features = self.ngram_vectorizer.get_feature_names_out()
        logger.info(f"Created master list with {len(ngram_features)} n-gram features.")

        # Subtoken features (get most common from training data)
        all_tokens = []
        for texts in train_data.values():
            content = " ".join(texts)
            all_tokens.extend(self.tokenizer.tokenize(content.lower()))
        
        subtoken_counts = Counter(all_tokens)
        subtoken_features = [item[0] for item in subtoken_counts.most_common(500)]
        logger.info(f"Created master list with {len(subtoken_features)} subtoken features.")
        
        self.master_feature_list = list(ngram_features) + list(subtoken_features)


    def _compute_combined_profile(self, texts: List[str]) -> np.ndarray:
        """Computes a normalized frequency vector for all features."""
        content = " ".join(texts)
        
        # N-gram profile
        ngram_counts = self.ngram_vectorizer.transform([content]).toarray().flatten()
        
        # Subtoken profile
        tokens = self.tokenizer.tokenize(content.lower())
        token_counts = Counter(tokens)
        subtoken_features = [f for f in self.master_feature_list if not f.startswith(' ')] # A bit hacky way to separate
        subtoken_profile = np.array([token_counts.get(sub, 0) for sub in subtoken_features])
        
        # Combine and normalize
        combined_profile = np.concatenate([ngram_counts, subtoken_profile])
        return combined_profile / (np.sum(combined_profile) + 1e-9)

    def _run_test_and_solve(self, test_texts: List[str]):
        """
        Computes the feature profile for the test set and solves the LP problem.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: TESTING & LP SOLVING")
        print("=" * 60)

        test_profile = self._compute_combined_profile(test_texts)
        logger.info(f"Test profile created with {len(test_profile)} features.")

        train_decades = sorted(self.training_profiles.keys())
        F = np.array([self.training_profiles[d] for d in train_decades]).T
        t = test_profile

        if F.shape[1] == 0:
            logger.error("Training failed to produce any decade profiles. Cannot solve.")
            return None

        d = cp.Variable(len(train_decades))
        objective = cp.Minimize(cp.sum_squares(F @ d - t))
        constraints = [d >= 0, cp.sum(d) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if d.value is None:
            logger.error("LP solver failed to find a solution.")
            return None

        temporal_distribution = {decade: val for decade, val in zip(train_decades, d.value)}
        
        return {
            'temporal_distribution': temporal_distribution,
            'num_features': len(self.master_feature_list),
            'status': 'Completed'
        }

    def _split_temporal_data(self, all_data: Dict[str, List[str]]) -> Tuple[Dict, Dict]:
        # This method is now simplified as we load all data and then separate test data.
        train_data = {d: all_data[d] for d in all_data if not d.startswith('test')}
        # Test data is handled separately in the main run function
        test_data = {} 
        return train_data, test_data
        
# --- MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(description="Run temporal inference pipeline on a given text file.")
    parser.add_argument("test_file", type=str, help="Path to the text file to be analyzed.")
    args = parser.parse_args()

    logger.info("Loading temporal data...")
    all_data = load_all_temporal_data() 
    if not all_data:
        logger.error("Failed to load data. Exiting.")
        return

    pipeline = FinalStablePipeline(all_data)
    results = pipeline.run_complete_pipeline(args.test_file)

    if results:
        print("\n" + "=" * 60)
        print(f"FINAL RESULTS FOR: {Path(args.test_file).name}")
        print("=" * 60)

        print(f"\nUsed {results['num_features']} combined subtoken and n-gram features.")
        
        temporal_dist = results['temporal_distribution']
        print("\n🏆 INFERRED TEMPORAL DISTRIBUTION:")
        for decade, prob in sorted(temporal_dist.items()):
            if prob > 0.001:
                percentage = prob * 100
                bar = "█" * int(percentage / 2)
                print(f"{decade:>10}: {percentage:6.2f}% | {bar}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results_{Path(args.test_file).stem}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: str(x))
        logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main() 