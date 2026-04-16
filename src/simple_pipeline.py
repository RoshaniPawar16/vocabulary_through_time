"""
A more robust, file-based pipeline for temporal analysis using HDBSCAN.
"""

import logging
from pathlib import Path
import time
import json
import re
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
import torch
from sklearn.linear_model import LinearRegression
import random

from src.data.dataset_manager import TemporalDatasetManager
from src.config import (
    PROCESSED_DATA_DIR,
    TARGET_WORDS,
    TRAIN_DECADES,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    WSI_CLUSTERING_PARAMS, # We'll borrow random_state and min_samples
    TEST_TEXT_PATH,
    RESULTS_DIR
)

# --- Logging Configuration ---
LOGS_DIR = Path(__file__).parent.parent / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "robust_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_text_from_file(filepath: Path) -> str:
    """Loads text content from a file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def find_word_usages(text: str, words: list) -> dict:
    """Finds all sentences containing the target words."""
    usages = defaultdict(list)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        for word in words:
            if re.search(r'\b' + re.escape(word) + r'\b', sentence, re.IGNORECASE):
                usages[word].append(sentence)
    return usages

def get_historical_usages():
    """Loads usages of target words from all historical decades."""
    historical_usages = defaultdict(lambda: defaultdict(list))
    logger.info("Loading historical texts from data/processed...")
    
    # Filter for decades before 1990 for this specific loader
    historical_decades = [d for d in TRAIN_DECADES if int(d[:4]) < 1990]

    for decade in historical_decades:
        decade_path = PROCESSED_DATA_DIR / decade
        if not decade_path.exists():
            logger.warning(f"Directory for decade {decade} not found. Skipping.")
            continue
        
        logger.info(f" -> Loading texts from {decade}...")
        file_count = 0
        for text_file in sorted(decade_path.glob("*.txt")):
            text = load_text_from_file(text_file)
            usages = find_word_usages(text, TARGET_WORDS)
            for word, sentences in usages.items():
                historical_usages[word][decade].extend(sentences)
            file_count += 1
        logger.info(f"    - Found usages in {file_count} files.")
    return historical_usages

def get_modern_usages():
    """Loads usages of target words from modern decades using the dataset manager."""
    modern_usages = defaultdict(lambda: defaultdict(list))
    logger.info("Loading modern texts using TemporalDatasetManager...")
    manager = TemporalDatasetManager()
    
    modern_decades_to_load = [d for d in TRAIN_DECADES if int(d[:4]) >= 1990]
    if not modern_decades_to_load:
        return modern_usages

    modern_texts = manager.load_modern_web_content(target_decades=modern_decades_to_load)

    for decade, texts_with_sources in modern_texts.items():
        logger.info(f" -> Processing {len(texts_with_sources)} texts for {decade}...")
        for text, source in texts_with_sources:
            if not isinstance(text, str): continue
            usages = find_word_usages(text, TARGET_WORDS)
            for word, sentences in usages.items():
                modern_usages[word][decade].extend(sentences)
    return modern_usages

def get_word_in_context_embeddings(word: str, sentences: list, model: SentenceTransformer, device: str):
    """
    Generates contextual embeddings for a specific word in a list of sentences.
    This is more precise than embedding the entire sentence.
    """
    model.to(device)
    tokenizer = model.tokenizer
    bert_model = model[0].auto_model

    embeddings = []
    for sentence in sentences:
        # Tokenize sentence and find the target word's tokens
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)
        token_ids = inputs['input_ids'][0]
        
        # This is a simplified approach to find word tokens.
        # A more robust solution would handle word pieces and complex tokenization.
        word_tokens_str = tokenizer.tokenize(word)
        word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens_str)
        
        # Find the sequence of word_token_ids in the sentence's token_ids
        found_indices = []
        for i in range(len(token_ids) - len(word_token_ids) + 1):
            if torch.equal(token_ids[i:i+len(word_token_ids)], torch.tensor(word_token_ids, device=device)):
                found_indices.extend(range(i, i + len(word_token_ids)))
                break # Take the first match
        
        if not found_indices:
            continue

        # Get hidden states
        with torch.no_grad():
            outputs = bert_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0] # Last hidden layer

        # Average the embeddings of the word's tokens
        word_embedding = hidden_states[found_indices].mean(dim=0)
        embeddings.append(word_embedding.cpu().numpy())

    if not embeddings:
        return np.array([])
        
    return np.vstack(embeddings)

def discover_word_senses_robust(word: str, sentences: list, model: SentenceTransformer):
    """Discovers word senses using the more robust HDBSCAN algorithm."""
    logger.info(f"Discovering senses for word: '{word}' ({len(sentences)} usages) using HDBSCAN")
    
    embeddings = get_word_in_context_embeddings(word, sentences, model, EMBEDDING_DEVICE)
    
    if embeddings.shape[0] < WSI_CLUSTERING_PARAMS['min_samples_per_word']:
        logger.warning(f"  - Not enough valid embeddings found for '{word}' to perform clustering.")
        return None

    # Option 1 Fix: Make clustering less strict
    min_cluster_size = 2
    logger.info(f"  - Using a less strict min_cluster_size: {min_cluster_size}")
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size, metric='euclidean', gen_min_span_tree=True, prediction_data=True)
    clusterer.fit(embeddings)
    
    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = np.sum(labels == -1)
    
    logger.info(f"  - Found {n_clusters} senses and {noise_points} noise points.")

    # Calculate silhouette score only for non-noise points if there are clusters
    if n_clusters > 1:
        clustered_indices = labels != -1
        if np.sum(clustered_indices) > 1:
            try:
                core_embeddings = embeddings[clustered_indices]
                core_labels = labels[clustered_indices]
                silhouette = silhouette_score(core_embeddings, core_labels)
                logger.info(f"  - Silhouette Score (on core points): {silhouette:.4f}")
            except ValueError as e:
                logger.warning(f"  - Could not compute silhouette score for '{word}': {e}")
        else:
            logger.warning("  - Not enough clustered points to compute silhouette score.")
    
    return clusterer

def get_sense_distributions_robust(usages: dict, sense_models: dict, embedder: SentenceTransformer):
    """Calculates the distribution of senses for each decade, ignoring noise."""
    distributions = defaultdict(lambda: defaultdict(dict))
    
    for word, decade_usages in usages.items():
        if word not in sense_models: continue
        clusterer = sense_models[word]
        
        for decade, sentences in decade_usages.items():
            if not sentences:
                distributions[word][decade] = np.array([])
                continue
            
            embeddings = get_word_in_context_embeddings(word, sentences, embedder, EMBEDDING_DEVICE)

            if embeddings.shape[0] == 0:
                distributions[word][decade] = np.array([])
                continue

            labels, _ = hdbscan.approximate_predict(clusterer, embeddings)
            
            # Filter out noise points (label -1)
            core_labels = labels[labels != -1]
            n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
            
            if len(core_labels) > 0:
                counts = np.bincount(core_labels, minlength=n_clusters)
                total = np.sum(counts)
                distributions[word][decade] = counts / total if total > 0 else np.zeros(n_clusters)
            else:
                distributions[word][decade] = np.zeros(n_clusters)

    return distributions

def analyze_test_text_robust(sense_models: dict, embedder: SentenceTransformer):
    """Analyzes the modern test text and calculates its sense distribution."""
    logger.info(f"Analyzing modern test text: {TEST_TEXT_PATH}")
    text = load_text_from_file(TEST_TEXT_PATH)
    usages = find_word_usages(text, TARGET_WORDS)
    
    test_dist = {}
    
    for word, sentences in usages.items():
        if not sentences or word not in sense_models:
            continue
            
        clusterer = sense_models[word]
        embeddings = get_word_in_context_embeddings(word, sentences, embedder, EMBEDDING_DEVICE)
        
        if embeddings.shape[0] == 0:
            continue

        labels, _ = hdbscan.approximate_predict(clusterer, embeddings)
        
        core_labels = labels[labels != -1]
        n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)

        if len(core_labels) > 0:
            counts = np.bincount(core_labels, minlength=n_clusters)
            total = np.sum(counts)
            test_dist[word] = counts / total if total > 0 else np.zeros(n_clusters)
        else:
            test_dist[word] = np.zeros(n_clusters)
            
    return test_dist
    
def infer_temporal_affinity_jsd(test_dist: dict, historical_dists: dict):
    """
    Infers the temporal affinity of the test text using Jensen-Shannon Divergence.
    This directly compares the test distribution to each decade's distribution.
    """
    affinity_scores = defaultdict(float)
    
    # Get a union of all decades present in the data
    all_decades = set()
    for word_dists in historical_dists.values():
        all_decades.update(word_dists.keys())
    
    if not all_decades:
        return {}

    for decade_str in sorted(list(all_decades)):
        total_distance = 0.0
        words_in_common = 0

        for word, t_dist in test_dist.items():
            if word in historical_dists and decade_str in historical_dists[word]:
                h_dist = historical_dists[word][decade_str]
                
                # Ensure distributions are valid and have the same length
                if h_dist.size == 0 or t_dist.size == 0 or len(h_dist) != len(t_dist):
                    continue
                
                # JSD requires non-zero probabilities. Add a tiny epsilon.
                epsilon = 1e-10
                t_dist_norm = (t_dist + epsilon) / (t_dist.sum() + epsilon * len(t_dist))
                h_dist_norm = (h_dist + epsilon) / (h_dist.sum() + epsilon * len(h_dist))

                distance = jensenshannon(t_dist_norm, h_dist_norm)
                total_distance += distance
                words_in_common += 1

        if words_in_common > 0:
            avg_distance = total_distance / words_in_common
            # Convert distance to similarity score (lower distance = higher similarity)
            affinity_scores[decade_str] = 1 / (1 + avg_distance)
        else:
            affinity_scores[decade_str] = 0

    total_affinity = sum(affinity_scores.values())
    if total_affinity == 0: return {}
    final_distribution = {decade: (score / total_affinity) * 100 for decade, score in affinity_scores.items()}
    
    return final_distribution

def main():
    start_time = time.time()
    logger.info("Starting ROBUST Temporal Analysis Pipeline - Less Strict Clustering (min_cluster_size=2)")
    logger.info("======================================================")

    logger.info("STEP 1: Loading all text data...")
    historical_usages = get_historical_usages()
    modern_usages = get_modern_usages()
    
    # Merge dictionaries
    for word, decade_usages in modern_usages.items():
        for decade, sentences in decade_usages.items():
            historical_usages[word][decade].extend(sentences)

    if not historical_usages:
        logger.error("No data found. Aborting.")
        return

    logger.info(f"STEP 2: Loading embedding model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("STEP 3: Discovering word senses from all historical data...")
    sense_models = {}
    all_usages_for_senses = defaultdict(list)
    for word in TARGET_WORDS:
        for decade in TRAIN_DECADES:
            if word in historical_usages and decade in historical_usages[word]:
                 all_usages_for_senses[word].extend(historical_usages[word][decade])

    for word, sentences in all_usages_for_senses.items():
        # *** NEW STRATEGY: Limit sentences to 100 for this experiment ***
        if len(sentences) > 100:
            logger.info(f"Sampling 100 sentences for '{word}' from {len(sentences)} total usages.")
            sentences_to_process = random.sample(sentences, 100)
        else:
            sentences_to_process = sentences

        if len(sentences_to_process) < WSI_CLUSTERING_PARAMS['min_samples_per_word']:
            logger.warning(f"Skipping word '{word}' due to insufficient samples ({len(sentences_to_process)} found).")
            continue

        sense_model = discover_word_senses_robust(word, sentences_to_process, embedder)
        if sense_model:
            sense_models[word] = sense_model

    logger.info("STEP 4: Calculating historical sense distributions...")
    distributions = get_sense_distributions_robust(historical_usages, sense_models, embedder)
    
    logger.info("STEP 5: Analyzing test text...")
    test_distribution = analyze_test_text_robust(sense_models, embedder)
    if not test_distribution:
        logger.error("Could not find any target words in the test text. Aborting.")
        return
    
    logger.info("STEP 6: Inferring temporal affinity using JSD...")
    final_distribution = infer_temporal_affinity_jsd(test_distribution, distributions)

    logger.info("==========================================")
    logger.info("           FINAL ANALYSIS RESULTS         ")
    logger.info("==========================================")
    
    if not final_distribution:
        logger.warning("Could not calculate final affinity scores.")
    else:
        logger.info(f"Temporal affinity of '{TEST_TEXT_PATH.name}':")
        for decade, score in sorted(final_distribution.items(), key=lambda item: item[1], reverse=True):
            logger.info(f"  - {decade}: {score:.2f}%")
        
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / "less_strict_clustering_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_distribution, f, indent=4)
    logger.info(f"Results saved to {results_file}")

    end_time = time.time()
    logger.info(f"Robust pipeline execution finished in {end_time - start_time:.2f} seconds.")
    logger.info("======================================================")


if __name__ == "__main__":
    if not TEST_TEXT_PATH.is_file():
        logger.warning(f"Test file not found at {TEST_TEXT_PATH}. Creating a dummy file.")
        TEST_TEXT_PATH.parent.mkdir(exist_ok=True, parents=True)
        with open(TEST_TEXT_PATH, 'w') as f:
            f.write("This is a modern test paper about a machine learning model and its network weights.")
    main()
