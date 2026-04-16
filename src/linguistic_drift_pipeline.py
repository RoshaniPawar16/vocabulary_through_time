"""
A robust temporal analysis pipeline based on linguistic drift, measured by
the frequency distribution of subword tokens across decades.
"""

import logging
from pathlib import Path
import time
import json
import numpy as np
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from scipy.spatial.distance import jensenshannon

from src.data.dataset_manager import TemporalDatasetManager
from src.config import (
    PROCESSED_DATA_DIR,
    TRAIN_DECADES,
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
        logging.FileHandler(LOGS_DIR / "linguistic_drift_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_all_texts():
    """
    Loads all texts from historical and modern sources.
    Returns a dictionary mapping decades to lists of text content.
    """
    all_texts = defaultdict(list)
    logger.info("Loading all available texts...")

    # 1. Load historical texts from data/processed
    logger.info(" -> Loading historical texts from data/processed...")
    historical_decades = [d for d in TRAIN_DECADES if int(d[:4]) < 1990]
    for decade in historical_decades:
        decade_path = PROCESSED_DATA_DIR / decade
        if not decade_path.exists():
            continue
        for text_file in sorted(decade_path.glob("*.txt")):
            with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_texts[decade].append(f.read())
        logger.info(f"    - Loaded {len(all_texts[decade])} texts for {decade}.")

    # 2. Load modern texts using the dataset manager
    logger.info(" -> Loading modern texts using TemporalDatasetManager...")
    manager = TemporalDatasetManager()
    modern_decades_to_load = [d for d in TRAIN_DECADES if int(d[:4]) >= 1990]
    if modern_decades_to_load:
        modern_data = manager.load_modern_web_content(target_decades=modern_decades_to_load)
        for decade, texts_with_sources in modern_data.items():
            for text, source in texts_with_sources:
                if isinstance(text, str):
                    all_texts[decade].append(text)
            logger.info(f"    - Loaded {len(modern_data.get(decade, []))} texts for {decade}.")
            
    return all_texts

def create_token_fingerprint(texts: list, tokenizer, batch_size=100) -> Counter:
    """
    Creates a frequency distribution of subword tokens for a list of texts.
    """
    if not texts:
        return Counter()

    token_counts = Counter()
    logger.info(f"   -> Tokenizing {len(texts)} texts...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # The tokenizer returns a list of lists of token IDs
        tokenized_batch = tokenizer(batch, truncation=True, max_length=512)['input_ids']
        for token_list in tokenized_batch:
            token_counts.update(token_list)
    return token_counts

def main():
    start_time = time.time()
    logger.info("Starting Linguistic Drift Pipeline")
    logger.info("=====================================")

    logger.info("STEP 1: Loading all texts for all decades...")
    all_texts_by_decade = load_all_texts()
    if not all_texts_by_decade:
        logger.error("No data found. Aborting.")
        return

    logger.info("STEP 2: Initializing tokenizer (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    logger.info("STEP 3: Creating linguistic fingerprints for each decade...")
    fingerprints = {}
    for decade, texts in all_texts_by_decade.items():
        logger.info(f" -> Processing decade: {decade}")
        if texts:
            fingerprints[decade] = create_token_fingerprint(texts, tokenizer)

    logger.info("STEP 4: Creating linguistic fingerprint for the test text...")
    with open(TEST_TEXT_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        test_text = f.read()
    test_fingerprint = create_token_fingerprint([test_text], tokenizer)

    logger.info("STEP 5: Comparing fingerprints using Jensen-Shannon Divergence...")
    
    # Create a global vocabulary of all tokens encountered
    global_vocab = set(test_fingerprint.keys())
    for fp in fingerprints.values():
        global_vocab.update(fp.keys())
    
    sorted_vocab = sorted(list(global_vocab))
    vocab_map = {token: i for i, token in enumerate(sorted_vocab)}
    vocab_size = len(sorted_vocab)
    
    # Convert test fingerprint to a probability vector
    test_vector = np.zeros(vocab_size)
    for token, count in test_fingerprint.items():
        test_vector[vocab_map[token]] = count
    test_vector /= test_vector.sum()

    # Calculate JSD for each decade
    affinity_scores = {}
    for decade, fp in fingerprints.items():
        decade_vector = np.zeros(vocab_size)
        for token, count in fp.items():
            decade_vector[vocab_map[token]] = count
        decade_vector /= decade_vector.sum()
        
        # JSD requires non-zero probabilities. Add a tiny epsilon.
        epsilon = 1e-10
        test_vector_norm = test_vector + epsilon
        decade_vector_norm = decade_vector + epsilon

        distance = jensenshannon(test_vector_norm, decade_vector_norm)
        affinity_scores[decade] = 1 / (1 + distance) # Lower distance = higher similarity
    
    total_affinity = sum(affinity_scores.values())
    final_distribution = {decade: (score / total_affinity) * 100 for decade, score in affinity_scores.items()}

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
    
    # Save final percentage results
    results_file = RESULTS_DIR / "linguistic_drift_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_distribution, f, indent=4)
    logger.info(f"Final affinity scores saved to {results_file}")

    # Save the raw fingerprints for inspection
    fingerprint_data = {
        "vocabulary": {v: k for k, v in tokenizer.get_vocab().items()},
        "test_fingerprint": test_fingerprint,
        "decade_fingerprints": fingerprints
    }
    fingerprints_file = RESULTS_DIR / "linguistic_fingerprints.json"
    with open(fingerprints_file, 'w') as f:
        json.dump(fingerprint_data, f, indent=4)
    logger.info(f"Raw linguistic fingerprints saved to {fingerprints_file}")

    end_time = time.time()
    logger.info(f"Linguistic drift pipeline execution finished in {end_time - start_time:.2f} seconds.")
    logger.info("======================================================")

if __name__ == "__main__":
    if not TEST_TEXT_PATH.is_file():
        logger.warning(f"Test file not found at {TEST_TEXT_PATH}. Creating a dummy file.")
        TEST_TEXT_PATH.parent.mkdir(exist_ok=True, parents=True)
        with open(TEST_TEXT_PATH, 'w') as f:
            f.write("This is a modern test paper about a machine learning model and its network weights.")
    main()
