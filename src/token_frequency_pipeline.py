"""
A more robust pipeline that uses subword token frequency distributions 
to determine the temporal affinity of a text, abandoning the word-sense model.
"""

import logging
from pathlib import Path
import time
import json
import numpy as np
from collections import Counter
from transformers import GPT2TokenizerFast
from scipy.spatial.distance import jensenshannon
from src.config import PROCESSED_DATA_DIR, TRAIN_DECADES, TEST_TEXT_PATH, RESULTS_DIR
from src.data.dataset_manager import TemporalDatasetManager

# --- Logging Configuration ---
LOGS_DIR = Path(__file__).parent.parent / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "token_frequency_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_all_texts_for_decade(decade: str, manager: TemporalDatasetManager) -> str:
    """Loads all texts for a given decade into a single string."""
    logger.info(f"Loading all texts for {decade}...")
    all_text = []
    
    # Load historical data from processed files
    if int(decade[:4]) < 1990:
        decade_path = PROCESSED_DATA_DIR / decade
        if decade_path.exists():
            for text_file in sorted(decade_path.glob("*.txt")):
                with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                    all_text.append(f.read())
            logger.info(f" -> Loaded {len(all_text)} files from data/processed/{decade}")
    # Load modern data using the manager
    else:
        modern_texts = manager.load_modern_web_content(target_decades=[decade])
        for text, source in modern_texts.get(decade, []):
             if isinstance(text, str):
                all_text.append(text)
        logger.info(f" -> Loaded {len(all_text)} texts from TemporalDatasetManager for {decade}")

    return "\n".join(all_text)

def create_linguistic_fingerprint(text: str, tokenizer) -> np.ndarray:
    """
    Creates a frequency distribution of subword tokens for a given text.
    """
    if not text:
        return np.array([])
        
    # Tokenize the entire text
    token_ids = tokenizer(text, truncation=False)['input_ids']
    
    if not token_ids:
        return np.array([])

    # Count token occurrences
    token_counts = Counter(token_ids)
    
    # Create a frequency vector over the entire vocabulary
    vocab_size = tokenizer.vocab_size
    freq_vector = np.zeros(vocab_size, dtype=np.float32)
    for token_id, count in token_counts.items():
        if token_id < vocab_size:
            freq_vector[token_id] = count
    
    # Normalize to a probability distribution
    total_tokens = freq_vector.sum()
    if total_tokens > 0:
        return freq_vector / total_tokens
    else:
        return np.array([])

def main():
    start_time = time.time()
    logger.info("Starting Token Frequency Temporal Analysis Pipeline")
    logger.info("======================================================")

    logger.info("STEP 1: Initializing Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    manager = TemporalDatasetManager()

    logger.info("STEP 2: Creating Linguistic Fingerprints for each decade...")
    decade_fingerprints = {}
    for decade in TRAIN_DECADES:
        decade_text = get_all_texts_for_decade(decade, manager)
        if decade_text:
            fingerprint = create_linguistic_fingerprint(decade_text, tokenizer)
            if fingerprint.size > 0:
                decade_fingerprints[decade] = fingerprint
                logger.info(f"  -> Fingerprint created for {decade} (Vocabulary size: {len(fingerprint)})")
        else:
            logger.warning(f"No text found for decade {decade}. Skipping.")

    logger.info("STEP 3: Creating fingerprint for the test text...")
    if not TEST_TEXT_PATH.is_file():
        logger.error(f"Test file not found at {TEST_TEXT_PATH}. Aborting.")
        return
        
    with open(TEST_TEXT_PATH, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    test_fingerprint = create_linguistic_fingerprint(test_text, tokenizer)
    if test_fingerprint.size == 0:
        logger.error("Could not create a fingerprint for the test text. Aborting.")
        return

    logger.info("STEP 4: Comparing fingerprints using Jensen-Shannon Divergence...")
    affinity_scores = {}
    for decade, decade_fp in decade_fingerprints.items():
        # JSD requires same-length, non-zero probability vectors.
        # Our fingerprinting function already ensures this.
        epsilon = 1e-10
        test_fp_norm = (test_fingerprint + epsilon)
        decade_fp_norm = (decade_fp + epsilon)

        distance = jensenshannon(test_fp_norm, decade_fp_norm)
        # Convert distance to a similarity score
        affinity_scores[decade] = 1 / (1 + distance)

    total_affinity = sum(affinity_scores.values())
    if total_affinity == 0:
        logger.error("Could not calculate final affinity scores.")
        final_distribution = {}
    else:
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
    results_file = RESULTS_DIR / "token_frequency_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_distribution, f, indent=4)
    logger.info(f"Results saved to {results_file}")

    end_time = time.time()
    logger.info(f"Token Frequency pipeline execution finished in {end_time - start_time:.2f} seconds.")
    logger.info("======================================================")

if __name__ == "__main__":
    main()
