"""
Word Sense Induction and Disambiguation Module

This module is responsible for taking a corpus of text and a target word,
and discovering the different "senses" (meanings) of that word based on
the contexts in which it appears.

It works by:
1. Finding all sentences containing the target word.
2. Embedding these sentences using a sentence-transformer model.
3. Clustering the embeddings to group similar contexts. Each cluster
   represents a distinct word sense.
4. Using silhouette scores to find the optimal number of word senses.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE, WSI_CLUSTERING_PARAMS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WordSenseDiscoverer:
    """
    Discovers word senses by clustering sentence embeddings.
    """

    def __init__(self, all_temporal_data: Dict[str, List[str]]):
        self.all_data = all_temporal_data
        logger.info(f"Loading sentence embedding model: {EMBEDDING_MODEL}")
        # Load model explicitly to CPU to avoid Metal/MPS issues
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        self.word_sense_inventory: Dict[str, Dict] = {}

    def discover_senses_for_word(self, word: str) -> bool:
        """
        Extracts contexts, embeds them, and clusters them to find senses.

        Args:
            word (str): The word to analyze.

        Returns:
            bool: True if senses were successfully discovered, False otherwise.
        """
        logger.info(f"Discovering senses for word: '{word}'")

        # 1. Extract contexts (sentences)
        contexts = self._extract_contexts(word)
        if len(contexts) < WSI_CLUSTERING_PARAMS['min_samples_per_word']:
            logger.warning(f"Insufficient contexts for '{word}' (found {len(contexts)}). Skipping.")
            return False

        # 2. Embed contexts
        logger.info(f"  - Embedding {len(contexts)} contexts...")
        context_vectors = self.model.encode([ctx['text'] for ctx in contexts], show_progress_bar=True)

        # 3. Find optimal number of clusters (k)
        best_k, best_score = self._find_optimal_k(context_vectors, word)
        if best_k is None:
            return False
        
        logger.info(f"  - Optimal k for '{word}' is {best_k} with silhouette score: {best_score:.4f}")

        # 4. Perform final clustering and store results
        kmeans = KMeans(n_clusters=best_k, random_state=WSI_CLUSTERING_PARAMS['random_state'], n_init=10)
        labels = kmeans.fit_predict(context_vectors)

        sense_data = {
            'contexts': [],
            'sense_labels': labels.tolist(),
            'num_senses': best_k,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
        for i, context in enumerate(contexts):
            sense_data['contexts'].append({
                'text': context['text'],
                'decade': context['decade'],
                'sense_id': labels[i]
            })

        self.word_sense_inventory[word] = sense_data
        return True

    def _extract_contexts(self, word: str) -> List[Dict]:
        """
        Finds all sentences in the corpus containing the given word.
        """
        contexts = []
        # Use regex to find the whole word, ignoring case
        # \b ensures we match whole words only (e.g., 'power' not 'powerful')
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)

        for decade, texts in self.all_data.items():
            full_text = " ".join(texts)
            # Simple sentence splitting on periods. Good enough for this data.
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            for sentence in sentences:
                if pattern.search(sentence):
                    contexts.append({'text': sentence.strip(), 'decade': decade})
        return contexts

    def _find_optimal_k(self, vectors: np.ndarray, word: str) -> Tuple[int, float]:
        """
        Calculates silhouette scores for different k to find the best number of clusters.
        """
        min_k = WSI_CLUSTERING_PARAMS['min_clusters']
        max_k = WSI_CLUSTERING_PARAMS['max_clusters']
        random_state = WSI_CLUSTERING_PARAMS['random_state']

        best_k = None
        best_score = -1

        logger.info(f"  - Searching for optimal k in range [{min_k}, {max_k-1}] for '{word}'...")
        
        for k in range(min_k, max_k):
            if len(vectors) < k:
                logger.warning(f"  - Cannot test k={k}, not enough samples ({len(vectors)}).")
                break
            
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(vectors)
            
            # Silhouette score requires at least 2 labels.
            if len(set(labels)) > 1:
                score = silhouette_score(vectors, labels)
                logger.info(f"    - k={k}, Silhouette Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
            else:
                logger.warning(f"  - k={k} resulted in a single cluster. Cannot compute silhouette score.")

        if best_k is None:
             logger.warning(f"Could not determine optimal k for '{word}'. Assigning to a single sense.")
             return 1, 0.0 # Default to a single sense
             
        return best_k, best_score

    def get_sense_distribution_for_decade(self, word: str, decade: str) -> np.ndarray:
        """
        Calculates the frequency distribution of senses for a word in a specific decade.
        """
        if word not in self.word_sense_inventory:
            raise ValueError(f"Senses for '{word}' have not been discovered yet.")

        inventory = self.word_sense_inventory[word]
        num_senses = inventory['num_senses']
        
        sense_counts = defaultdict(int)
        total_count = 0

        for context in inventory['contexts']:
            if context['decade'] == decade:
                sense_counts[context['sense_id']] += 1
                total_count += 1
        
        if total_count == 0:
            return np.zeros(num_senses)

        distribution = np.array([sense_counts[i] for i in range(num_senses)])
        return distribution / total_count 