#!/usr/bin/env python3
"""
Automated Word Discovery for Temporal Analysis

This module finds words that are suitable for semantic shift analysis by
identifying words that are common in both historical and modern corpora.
"""

import nltk
from collections import Counter
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    logger.info("Downloading necessary NLTK data (punkt, averaged_perceptron_tagger)...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    logger.info("NLTK data downloaded.")

class WordDiscoverer:
    """
    Discovers candidate words for semantic shift analysis by comparing
    a historical and a modern text corpus.
    """
    def __init__(self, all_temporal_data: Dict[str, List[str]], 
                 historical_end_year: int = 1950, 
                 modern_start_year: int = 1980):
        self.all_data = all_temporal_data
        self.historical_end_year = historical_end_year
        self.modern_start_year = modern_start_year
        
        self.historical_corpus, self.modern_corpus = self._split_corpus()

    def _split_corpus(self) -> Tuple[str, str]:
        """Splits the full dataset into a historical and modern corpus."""
        logger.info(f"Splitting corpus into historical (pre-{self.historical_end_year}) and modern (post-{self.modern_start_year})...")
        
        historical_texts = []
        modern_texts = []

        for decade, texts in self.all_data.items():
            try:
                decade_start_year = int(decade.replace('s', ''))
                if decade_start_year < self.historical_end_year:
                    historical_texts.extend(texts)
                elif decade_start_year >= self.modern_start_year:
                    modern_texts.extend(texts)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse decade: {decade}. Skipping.")
                continue

        logger.info(f"Historical corpus: {len(historical_texts)} texts. Modern corpus: {len(modern_texts)} texts.")
        return " ".join(historical_texts), " ".join(modern_texts)

    def _get_frequent_words(self, corpus: str, top_n: int = 2000) -> Dict[str, int]:
        """Tokenizes text, performs POS tagging, and returns frequent nouns/verbs."""
        tokens = nltk.word_tokenize(corpus.lower())
        
        # Keep only alphabetic tokens of reasonable length
        words = [word for word in tokens if word.isalpha() and len(word) > 3]
        
        # Part-of-speech tagging
        tagged_words = nltk.pos_tag(words)
        
        # Filter for nouns (NN, NNS) and verbs (VB, VBD, VBG, VBN, VBP, VBZ)
        nouns_and_verbs = [
            word for word, tag in tagged_words 
            if tag.startswith('NN') or tag.startswith('VB')
        ]
        
        # Return the most common words
        return dict(Counter(nouns_and_verbs).most_common(top_n))

    def find_candidate_words(self, top_n: int = 2000, min_freq: int = 50) -> List[str]:
        """
        Finds words that are frequent in both historical and modern corpora.

        Args:
            top_n: The number of most frequent words to consider from each corpus.
            min_freq: The minimum frequency for a word to be considered in either corpus.

        Returns:
            A list of candidate words suitable for semantic shift analysis.
        """
        logger.info("Finding frequent words in historical corpus...")
        historical_freqs = self._get_frequent_words(self.historical_corpus, top_n)
        
        logger.info("Finding frequent words in modern corpus...")
        modern_freqs = self._get_frequent_words(self.modern_corpus, top_n)

        # Find the intersection of words that are frequent in both
        historical_set = set(word for word, freq in historical_freqs.items() if freq >= min_freq)
        modern_set = set(word for word, freq in modern_freqs.items() if freq >= min_freq)
        
        common_words = list(historical_set.intersection(modern_set))
        
        logger.info(f"Found {len(common_words)} common, frequent words to be used as candidates.")
        
        # Optional: Add some high-confidence manual words just in case
        manual_words = ['power', 'cell', 'engine', 'broadcast', 'line', 'press', 'call']
        final_candidates = sorted(list(set(common_words + manual_words)))
        
        logger.info(f"Total candidates after adding manual list: {len(final_candidates)}")

        return final_candidates

if __name__ == '__main__':
    # This is a demo showing how to use the WordDiscoverer
    # It requires a data loading function, which we will have in the main pipeline.
    # from some_data_loader import load_all_data
    
    # print("Demonstration of WordDiscoverer")
    # all_data = load_all_data() # Fictional data loader
    # discoverer = WordDiscoverer(all_data)
    # candidate_words = discoverer.find_candidate_words()
    # print("\nFound candidate words:")
    # print(candidate_words)
    pass 