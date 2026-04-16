"""
Configuration for the Temporal Analysis Pipeline
"""
from typing import List, Dict
from pathlib import Path

# --- Path Definitions ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# --- Test Data ---
TEST_TEXT_PATH = DATA_DIR / "test" / "modern_abstract.txt"

# --- Directory Creation ---
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
TEST_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Analysis Configuration ---
TARGET_WORDS: List[str] = [
    # Technology & Science
    'telegraph', 'engine', 'electric', 'photograph', 'steam', 'railway', 'wireless', 'atomic', 'motor', 'chemical',
    # Society & Politics
    'government', 'empire', 'colony', 'liberty', 'democracy', 'revolution', 'socialism', 'capital', 'labour', 'union',
    # Abstract Concepts
    'spirit', 'faith', 'reason', 'progress', 'civilization', 'character', 'virtue', 'genius', 'truth', 'beauty',
    # Commerce & Economy
    'trade', 'market', 'industry', 'company', 'wealth', 'stock', 'bank', 'price', 'manufacture', 'commercial',
    # Culture & Communication
    'press', 'journal', 'publication', 'stage', 'theatre', 'novel', 'education', 'public', 'class', 'race'
]
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = 'cpu'
TRAIN_DECADES: List[str] = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s", "1990s", "2000s"]

# --- Word Sense Induction (Clustering) parameters ---
WSI_CLUSTERING_PARAMS: Dict = {
    'min_clusters': 2,
    'max_clusters': 5,
    'random_state': 42,
    'min_samples_per_word': 5
}

# --- Dummy variables for legacy compatibility with dataset_manager.py ---
TIME_PERIODS = {
    "1850s": (1850, 1859), "1860s": (1860, 1869), "1870s": (1870, 1879),
    "1880s": (1880, 1889), "1890s": (1890, 1899), "1900s": (1900, 1909),
    "1910s": (1910, 1919), "1920s": (1920, 1929), "1930s": (1930, 1939),
    "1940s": (1940, 1949), "1950s": (1950, 1959), "1960s": (1960, 1969),
    "1970s": (1970, 1979), "1980s": (1980, 1989), "1990s": (1990, 1999),
    "2000s": (2000, 2009), "2010s": (2010, 2019), "2020s": (2020, 2029)
}

ANALYSIS_CONFIG = {
    'target_words': TARGET_WORDS,
}
