"""
Gutenberg Data Downloader

This script downloads a curated list of plain-text books from Project Gutenberg
to populate our dataset for the 1880s and 1920s.
"""

import requests
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Books selected for their publication date and influence.
BOOKS_TO_DOWNLOAD = {
    "1850s": {
        "2701": "moby_dick",         # Moby Dick (1851)
        "205": "walden",             # Walden (1854)
        "98": "tale_of_two_cities"   # A Tale of Two Cities (1859)
    },
    "1860s": {
        "1400": "great_expectations", # Great Expectations (1861)
        "11": "alices_adventures",    # Alice's Adventures in Wonderland (1865)
        "514": "little_women"        # Little Women (1868)
    },
    "1870s": {
        "74": "tom_sawyer",          # The Adventures of Tom Sawyer (1876)
        "105": "around_the_world",    # Around the World in Eighty Days (1873)
        "28054": "through_looking_glass" # Through the Looking-Glass (1871)
    },
    "1880s": {
        "76": "huck_finn",
        "244": "study_in_scarlet",
        "43": "jekyll_and_hyde",
        "120": "treasure_island",
    },
    "1890s": {
        "2591": "dorian_gray",       # The Picture of Dorian Gray (1890)
        "345": "dracula",            # Dracula (1897)
        "35": "time_machine"         # The Time Machine (1895)
    },
    "1900s": {
        "215": "call_of_the_wild",  # The Call of the Wild (1903)
        "2852": "wind_in_willows",   # The Wind in the Willows (1908)
        "45": "anne_of_green_gables" # Anne of Green Gables (1908)
    },
    "1910s": {
        "64": "a_princess_of_mars",  # A Princess of Mars (1917)
        "219": "heart_of_darkness", # Heart of Darkness (pre-1923, often cited 1899/1902 but fits here)
        "16": "peter_pan"           # Peter Pan (1911)
    },
    "1920s": {
        "67138": "sun_also_rises",
        "65203": "roger_ackroyd",
        "4300": "ulysses",
        "1156": "babbitt",
    }
}

PROJECT_ROOT = Path(__file__).parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "processed"

def clean_gutenberg_text(text: str) -> str:
    """
    Removes the Project Gutenberg header and footer.
    """
    start_pattern = r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*"
    end_pattern = r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*"
    
    # Find start and end markers
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    end_match = re.search(end_pattern, text, re.IGNORECASE)
    
    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]
        
    return text.strip()

def download_book(book_id: str, filename: str, decade_path: Path):
    """
    Downloads and saves a single book.
    """
    target_file = decade_path / f"{filename}.txt"
    if target_file.exists():
        logger.info(f"'{filename}.txt' already exists in {decade_path.name}. Skipping.")
        return

    # URL format for plain text UTF-8 files
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # The content is bytes, so we need to decode it
        raw_text = response.content.decode('utf-8')
        cleaned_text = clean_gutenberg_text(raw_text)
        
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logger.info(f"Successfully downloaded and saved '{filename}.txt'")

    except requests.exceptions.HTTPError as e:
        logger.warning(f"Could not download book {book_id} from {url}. HTTP Error: {e.response.status_code}. Trying alternative URL.")
        # Fallback for older directory structures
        url_alt = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        try:
            response = requests.get(url_alt)
            response.raise_for_status()
            raw_text = response.content.decode('utf-8')
            cleaned_text = clean_gutenberg_text(raw_text)
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            logger.info(f"Successfully downloaded and saved '{filename}.txt' from alternate URL.")
        except requests.exceptions.RequestException as e_alt:
            logger.error(f"Failed to download book {book_id} from both primary and alternate URLs. Error: {e_alt}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download book {book_id}. Error: {e}")


def main():
    logger.info("Starting Gutenberg data download process...")
    
    for decade, books in BOOKS_TO_DOWNLOAD.items():
        decade_path = TARGET_DIR / decade
        decade_path.mkdir(exist_ok=True)
        logger.info(f"--- Processing decade: {decade} ---")
        
        for book_id, filename in books.items():
            download_book(book_id, filename, decade_path)
            
    logger.info("Data download process complete.")


if __name__ == "__main__":
    main() 