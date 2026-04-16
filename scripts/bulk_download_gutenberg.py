import requests
import csv
import re
from pathlib import Path
import logging
import gzip
from io import StringIO
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DECADES_TO_DOWNLOAD = {
    "1850s": (1850, 1859), "1860s": (1860, 1869), "1870s": (1870, 1879),
    "1880s": (1880, 1889), "1890s": (1890, 1899), "1900s": (1900, 1909),
    "1910s": (1910, 1919), "1920s": (1920, 1929)
}
MAX_BOOKS_PER_DECADE = 50
MAX_BOOKS_PER_AUTHOR_PER_DECADE = 3
GUTENBERG_CSV_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"

PROJECT_ROOT = Path(__file__).parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "processed"

def clean_gutenberg_text(text: str) -> str:
    start_pattern = r"\\*\\*\\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \\*\\*\\*"
    end_pattern = r"\\*\\*\\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \\*\\*\\*"
    
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    end_match = re.search(end_pattern, text, re.IGNORECASE)
    
    if start_match and end_match:
        start_pos = start_match.end()
        end_pos = end_match.start()
        return text[start_pos:end_pos].strip()
    
    return ""

def find_publication_year(text_header: str) -> int | None:
    """Heuristically find the original publication year from the text header."""
    # Look for a 4-digit number in the target range.
    # This is not perfect but the best we can do without reliable metadata.
    potential_years = re.findall(r'\b(18[5-9]\d|19[0-2]\d)\b', text_header)
    if potential_years:
        # Often the first year mentioned is the original publication.
        return int(potential_years[0])
    return None

def download_and_process_book(book: dict, books_by_decade: defaultdict, author_counts_by_decade: defaultdict) -> bool:
    book_id = book.get('Text#')
    if not book_id:
        return False

    # A list of common URL patterns for Gutenberg .txt files
    urls_to_try = [
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    ]

    text = None
    logger.info(f"Processing book ID: {book_id}")
    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                text = response.text
                logger.info(f"  -> Success downloading from {url}")
                break # Success
        except requests.exceptions.RequestException:
            logger.debug(f"  -> URL failed: {url}. Trying next.")
            continue
    
    if not text:
        logger.warning(f"  -> FAILED to download book ID {book_id} from all URLs.")
        return False

    try:
        # 1. Find publication year from header
        header = text[:3000] # Check first 3000 chars for metadata
        year = find_publication_year(header)
        if not year:
            logger.info(f"  -> No valid year found for book ID {book_id}.")
            return False

        # 2. Determine decade and apply filters
        book_decade = None
        for decade_name, (start_year, end_year) in DECADES_TO_DOWNLOAD.items():
            if start_year <= year <= end_year:
                book_decade = decade_name
                break
        
        if not book_decade:
            logger.info(f"  -> Year {year} for book {book_id} not in a target decade.")
            return False

        if len(books_by_decade[book_decade]) >= MAX_BOOKS_PER_DECADE:
            logger.info(f"  -> Decade {book_decade} is full. Skipping.")
            return False

        authors = book.get('Authors', 'Unknown')
        author_key = authors.split(';')[0].strip()
        if author_counts_by_decade[book_decade][author_key] >= MAX_BOOKS_PER_AUTHOR_PER_DECADE:
            logger.info(f"  -> Author {author_key} limit reached for {book_decade}. Skipping.")
            return False

        # 3. Clean and save text
        logger.info(f"  -> SAVING book {book_id} to decade {book_decade}.")
        cleaned_text = clean_gutenberg_text(text)
        if cleaned_text and len(cleaned_text) > 1000:
            decade_path = TARGET_DIR / book_decade
            decade_path.mkdir(exist_ok=True)
            
            title = book.get('Title', f'unknown_title_{book_id}').replace(' ', '_').lower()
            clean_title = re.sub(r'[^a-z0-9_]', '', title)[:50]
            filename = f"{book_id}_{clean_title}.txt"
            
            if any(f.name.startswith(f"{book_id}_") for f in decade_path.iterdir()):
                logger.info(f"  -> Already exists. Skipping.")
                return False

            target_file = decade_path / filename
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            
            logger.info(f"  -> Successfully saved {filename} to {book_decade} (Year: {year})")
            books_by_decade[book_decade].append(book)
            author_counts_by_decade[book_decade][author_key] += 1
            return True
        else:
            logger.warning(f"  -> No valid content found for book ID {book_id}")
            return False
            
    except Exception as e:
        logger.error(f"  -> An unexpected error occurred while processing book {book_id}: {e}")
        return False

def main():
    logger.info("Starting large-scale Gutenberg data download process with publication year extraction...")
    
    try:
        response = requests.get(GUTENBERG_CSV_URL, timeout=60)
        response.raise_for_status()
        
        gzip_file = gzip.decompress(response.content)
        csv_text = gzip_file.decode('utf-8')
        all_books = list(csv.DictReader(StringIO(csv_text)))
        
        logger.info(f"Successfully downloaded and parsed catalog with {len(all_books)} entries.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download Gutenberg catalog: {e}")
        return
    except Exception as e:
        logger.error(f"Failed to parse Gutenberg catalog: {e}")
        return

    # --- Filter and Download Books ---
    books_by_decade = defaultdict(list)
    author_counts_by_decade = defaultdict(lambda: defaultdict(int))
    total_downloaded = 0

    # Filter for English text books first
    english_texts = [b for b in all_books if b.get('Type') == 'Text' and b.get('Language') == 'en']
    logger.info(f"Found {len(english_texts)} English text books to process.")

    for book in english_texts:
        # Check if all decades are full
        if all(len(books_by_decade[d]) >= MAX_BOOKS_PER_DECADE for d in DECADES_TO_DOWNLOAD):
            logger.info("All target decades are full. Stopping download process.")
            break
        
        if download_and_process_book(book, books_by_decade, author_counts_by_decade):
            total_downloaded += 1
        
        time.sleep(1) # Be polite to Gutenberg servers
    
    logger.info(f"Bulk data download process complete. Downloaded {total_downloaded} new books.")

if __name__ == "__main__":
    main() 