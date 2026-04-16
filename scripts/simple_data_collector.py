#!/usr/bin/env python3
"""
SIMPLE DATA COLLECTOR - Real Historical Texts Only
No synthetic data, no complex caching, just direct downloads from real sources.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List
import requests
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target decades and counts
DECADES = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s", 
          "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]

TEXTS_PER_DECADE = 50  # Keep it simple - 50 real texts per decade

# Output directory
OUTPUT_DIR = Path("../data/real_historical_texts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def collect_gutenberg_data():
    """Collect from Project Gutenberg (1850s-1920s) - Already done"""
    logger.info("✅ Project Gutenberg data already collected (406 texts)")
    
    # Count existing Gutenberg files
    gutenberg_count = 0
    for decade in ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s"]:
        decade_dir = Path(f"../data/processed/{decade}")
        if decade_dir.exists():
            files = list(decade_dir.glob("*.txt"))
            gutenberg_count += len(files)
            logger.info(f"  {decade}: {len(files)} texts")
    
    logger.info(f"Total Gutenberg texts: {gutenberg_count}")
    return gutenberg_count

def collect_oscar_data():
    """Collect from OSCAR dataset (1950s-2020s)"""
    logger.info("🚀 Collecting OSCAR data for modern decades...")
    
    modern_decades = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    oscar_texts = {decade: [] for decade in modern_decades}
    
    try:
        # Load OSCAR dataset - simple approach
        logger.info("Loading OSCAR dataset...")
        dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", streaming=True)
        
        processed = 0
        collected = 0
        target_total = len(modern_decades) * TEXTS_PER_DECADE
        
        for example in dataset:
            if collected >= target_total:
                break
                
            processed += 1
            if processed % 10000 == 0:
                logger.info(f"Processed {processed} OSCAR samples, collected {collected}/{target_total}")
            
            if "text" not in example or not example["text"]:
                continue
                
            text = example["text"]
            if len(text) < 1000:  # Skip short texts
                continue
            
            # Simple decade detection - look for years in text
            import re
            years = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', text)
            if not years:
                continue
            
            # Assign to most common decade
            decade_counts = {}
            for year in years:
                decade = f"{int(year) // 10 * 10}s"
                if decade in modern_decades:
                    decade_counts[decade] = decade_counts.get(decade, 0) + 1
            
            if not decade_counts:
                continue
                
            best_decade = max(decade_counts.keys(), key=lambda d: decade_counts[d])
            
            # Add if we need more for this decade
            if len(oscar_texts[best_decade]) < TEXTS_PER_DECADE:
                oscar_texts[best_decade].append(text)
                collected += 1
                
                # Save immediately to avoid losing data
                decade_file = OUTPUT_DIR / f"{best_decade}_oscar.json"
                with open(decade_file, 'w') as f:
                    json.dump(oscar_texts[best_decade], f)
        
        # Log results
        total_oscar = sum(len(texts) for texts in oscar_texts.values())
        logger.info(f"✅ OSCAR collection complete: {total_oscar} texts")
        for decade, texts in oscar_texts.items():
            logger.info(f"  {decade}: {len(texts)} texts")
            
        return total_oscar
        
    except Exception as e:
        logger.error(f"❌ OSCAR collection failed: {e}")
        return 0

def collect_british_library_data():
    """Collect from British Library (1500s-1900s)"""
    logger.info("🚀 Collecting British Library data...")
    
    historical_decades = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s"]
    bl_texts = {decade: [] for decade in historical_decades}
    
    try:
        # Try to load British Library dataset - simple approach
        logger.info("Loading British Library dataset...")
        
        # Try different configs
        configs = ["1800_1899", "1700_1799", "1510_1699"]
        
        for config in configs:
            try:
                logger.info(f"Trying British Library config: {config}")
                dataset = load_dataset("TheBritishLibrary/blbooks", config, split="train")
                
                processed = 0
                collected = 0
                
                for record in dataset:
                    processed += 1
                    if processed % 1000 == 0:
                        logger.info(f"Processed {processed} BL records from {config}, collected {collected}")
                    
                    if "text" not in record or not record["text"]:
                        continue
                        
                    text = record["text"]
                    if len(text) < 1000:  # Skip short texts
                        continue
                    
                    # Simple assignment based on config
                    if config == "1800_1899":
                        target_decades = ["1850s", "1860s", "1870s", "1880s", "1890s"]
                    elif config == "1700_1799":
                        target_decades = ["1850s", "1860s"]  # Map 18th century to early Victorian
                    else:  # 1510_1699
                        target_decades = ["1850s"]  # Map early modern to Victorian
                    
                    # Assign to decade that needs more texts
                    available_decades = [d for d in target_decades if len(bl_texts[d]) < TEXTS_PER_DECADE]
                    if available_decades:
                        chosen_decade = min(available_decades, key=lambda d: len(bl_texts[d]))
                        bl_texts[chosen_decade].append(text)
                        collected += 1
                        
                        # Save immediately
                        decade_file = OUTPUT_DIR / f"{chosen_decade}_british_library.json"
                        with open(decade_file, 'w') as f:
                            json.dump(bl_texts[chosen_decade], f)
                
                logger.info(f"✅ Completed British Library config {config}: {collected} texts")
                
            except Exception as e:
                logger.warning(f"⚠️ British Library config {config} failed: {e}")
                continue
        
        # Log results
        total_bl = sum(len(texts) for texts in bl_texts.values())
        logger.info(f"✅ British Library collection complete: {total_bl} texts")
        for decade, texts in bl_texts.items():
            logger.info(f"  {decade}: {len(texts)} texts")
            
        return total_bl
        
    except Exception as e:
        logger.error(f"❌ British Library collection failed: {e}")
        return 0

def collect_wikipedia_data():
    """Collect from Wikipedia for recent decades"""
    logger.info("🚀 Collecting Wikipedia data for recent decades...")
    
    recent_decades = ["1990s", "2000s", "2010s", "2020s"]
    wiki_texts = {decade: [] for decade in recent_decades}
    
    try:
        # Load Simple Wikipedia
        logger.info("Loading Simple Wikipedia dataset...")
        dataset = load_dataset("wikipedia", "20220301.simple", split="train")
        
        processed = 0
        collected = 0
        target_total = len(recent_decades) * TEXTS_PER_DECADE
        
        for article in dataset:
            if collected >= target_total:
                break
                
            processed += 1
            if processed % 5000 == 0:
                logger.info(f"Processed {processed} Wikipedia articles, collected {collected}/{target_total}")
            
            if "text" not in article or not article["text"]:
                continue
                
            text = article["text"]
            if len(text) < 1000:  # Skip short articles
                continue
            
            # Simple decade detection
            import re
            years = re.findall(r'\b(19[9]\d|20[0-2]\d)\b', text)
            if not years:
                continue
            
            # Find most common decade
            decade_counts = {}
            for year in years:
                decade = f"{int(year) // 10 * 10}s"
                if decade in recent_decades:
                    decade_counts[decade] = decade_counts.get(decade, 0) + 1
            
            if not decade_counts:
                continue
                
            best_decade = max(decade_counts.keys(), key=lambda d: decade_counts[d])
            
            # Add if we need more for this decade
            if len(wiki_texts[best_decade]) < TEXTS_PER_DECADE:
                wiki_texts[best_decade].append(text)
                collected += 1
                
                # Save immediately
                decade_file = OUTPUT_DIR / f"{best_decade}_wikipedia.json"
                with open(decade_file, 'w') as f:
                    json.dump(wiki_texts[best_decade], f)
        
        # Log results
        total_wiki = sum(len(texts) for texts in wiki_texts.values())
        logger.info(f"✅ Wikipedia collection complete: {total_wiki} texts")
        for decade, texts in wiki_texts.items():
            logger.info(f"  {decade}: {len(texts)} texts")
            
        return total_wiki
        
    except Exception as e:
        logger.error(f"❌ Wikipedia collection failed: {e}")
        return 0

def main():
    """Main data collection function"""
    logger.info("🚀 STARTING SIMPLE REAL DATA COLLECTION")
    logger.info("=" * 60)
    
    total_texts = 0
    
    # 1. Count existing Gutenberg data
    gutenberg_count = collect_gutenberg_data()
    total_texts += gutenberg_count
    
    # 2. Collect OSCAR data for modern decades
    oscar_count = collect_oscar_data()
    total_texts += oscar_count
    
    # 3. Collect British Library data for historical decades
    bl_count = collect_british_library_data()
    total_texts += bl_count
    
    # 4. Collect Wikipedia data for recent decades
    wiki_count = collect_wikipedia_data()
    total_texts += wiki_count
    
    # Final summary
    logger.info("=" * 60)
    logger.info("📊 FINAL DATA COLLECTION SUMMARY")
    logger.info(f"Project Gutenberg (1850s-1920s): {gutenberg_count} texts")
    logger.info(f"OSCAR Dataset (1950s-2020s): {oscar_count} texts")
    logger.info(f"British Library (1850s-1900s): {bl_count} texts")
    logger.info(f"Wikipedia (1990s-2020s): {wiki_count} texts")
    logger.info(f"TOTAL REAL TEXTS: {total_texts}")
    
    # Create summary file
    summary = {
        "total_texts": total_texts,
        "sources": {
            "gutenberg": gutenberg_count,
            "oscar": oscar_count,
            "british_library": bl_count,
            "wikipedia": wiki_count
        },
        "target_per_decade": TEXTS_PER_DECADE,
        "decades_covered": DECADES
    }
    
    with open(OUTPUT_DIR / "collection_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Data collection complete! Check {OUTPUT_DIR} for results.")

if __name__ == "__main__":
    main() 