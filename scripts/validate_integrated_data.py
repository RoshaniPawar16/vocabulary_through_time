#!/usr/bin/env python3
"""
DATA VALIDATION SCRIPT - Validate integrated data for semantic shift analysis
"""

import json
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data():
    """Validate the integrated data"""
    logger.info("🔍 Validating integrated data...")
    
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        logger.error("❌ Processed directory not found")
        return False
    
    # Semantic shift words to check for
    shift_words = [
        'broadcast', 'engine', 'power', 'press', 'line', 'call', 'port', 'cell', 
        'gay', 'nice', 'awful', 'cool', 'pilot', 'computer', 'memory', 'network', 
        'web', 'mouse', 'terminal', 'bank', 'credit', 'card', 'account', 'friend', 
        'community', 'traffic', 'platform', 'fuel', 'battery', 'solar', 'grid'
    ]
    
    validation_results = {
        "total_texts": 0,
        "decades_covered": [],
        "words_found": {},
        "texts_with_shift_words": 0,
        "average_shift_words_per_text": 0
    }
    
    total_shift_word_count = 0
    
    # Check each decade
    for decade_dir in sorted(processed_dir.iterdir()):
        if decade_dir.is_dir():
            decade = decade_dir.name
            text_files = list(decade_dir.glob("*.txt"))
            
            if text_files:
                validation_results["decades_covered"].append(decade)
                validation_results["total_texts"] += len(text_files)
                
                decade_shift_words = 0
                
                # Check each text file
                for text_file in text_files:
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read().lower()
                        
                        # Count shift words in this text
                        text_shift_words = sum(1 for word in shift_words if word in text)
                        decade_shift_words += text_shift_words
                        
                        if text_shift_words > 0:
                            validation_results["texts_with_shift_words"] += 1
                            
                    except Exception as e:
                        logger.warning(f"Error reading {text_file}: {e}")
                
                total_shift_word_count += decade_shift_words
                
                # Store decade results
                validation_results["words_found"][decade] = {
                    "texts": len(text_files),
                    "shift_words": decade_shift_words,
                    "avg_per_text": decade_shift_words / len(text_files) if text_files else 0
                }
    
    # Calculate averages
    if validation_results["total_texts"] > 0:
        validation_results["average_shift_words_per_text"] = total_shift_word_count / validation_results["total_texts"]
    
    # Save validation results
    validation_file = Path("data/data_validation_report.json")
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Print summary
    logger.info("📊 VALIDATION SUMMARY")
    logger.info(f"Total texts: {validation_results['total_texts']}")
    logger.info(f"Decades covered: {len(validation_results['decades_covered'])}")
    logger.info(f"Texts with shift words: {validation_results['texts_with_shift_words']}")
    logger.info(f"Average shift words per text: {validation_results['average_shift_words_per_text']:.2f}")
    
    # Check if data is sufficient
    if validation_results["total_texts"] >= 3000:
        logger.info("✅ SUFFICIENT DATA: Ready for semantic shift analysis!")
        return True
    else:
        logger.warning("⚠️ INSUFFICIENT DATA: May need more collection")
        return False

if __name__ == "__main__":
    validate_data()
