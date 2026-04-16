#!/usr/bin/env python3
"""
UPDATE SEMANTIC SHIFT WORDS - Replace with historical words
This script updates the semantic shift detector to use words that are more likely to appear in historical texts.
"""

import logging
import json
from pathlib import Path
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HISTORICAL SEMANTIC SHIFT WORDS - More likely to appear in texts
HISTORICAL_SHIFT_WORDS = [
    # HIGH PROBABILITY (appear in most historical texts)
    'broadcast', 'engine', 'power', 'press', 'line', 'call', 'port', 'cell', 
    'gay', 'nice', 'awful', 'cool', 'pilot',
    
    # MEDIUM PROBABILITY (appear in many texts)
    'computer', 'memory', 'network', 'web', 'mouse', 'terminal', 'bank', 
    'credit', 'card', 'account', 'friend', 'community', 'traffic', 'platform', 
    'fuel', 'battery', 'solar', 'grid'
]

def update_semantic_shift_enhancement():
    """Update the semantic shift enhancement file with historical words"""
    logger.info("🔧 Updating semantic shift enhancement with historical words...")
    
    file_path = Path("semantic_shift_analysis/semantic_shift_enhancement.py")
    
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Read the current file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the semantic_shift_words list
        pattern = r'self\.semantic_shift_words\s*=\s*\[(.*?)\]'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            logger.error("❌ Could not find semantic_shift_words list in file")
            return False
        
        # Create new word list
        new_words_list = ',\n            '.join([f"'{word}'" for word in HISTORICAL_SHIFT_WORDS])
        
        # Replace the old list with the new one
        new_content = re.sub(
            pattern, 
            f'self.semantic_shift_words = [\n            {new_words_list}\n        ]',
            content,
            flags=re.DOTALL
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"✅ Updated {file_path} with {len(HISTORICAL_SHIFT_WORDS)} historical words")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating file: {e}")
        return False

def update_main_pipeline():
    """Update the main pipeline to use historical words"""
    logger.info("🔧 Updating main pipeline with historical words...")
    
    file_path = Path("semantic_shift_analysis/main_pipeline.py")
    
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Read the current file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the semantic_shift_words list in the main pipeline
        pattern = r'self\.semantic_shift_words\s*=\s*\[(.*?)\]'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            logger.error("❌ Could not find semantic_shift_words list in main pipeline")
            return False
        
        # Create new word list
        new_words_list = ',\n            '.join([f"'{word}'" for word in HISTORICAL_SHIFT_WORDS])
        
        # Replace the old list with the new one
        new_content = re.sub(
            pattern, 
            f'self.semantic_shift_words = [\n            {new_words_list}\n        ]',
            content,
            flags=re.DOTALL
        )
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"✅ Updated {file_path} with {len(HISTORICAL_SHIFT_WORDS)} historical words")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating main pipeline: {e}")
        return False

def create_word_analysis_report():
    """Create a report analyzing the historical words"""
    logger.info("📊 Creating word analysis report...")
    
    # Word categories
    word_categories = {
        "Technology/Computing": ['computer', 'memory', 'network', 'web', 'mouse', 'terminal'],
        "Communication/Media": ['broadcast', 'press', 'line', 'call'],
        "Transportation": ['engine', 'traffic', 'platform', 'pilot'],
        "Finance/Commerce": ['bank', 'credit', 'card', 'account'],
        "Social/Cultural": ['friend', 'community', 'gay', 'nice', 'awful', 'cool'],
        "Energy/Environment": ['power', 'fuel', 'battery', 'solar', 'grid'],
        "Infrastructure": ['port', 'cell']
    }
    
    # Create analysis report
    analysis = {
        "total_words": len(HISTORICAL_SHIFT_WORDS),
        "word_categories": word_categories,
        "words_by_category": {cat: len(words) for cat, words in word_categories.items()},
        "all_words": HISTORICAL_SHIFT_WORDS,
        "expected_coverage": "High - these words appear frequently in historical texts",
        "semantic_shift_potential": "High - documented shifts across centuries"
    }
    
    # Save report
    report_file = Path("data/historical_words_analysis.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"✅ Word analysis report saved to {report_file}")
    return analysis

def main():
    """Main function to update semantic shift words"""
    logger.info("🚀 UPDATING SEMANTIC SHIFT WORDS WITH HISTORICAL WORDS")
    logger.info("=" * 60)
    
    # 1. Update semantic shift enhancement
    success1 = update_semantic_shift_enhancement()
    
    # 2. Update main pipeline
    success2 = update_main_pipeline()
    
    # 3. Create analysis report
    analysis = create_word_analysis_report()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 UPDATE SUMMARY")
    logger.info(f"Semantic Shift Enhancement: {'✅' if success1 else '❌'}")
    logger.info(f"Main Pipeline: {'✅' if success2 else '❌'}")
    logger.info(f"Word Analysis Report: ✅")
    logger.info(f"Total Historical Words: {len(HISTORICAL_SHIFT_WORDS)}")
    logger.info(f"Word Categories: {len(analysis['word_categories'])}")
    
    if success1 and success2:
        logger.info("🎯 SUCCESS: All files updated with historical words!")
        logger.info("📝 Next step: Run the aggressive data collector")
        return True
    else:
        logger.error("❌ FAILURE: Some files could not be updated")
        return False

if __name__ == "__main__":
    main() 