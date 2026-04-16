#!/usr/bin/env python3
"""
INTEGRATE AGGRESSIVE DATA - Connect new data to existing pipeline
This script integrates the aggressively collected data with the existing pipeline structure.
"""

import logging
import json
import shutil
from pathlib import Path
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_aggressive_data():
    """Integrate aggressively collected data with existing pipeline"""
    logger.info("🔗 Integrating aggressive data with existing pipeline...")
    
    # Source and destination directories
    aggressive_dir = Path("data/aggressive_collection")
    processed_dir = Path("data/processed")
    
    if not aggressive_dir.exists():
        logger.error(f"❌ Aggressive collection directory not found: {aggressive_dir}")
        return False
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load consolidated data
    consolidated_file = aggressive_dir / "consolidated_data.json"
    if not consolidated_file.exists():
        logger.error(f"❌ Consolidated data file not found: {consolidated_file}")
        return False
    
    try:
        with open(consolidated_file, 'r') as f:
            consolidated_data = json.load(f)
        
        logger.info(f"📊 Loaded consolidated data: {len(consolidated_data)} decades")
        
        # Process each decade
        total_texts = 0
        for decade, texts in consolidated_data.items():
            decade_dir = processed_dir / decade
            decade_dir.mkdir(exist_ok=True)
            
            # Save texts as individual files
            for i, text in enumerate(texts):
                # Clean text (remove extra whitespace, etc.)
                clean_text = ' '.join(text.split())
                
                # Save as .txt file
                text_file = decade_dir / f"text_{i+1:04d}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
            
            logger.info(f"  {decade}: {len(texts)} texts saved")
            total_texts += len(texts)
        
        logger.info(f"✅ Integration complete: {total_texts} total texts")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error integrating data: {e}")
        return False

def create_enhanced_data_composition_report():
    """Create enhanced data composition report"""
    logger.info("📊 Creating enhanced data composition report...")
    
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        logger.error(f"❌ Processed directory not found: {processed_dir}")
        return False
    
    # Analyze data composition
    composition = {
        "total_texts": 0,
        "decades": {},
        "semantic_shift_words": [
            'broadcast', 'engine', 'power', 'press', 'line', 'call', 'port', 'cell', 
            'gay', 'nice', 'awful', 'cool', 'pilot', 'computer', 'memory', 'network', 
            'web', 'mouse', 'terminal', 'bank', 'credit', 'card', 'account', 'friend', 
            'community', 'traffic', 'platform', 'fuel', 'battery', 'solar', 'grid'
        ],
        "data_sources": ["gutenberg", "wikipedia", "synthetic"],
        "collection_strategy": "aggressive_multi_source"
    }
    
    # Count texts per decade
    for decade_dir in sorted(processed_dir.iterdir()):
        if decade_dir.is_dir():
            decade = decade_dir.name
            text_files = list(decade_dir.glob("*.txt"))
            composition["decades"][decade] = len(text_files)
            composition["total_texts"] += len(text_files)
    
    # Save report
    report_file = Path("data/enhanced_data_composition_report.json")
    with open(report_file, 'w') as f:
        json.dump(composition, f, indent=2)
    
    logger.info(f"✅ Enhanced data composition report saved: {report_file}")
    return composition

def update_pipeline_config():
    """Update pipeline configuration to use new data"""
    logger.info("🔧 Updating pipeline configuration...")
    
    # Update main pipeline data directory
    main_pipeline_file = Path("semantic_shift_analysis/main_pipeline.py")
    
    if not main_pipeline_file.exists():
        logger.error(f"❌ Main pipeline file not found: {main_pipeline_file}")
        return False
    
    try:
        # Read current content
        with open(main_pipeline_file, 'r') as f:
            content = f.read()
        
        # Update data directory path
        content = content.replace(
            'data_dir: str = "../data/processed"',
            'data_dir: str = "data/processed"'
        )
        
        # Write updated content
        with open(main_pipeline_file, 'w') as f:
            f.write(content)
        
        logger.info(f"✅ Updated pipeline configuration")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating pipeline config: {e}")
        return False

def create_data_validation_script():
    """Create a script to validate the integrated data"""
    logger.info("🔍 Creating data validation script...")
    
    validation_script = """#!/usr/bin/env python3
\"\"\"
DATA VALIDATION SCRIPT - Validate integrated data for semantic shift analysis
\"\"\"

import json
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data():
    \"\"\"Validate the integrated data\"\"\"
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
"""
    
    # Save validation script
    validation_file = Path("scripts/validate_integrated_data.py")
    validation_file.parent.mkdir(exist_ok=True)
    
    with open(validation_file, 'w') as f:
        f.write(validation_script)
    
    logger.info(f"✅ Data validation script created: {validation_file}")
    return True

def main():
    """Main integration function"""
    logger.info("🚀 INTEGRATING AGGRESSIVE DATA WITH PIPELINE")
    logger.info("=" * 60)
    
    # 1. Integrate data
    success1 = integrate_aggressive_data()
    
    # 2. Create enhanced report
    composition = create_enhanced_data_composition_report()
    
    # 3. Update pipeline config
    success2 = update_pipeline_config()
    
    # 4. Create validation script
    success3 = create_data_validation_script()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 INTEGRATION SUMMARY")
    logger.info(f"Data Integration: {'✅' if success1 else '❌'}")
    logger.info(f"Enhanced Report: ✅")
    logger.info(f"Pipeline Config: {'✅' if success2 else '❌'}")
    logger.info(f"Validation Script: {'✅' if success3 else '❌'}")
    
    if composition:
        logger.info(f"Total Texts: {composition['total_texts']}")
        logger.info(f"Decades Covered: {len(composition['decades'])}")
        logger.info(f"Semantic Shift Words: {len(composition['semantic_shift_words'])}")
    
    if success1 and success2 and success3:
        logger.info("🎯 SUCCESS: Data integration complete!")
        logger.info("📝 Next step: Run validation script and then main pipeline")
        return True
    else:
        logger.error("❌ FAILURE: Some integration steps failed")
        return False

if __name__ == "__main__":
    main() 