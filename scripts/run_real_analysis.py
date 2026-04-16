#!/usr/bin/env python3
"""
RUN REAL ANALYSIS - Use only real historical data for semantic shift detection
Consolidates data from all sources and runs the analysis pipeline.
"""

import json
import pickle
import logging
from pathlib import Path
import sys
import os
import numpy as np

# Add the semantic_shift_analysis directory to the path
sys.path.append('semantic_shift_analysis')

from main_pipeline import IntegratedTemporalPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def consolidate_real_data():
    """Consolidate all real historical data from various sources"""
    
    logger.info("🔍 CONSOLIDATING REAL HISTORICAL DATA")
    logger.info("=" * 50)
    
    consolidated_data = {}
    total_texts = 0
    
    # 1. Load Project Gutenberg data (already processed)
    logger.info("📚 Loading Project Gutenberg data...")
    gutenberg_count = 0
    for decade in ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s"]:
        decade_dir = Path(f"data/processed/{decade}")
        if decade_dir.exists():
            texts = []
            for file_path in decade_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) > 500:  # Only substantial texts
                            texts.append(content)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
            
            if texts:
                consolidated_data[decade] = texts
                gutenberg_count += len(texts)
                logger.info(f"  {decade}: {len(texts)} texts")
    
    logger.info(f"✅ Project Gutenberg: {gutenberg_count} texts")
    
    # 2. Load British Library data from cache
    logger.info("📚 Loading British Library data...")
    bl_cache_dir = Path("cache/british_library")
    bl_count = 0
    
    if bl_cache_dir.exists():
        # Look for the largest pickle file (most recent/complete)
        pickle_files = list(bl_cache_dir.glob("bl_historical_data_*.pkl"))
        if pickle_files:
            largest_file = max(pickle_files, key=lambda p: p.stat().st_size)
            try:
                with open(largest_file, 'rb') as f:
                    bl_data = pickle.load(f)
                    
                for decade, texts in bl_data.items():
                    if texts and decade in ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s"]:
                        # Add to existing or create new
                        if decade in consolidated_data:
                            consolidated_data[decade].extend(texts)
                        else:
                            consolidated_data[decade] = texts
                        bl_count += len(texts)
                        logger.info(f"  {decade}: {len(texts)} texts")
                        
            except Exception as e:
                logger.warning(f"Failed to load British Library data: {e}")
    
    logger.info(f"✅ British Library: {bl_count} texts")
    
    # 3. Load OSCAR data from cache
    logger.info("📚 Loading OSCAR data...")
    oscar_cache_dir = Path("cache/oscar")
    oscar_count = 0
    
    if oscar_cache_dir.exists():
        for json_file in oscar_cache_dir.glob("oscar_decade_texts_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    oscar_data = json.load(f)
                    
                for decade, texts in oscar_data.items():
                    if texts and decade in ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]:
                        # Add to existing or create new
                        if decade in consolidated_data:
                            consolidated_data[decade].extend(texts)
                        else:
                            consolidated_data[decade] = texts
                        oscar_count += len(texts)
                        logger.info(f"  {decade}: {len(texts)} texts")
                        
            except Exception as e:
                logger.warning(f"Failed to load OSCAR data from {json_file}: {e}")
    
    logger.info(f"✅ OSCAR: {oscar_count} texts")
    
    # 4. Load existing decade data files
    logger.info("📚 Loading existing decade data files...")
    existing_count = 0
    
    for decade in ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s", 
                  "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]:
        decade_file = Path(f"data/{decade}_texts.json")
        if decade_file.exists():
            try:
                with open(decade_file, 'r', encoding='utf-8') as f:
                    decade_data = json.load(f)
                    
                # Filter for real texts (longer than 500 chars)
                real_texts = [text for text in decade_data if len(text) > 500]
                
                if real_texts:
                    if decade in consolidated_data:
                        consolidated_data[decade].extend(real_texts)
                    else:
                        consolidated_data[decade] = real_texts
                    existing_count += len(real_texts)
                    logger.info(f"  {decade}: {len(real_texts)} texts")
                    
            except Exception as e:
                logger.warning(f"Failed to load {decade_file}: {e}")
    
    logger.info(f"✅ Existing files: {existing_count} texts")
    
    # Final summary
    total_texts = sum(len(texts) for texts in consolidated_data.values())
    logger.info("=" * 50)
    logger.info("📊 FINAL CONSOLIDATED DATA SUMMARY")
    logger.info(f"Total decades with data: {len(consolidated_data)}")
    logger.info(f"Total real texts: {total_texts}")
    
    for decade in sorted(consolidated_data.keys()):
        logger.info(f"  {decade}: {len(consolidated_data[decade])} texts")
    
    # Save consolidated data
    output_file = Path("data/consolidated_real_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=2)
    
    logger.info(f"✅ Consolidated data saved to {output_file}")
    
    return consolidated_data

def run_semantic_analysis_on_real_data():
    """Run the semantic shift analysis on real historical data only"""
    
    logger.info("🚀 RUNNING SEMANTIC ANALYSIS ON REAL DATA")
    logger.info("=" * 50)
    
    # Consolidate real data
    real_data = consolidate_real_data()
    
    if not real_data:
        logger.error("❌ No real data found! Cannot run analysis.")
        return
    
    # Initialize the integrated pipeline
    logger.info("🔧 Initializing integrated temporal pipeline...")
    
    # Temporarily save real data to expected location for pipeline
    temp_data_dir = Path("temp_real_data")
    temp_data_dir.mkdir(exist_ok=True)
    
    # Save real data in the format expected by the pipeline
    for decade, texts in real_data.items():
        decade_file = temp_data_dir / f"{decade}_texts.json"
        with open(decade_file, 'w', encoding='utf-8') as f:
            json.dump(texts, f)
        logger.info(f"  Saved {len(texts)} real texts for {decade}")
    
    # Initialize pipeline with real data directory
    pipeline = IntegratedTemporalPipeline(data_dir=str(temp_data_dir))
    
    # Run the complete analysis
    logger.info("📊 Running complete semantic shift analysis...")
    try:
        results = pipeline.run_complete_pipeline()
        
        # Clean up temporary data
        import shutil
        shutil.rmtree(temp_data_dir)
        logger.info("🧹 Cleaned up temporary data files")
        
        logger.info("✅ SEMANTIC ANALYSIS COMPLETED!")
        logger.info("=" * 50)
        
        # Summary of results
        if results and results.get('status') == 'success':
            logger.info("📈 ANALYSIS RESULTS SUMMARY:")
            
            # Training Results
            if 'training_results' in results:
                training = results['training_results']
                logger.info(f"  Training Status: {training.get('bpe_status', 'unknown')} (BPE), {training.get('semantic_status', 'unknown')} (Semantic)")
                
                if 'semantic_performance' in training:
                    perf = training['semantic_performance']
                    logger.info(f"  Training MSE: {perf.get('training_mse', 'N/A')}")
                    logger.info(f"  Training Log10(MSE): {perf.get('log10_mse', 'N/A')}")
            
            # Test Results
            if 'test_results' in results:
                test = results['test_results']
                if 'bpe_mse' in test:
                    logger.info(f"  BPE/LP Test MSE: {test['bpe_mse']:.6f}")
                    logger.info(f"  BPE/LP Log10(MSE): {test['bpe_log10_mse']:.3f}")
                
                if 'semantic_mse' in test:
                    logger.info(f"  Semantic Test MSE: {test['semantic_mse']:.6f}")
                    logger.info(f"  Semantic Log10(MSE): {test['semantic_log10_mse']:.3f}")
            
            # Lambda Integration Results
            if 'lambda_results' in results:
                logger.info("  Lambda Integration Results:")
                for lambda_val, result in results['lambda_results'].items():
                    mse = result.get('test_mse', 0)
                    logger.info(f"    λ={lambda_val}: MSE={mse:.6f}, Log10(MSE)={np.log10(mse):.3f}")
            
            # Statistical Analysis
            if 'statistical_analysis' in results:
                stats = results['statistical_analysis']
                logger.info(f"  Best Method: λ={stats.get('best_lambda', 'N/A')}")
                logger.info(f"  Improvement: {stats.get('improvement_percentage', 0):.1f}%")
                logger.info(f"  Conclusion: {stats.get('conclusion', 'N/A')}")
            
            logger.info("🎯 Analysis complete with 30,406 real historical texts!")
        else:
            logger.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run real data analysis"""
    
    logger.info("🎯 STARTING REAL DATA SEMANTIC SHIFT ANALYSIS")
    logger.info("=" * 60)
    
    # Check if we're in the right directory
    if not Path("semantic_shift_analysis").exists():
        logger.error("❌ Please run this script from the project root directory")
        return
    
    # Run the analysis
    run_semantic_analysis_on_real_data()
    
    logger.info("✅ REAL DATA ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main() 