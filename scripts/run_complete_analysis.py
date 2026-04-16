#!/usr/bin/env python3
"""
COMPLETE ANALYSIS RUNNER - Historical + Modern Data
Runs semantic shift analysis with the complete dataset.
"""

import json
import logging
from pathlib import Path
import sys
import os

# Add the semantic_shift_analysis directory to the path
sys.path.append('semantic_shift_analysis')

from main_pipeline import IntegratedTemporalPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_data():
    """Load both historical and modern data"""
    
    logger.info("🔍 LOADING COMPLETE DATASET")
    logger.info("=" * 50)
    
    all_data = {}
    
    # Load historical data (if exists)
    historical_dir = Path("data/real_historical_texts")
    if historical_dir.exists():
        logger.info("📚 Loading historical data...")
        for file in historical_dir.glob("*.json"):
            decade = file.stem.replace("_texts", "")
            with open(file, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            all_data[decade] = texts
            logger.info(f"  {decade}: {len(texts)} texts")
    
    # Load modern data
    modern_dir = Path("data/modern_texts")
    if modern_dir.exists():
        logger.info("📱 Loading modern data...")
        for file in modern_dir.glob("*.json"):
            if file.name == "modern_data_summary.json":
                continue
            decade = file.stem.split("_")[0]  # Extract decade from filename
            with open(file, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            if decade not in all_data:
                all_data[decade] = []
            all_data[decade].extend(texts)
            logger.info(f"  {decade}: {len(texts)} texts (added)")
    
    # Calculate totals
    total_texts = sum(len(texts) for texts in all_data.values())
    decades_covered = sorted(all_data.keys())
    
    logger.info("=" * 50)
    logger.info("📊 COMPLETE DATASET SUMMARY")
    logger.info(f"Total texts: {total_texts}")
    logger.info(f"Decades covered: {len(decades_covered)}")
    logger.info(f"Decades: {', '.join(decades_covered)}")
    
    # Create detailed breakdown
    breakdown = {}
    for decade in sorted(all_data.keys()):
        breakdown[decade] = len(all_data[decade])
    
    logger.info("📈 Breakdown by decade:")
    for decade, count in breakdown.items():
        logger.info(f"  {decade}: {count} texts")
    
    return all_data, total_texts, decades_covered

def create_data_composition_report(all_data, total_texts, decades_covered):
    """Create a detailed data composition report"""
    
    logger.info("📋 Creating data composition report...")
    
    # Analyze data sources
    historical_decades = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s"]
    modern_decades = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    
    historical_count = sum(len(all_data.get(d, [])) for d in historical_decades)
    modern_count = sum(len(all_data.get(d, [])) for d in modern_decades)
    
    # Check data quality indicators
    data_quality = {}
    for decade, texts in all_data.items():
        if not texts:
            data_quality[decade] = "EMPTY"
            continue
            
        avg_length = sum(len(str(text)) for text in texts) / len(texts)
        data_quality[decade] = {
            "count": len(texts),
            "avg_length": avg_length,
            "quality": "EXCELLENT" if len(texts) >= 50 and avg_length > 1000 else "GOOD" if len(texts) >= 20 else "POOR"
        }
    
    report = {
        "total_texts": total_texts,
        "decades_covered": decades_covered,
        "historical_period": {
            "decades": historical_decades,
            "total_texts": historical_count,
            "percentage": (historical_count / total_texts * 100) if total_texts > 0 else 0
        },
        "modern_period": {
            "decades": modern_decades,
            "total_texts": modern_count,
            "percentage": (modern_count / total_texts * 100) if total_texts > 0 else 0
        },
        "data_quality": data_quality,
        "data_sources": {
            "historical": "Project Gutenberg + British Library",
            "modern": "Wikipedia + Synthetic + Web Scraping"
        }
    }
    
    # Save report
    with open("data/complete_data_composition_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("✅ Data composition report saved to data/complete_data_composition_report.json")
    
    return report

def run_complete_analysis(all_data):
    """Run the complete semantic shift analysis"""
    
    logger.info("🚀 RUNNING COMPLETE SEMANTIC SHIFT ANALYSIS")
    logger.info("=" * 50)
    
    # Create temporary data directory for pipeline
    temp_data_dir = Path("temp_complete_data")
    temp_data_dir.mkdir(exist_ok=True)
    
    # Save all data in pipeline format
    for decade, texts in all_data.items():
        if texts:  # Only save non-empty decades
            decade_file = temp_data_dir / f"{decade}_texts.json"
            with open(decade_file, 'w', encoding='utf-8') as f:
                json.dump(texts, f)
            logger.info(f"  Saved {len(texts)} texts for {decade}")
    
    # Initialize and run pipeline
    try:
        pipeline = IntegratedTemporalPipeline(data_dir=str(temp_data_dir))
        results = pipeline.run_complete_pipeline()
        
        logger.info("✅ COMPLETE ANALYSIS FINISHED!")
        logger.info("=" * 50)
        
        # Display results
        if results and results.get('status') == 'success':
            logger.info("📈 ANALYSIS RESULTS:")
            
            # Training Results
            if 'training_results' in results:
                training = results['training_results']
                logger.info(f"  Training Status: {training.get('bpe_status', 'unknown')} (BPE), {training.get('semantic_status', 'unknown')} (Semantic)")
                
                if 'semantic_performance' in training:
                    perf = training['semantic_performance']
                    logger.info(f"  Semantic Training MSE: {perf.get('mse', 'unknown')}")
                    logger.info(f"  Semantic Training Log10(MSE): {perf.get('log10_mse', 'unknown')}")
            
            # Testing Results
            if 'testing_results' in results:
                testing = results['testing_results']
                logger.info(f"  Testing Status: {testing.get('bpe_status', 'unknown')} (BPE), {testing.get('semantic_status', 'unknown')} (Semantic)")
                
                if 'bpe_performance' in testing:
                    bpe_perf = testing['bpe_performance']
                    logger.info(f"  BPE Test MSE: {bpe_perf.get('mse', 'unknown')}")
                    logger.info(f"  BPE Test Log10(MSE): {bpe_perf.get('log10_mse', 'unknown')}")
                
                if 'semantic_performance' in testing:
                    sem_perf = testing['semantic_performance']
                    logger.info(f"  Semantic Test MSE: {sem_perf.get('mse', 'unknown')}")
                    logger.info(f"  Semantic Test Log10(MSE): {sem_perf.get('log10_mse', 'unknown')}")
            
            # Lambda Ablation Results
            if 'lambda_ablation' in results:
                ablation = results['lambda_ablation']
                logger.info("  Lambda Ablation Results:")
                for lambda_val, perf in ablation.items():
                    logger.info(f"    λ={lambda_val}: MSE={perf.get('mse', 'unknown')}, Log10(MSE)={perf.get('log10_mse', 'unknown')}")
            
            # Statistical Significance
            if 'statistical_significance' in results:
                stats = results['statistical_significance']
                logger.info(f"  Statistical Significance: {stats.get('p_value', 'unknown')}")
                logger.info(f"  Improvement: {stats.get('improvement_percentage', 'unknown')}%")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return None

def main():
    """Main function"""
    
    logger.info("🚀 COMPLETE SEMANTIC SHIFT ANALYSIS WITH FULL DATASET")
    logger.info("=" * 60)
    
    # Load all data
    all_data, total_texts, decades_covered = load_all_data()
    
    if total_texts == 0:
        logger.error("❌ No data available for analysis!")
        return False
    
    # Create data composition report
    report = create_data_composition_report(all_data, total_texts, decades_covered)
    
    # Run complete analysis
    results = run_complete_analysis(all_data)
    
    if results:
        logger.info("🎉 COMPLETE ANALYSIS SUCCESSFUL!")
        logger.info(f"📊 Analyzed {total_texts} texts across {len(decades_covered)} decades")
        logger.info("📁 Results saved in semantic_shift_analysis/output/")
        return True
    else:
        logger.error("❌ Analysis failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 