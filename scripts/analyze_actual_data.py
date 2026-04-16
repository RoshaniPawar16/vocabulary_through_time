#!/usr/bin/env python3
"""
ANALYZE ACTUAL DATA - Check what data we actually have in the directories
"""

import json
import logging
from pathlib import Path
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_files():
    """Analyze all data files we actually have"""
    
    logger.info("🔍 ANALYZING ACTUAL DATA COMPOSITION")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    all_data = {}
    
    # Analyze main data directory
    logger.info("📁 Analyzing main data directory...")
    for file in data_dir.glob("*.json"):
        if file.name == "pipeline_config.json":
            continue
            
        decade = file.stem.replace("_texts", "")
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Count texts and analyze content
            if isinstance(content, list):
                text_count = len(content)
                sample_texts = content[:3] if content else []
            else:
                text_count = 0
                sample_texts = []
            
            all_data[decade] = {
                "file": str(file),
                "text_count": text_count,
                "file_size_mb": file.stat().st_size / (1024 * 1024),
                "sample_texts": sample_texts,
                "source": "main_data_dir"
            }
            
            logger.info(f"  {decade}: {text_count} texts, {file.stat().st_size / (1024 * 1024):.2f} MB")
            
        except Exception as e:
            logger.error(f"  Error reading {file}: {e}")
    
    # Analyze modern_texts subdirectory
    modern_dir = data_dir / "modern_texts"
    if modern_dir.exists():
        logger.info("📱 Analyzing modern_texts subdirectory...")
        for file in modern_dir.glob("*.json"):
            if file.name == "modern_data_summary.json":
                continue
                
            # Extract decade and source from filename (e.g., "1950s_wikipedia.json")
            parts = file.stem.split("_")
            if len(parts) >= 2:
                decade = parts[0]
                source = parts[1]
            else:
                decade = file.stem
                source = "unknown"
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                if isinstance(content, list):
                    text_count = len(content)
                    sample_texts = content[:2] if content else []
                else:
                    text_count = 0
                    sample_texts = []
                
                # Add to existing decade or create new entry
                if decade in all_data:
                    # Add to existing data
                    all_data[decade]["text_count"] += text_count
                    all_data[decade]["file_size_mb"] += file.stat().st_size / (1024 * 1024)
                    all_data[decade]["sample_texts"].extend(sample_texts)
                    all_data[decade]["sources"] = all_data[decade].get("sources", []) + [source]
                else:
                    all_data[decade] = {
                        "file": str(file),
                        "text_count": text_count,
                        "file_size_mb": file.stat().st_size / (1024 * 1024),
                        "sample_texts": sample_texts,
                        "sources": [source],
                        "source": "modern_texts_dir"
                    }
                
                logger.info(f"  {decade} ({source}): {text_count} texts, {file.stat().st_size / (1024 * 1024):.2f} MB")
                
            except Exception as e:
                logger.error(f"  Error reading {file}: {e}")
    
    return all_data

def analyze_content_quality(all_data):
    """Analyze the quality and nature of the content"""
    
    logger.info("📊 ANALYZING CONTENT QUALITY")
    logger.info("=" * 40)
    
    for decade, data in sorted(all_data.items()):
        logger.info(f"\n📅 {decade}:")
        logger.info(f"  Total texts: {data['text_count']}")
        logger.info(f"  Total size: {data['file_size_mb']:.2f} MB")
        logger.info(f"  Sources: {data.get('sources', ['main_data_dir'])}")
        
        # Analyze sample texts
        if data['sample_texts']:
            logger.info("  Sample content analysis:")
            for i, text in enumerate(data['sample_texts'][:2]):
                if isinstance(text, list):
                    # Handle nested structure
                    actual_text = text[0] if text else "No text"
                    source = text[1] if len(text) > 1 else "unknown"
                else:
                    actual_text = str(text)
                    source = "unknown"
                
                # Truncate for display
                display_text = actual_text[:200] + "..." if len(actual_text) > 200 else actual_text
                logger.info(f"    Sample {i+1}: {display_text}")
                logger.info(f"    Source: {source}")
                logger.info(f"    Length: {len(actual_text)} chars")

def create_comprehensive_report(all_data):
    """Create a comprehensive data composition report"""
    
    logger.info("📋 CREATING COMPREHENSIVE DATA REPORT")
    logger.info("=" * 40)
    
    # Categorize decades
    historical_decades = ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s"]
    mid_century_decades = ["1930s", "1940s"]
    modern_decades = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
    
    # Calculate totals
    historical_count = sum(all_data.get(d, {}).get("text_count", 0) for d in historical_decades)
    mid_century_count = sum(all_data.get(d, {}).get("text_count", 0) for d in mid_century_decades)
    modern_count = sum(all_data.get(d, {}).get("text_count", 0) for d in modern_decades)
    total_count = sum(data.get("text_count", 0) for data in all_data.values())
    
    # Analyze data sources
    sources_analysis = {}
    for decade, data in all_data.items():
        sources = data.get("sources", ["main_data_dir"])
        for source in sources:
            if source not in sources_analysis:
                sources_analysis[source] = {"decades": [], "total_texts": 0}
            sources_analysis[source]["decades"].append(decade)
            sources_analysis[source]["total_texts"] += data.get("text_count", 0)
    
    report = {
        "total_texts": total_count,
        "decades_covered": sorted(all_data.keys()),
        "period_breakdown": {
            "historical": {
                "decades": historical_decades,
                "total_texts": historical_count,
                "percentage": (historical_count / total_count * 100) if total_count > 0 else 0
            },
            "mid_century": {
                "decades": mid_century_decades,
                "total_texts": mid_century_count,
                "percentage": (mid_century_count / total_count * 100) if total_count > 0 else 0
            },
            "modern": {
                "decades": modern_decades,
                "total_texts": modern_count,
                "percentage": (modern_count / total_count * 100) if total_count > 0 else 0
            }
        },
        "sources_analysis": sources_analysis,
        "detailed_breakdown": all_data
    }
    
    # Save report
    with open("data/ACTUAL_DATA_COMPOSITION_REPORT.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("✅ Comprehensive report saved to data/ACTUAL_DATA_COMPOSITION_REPORT.json")
    
    return report

def main():
    """Main function"""
    
    logger.info("🚀 ANALYZING ACTUAL DATA COMPOSITION")
    logger.info("=" * 60)
    
    # Analyze all data files
    all_data = analyze_data_files()
    
    if not all_data:
        logger.error("❌ No data files found!")
        return False
    
    # Analyze content quality
    analyze_content_quality(all_data)
    
    # Create comprehensive report
    report = create_comprehensive_report(all_data)
    
    # Summary
    total_texts = sum(data.get("text_count", 0) for data in all_data.values())
    decades_covered = len(all_data.keys())
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL SUMMARY")
    logger.info(f"Total texts: {total_texts:,}")
    logger.info(f"Decades covered: {decades_covered}")
    logger.info(f"Decades: {', '.join(sorted(all_data.keys()))}")
    
    # Period breakdown
    historical_count = sum(all_data.get(d, {}).get("text_count", 0) for d in ["1850s", "1860s", "1870s", "1880s", "1890s", "1900s", "1910s", "1920s"])
    modern_count = sum(all_data.get(d, {}).get("text_count", 0) for d in ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"])
    
    logger.info(f"Historical texts (1850s-1920s): {historical_count:,} ({historical_count/total_texts*100:.1f}%)")
    logger.info(f"Modern texts (1950s-2020s): {modern_count:,} ({modern_count/total_texts*100:.1f}%)")
    
    logger.info("✅ Analysis complete!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 