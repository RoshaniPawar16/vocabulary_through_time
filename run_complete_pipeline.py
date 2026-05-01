#!/usr/bin/env python3
"""
SEMANTIC SHIFT ENHANCEMENT PIPELINE EXECUTOR
Choose between complete and fast execution modes
"""

import sys
import os
import subprocess
from pathlib import Path
import json
from typing import Dict, List, Optional
from src.validation.temporal_inference import TemporalDistributionInference
from src.evaluation import run_evaluation

def check_requirements():
    """Check if all required dependencies are available"""
    print("🔍 Checking requirements...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found. Install with: pip install torch")
        return False
    
    try:
        import sentence_transformers
        print(f"✅ SentenceTransformers: {sentence_transformers.__version__}")
    except ImportError:
        print("❌ SentenceTransformers not found. Install with: pip install sentence-transformers")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not found. Install with: pip install scikit-learn")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found. Install with: pip install numpy")
        return False
    
    # Check data availability
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Make sure your processed data is in data/processed/")
        return False
    
    print(f"✅ Data directory found: {data_dir}")
    
    return True

def display_menu():
    """Display execution options"""
    print("\n" + "="*80)
    print("SEMANTIC SHIFT ENHANCEMENT PIPELINE")
    print("Building upon your existing semantic_shift_enhancement.py")
    print("="*80)
    
    print("\nChoose execution mode:")
    print("\n1. 🚀 FAST EXECUTION (Recommended)")
    print("   • Optimized for speed while maintaining academic integrity")
    print("   • Uses GPU acceleration and caching")
    print("   • Focuses on modern decades (1970s-2020s)")
    print("   • Streamlined manual validation (12 contexts)")
    print("   • Estimated time: 5-15 minutes")
    
    print("\n2. 🎯 COMPLETE EXECUTION")
    print("   • Full implementation with all requirements")
    print("   • Processes all decades and full word list")
    print("   • Comprehensive manual validation (30 contexts)")
    print("   • Maximum academic rigor")
    print("   • Estimated time: 30-60 minutes")
    
    print("\n3. 📊 VIEW EXISTING RESULTS")
    print("   • Display results from previous runs")
    
    print("\n4. 🛠️  CHECK SYSTEM STATUS")
    print("   • Verify dependencies and data availability")
    
    print("\n5. ❌ EXIT")

def view_existing_results():
    """View results from previous runs"""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ No results directory found. Run the pipeline first.")
        return
    
    print(f"\n📁 Results in {results_dir}:")
    
    # Check for result files
    result_files = [
        ("fast_complete_results.json", "Fast execution results"),
        ("complete_honest_results.json", "Complete execution results"),
        ("fast_training_convergence.png", "Fast training plots"),
        ("training_convergence.png", "Complete training plots")
    ]
    
    found_files = []
    for filename, description in result_files:
        filepath = results_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {description}: {filename} ({size_mb:.2f} MB)")
            found_files.append(filepath)
        else:
            print(f"❌ {description}: {filename} (not found)")
    
    if found_files:
        print(f"\n📊 Found {len(found_files)} result files")
        
        # Try to show basic statistics from JSON files
        for filepath in found_files:
            if filepath.suffix == '.json':
                try:
                    import json
                    with open(filepath) as f:
                        data = json.load(f)
                    
                    print(f"\n--- {filepath.name} Summary ---")
                    
                    if 'main_training' in data:
                        perf = data['main_training'].get('training_performance', {})
                        print(f"Training MSE: {perf.get('training_mse', 'N/A')}")
                        print(f"R-squared: {perf.get('r_squared', 'N/A')}")
                        print(f"Words processed: {data['main_training'].get('successful_words', 'N/A')}")
                    
                    if 'manual_clustering_validation' in data:
                        manual = data['manual_clustering_validation']
                        print(f"Manual validation accuracy: {manual.get('overall_accuracy', 'N/A'):.3f}")
                        print(f"Contexts annotated: {manual.get('total_contexts', 'N/A')}")
                    
                    if 'execution_timestamp' in data:
                        print(f"Executed: {data['execution_timestamp']}")
                    
                except Exception as e:
                    print(f"Error reading {filepath.name}: {e}")
    else:
        print("\n❌ No result files found. Run the pipeline first.")

def run_pipeline(mode):
    """Execute the chosen pipeline mode"""
    if mode == "fast":
        print("\n🚀 Starting FAST EXECUTION...")
        print("This will run the speed-optimized pipeline.")
        
        try:
            # Run the fast version
            result = subprocess.run([sys.executable, "semantic_shift_enhancement_fast.py"], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("\n✅ FAST EXECUTION COMPLETED SUCCESSFULLY!")
            else:
                print(f"\n❌ FAST EXECUTION FAILED (return code: {result.returncode})")
                
        except Exception as e:
            print(f"\n❌ Error running fast pipeline: {e}")
    
    elif mode == "complete":
        print("\n🎯 Starting COMPLETE EXECUTION...")
        print("This will run the comprehensive pipeline with all requirements.")
        
        try:
            # Run the complete version
            result = subprocess.run([sys.executable, "semantic_shift_enhancement_complete.py"], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("\n✅ COMPLETE EXECUTION COMPLETED SUCCESSFULLY!")
            else:
                print(f"\n❌ COMPLETE EXECUTION FAILED (return code: {result.returncode})")
                
        except Exception as e:
            print(f"\n❌ Error running complete pipeline: {e}")


def load_corpus(
    data_dir: str,
    decades: Optional[List[str]] = None,
    max_per_decade: int = 20,
) -> Dict[str, List[str]]:
    """
    Load corpus texts from data_dir for each requested decade.

    Handles four storage formats in priority order:
      1. data_dir/gutenberg/{decade}/*.txt
      2. data_dir/ia/{decade}/*.txt
      3. data_dir/{decade}_wikipedia.json  (JSON list of strings)
      4. data_dir/wikipedia_{decade}.json  (JSON dict with 'articles' array)

    Returns a dict mapping decade -> list of text strings (capped at max_per_decade).
    """
    base = Path(data_dir)
    ALL_DECADES = [
        "1850s", "1860s", "1870s", "1880s", "1890s",
        "1900s", "1910s", "1920s", "1930s", "1940s",
        "1950s", "1960s", "1970s", "1980s",
        "1990s", "2000s", "2010s", "2020s",
    ]
    target = decades if decades else ALL_DECADES

    corpus: Dict[str, List[str]] = {}
    for decade in target:
        texts: List[str] = []

        # Format 1: gutenberg .txt directory
        gutenberg_dir = base / "gutenberg" / decade
        if gutenberg_dir.exists():
            for p in sorted(gutenberg_dir.glob("*.txt"))[:max_per_decade]:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))

        # Format 2: internet-archive .txt directory.
        # IA data lives under the project root (vocabulary_through_time/data/),
        # not under the same base as gutenberg and Wikipedia, so we probe both.
        if not texts:
            ia_candidates = [
                base / "ia" / decade,
                Path(__file__).parent / "data" / "real_historical_texts" / "ia" / decade,
            ]
            for ia_dir in ia_candidates:
                if ia_dir.exists():
                    for p in sorted(ia_dir.glob("*.txt"))[:max_per_decade]:
                        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
                    if texts:
                        break

        # Format 3: {decade}_wikipedia.json -- JSON list of strings
        if not texts:
            j = base / f"{decade}_wikipedia.json"
            if j.exists():
                data = json.loads(j.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    texts = [s for s in data if isinstance(s, str)][:max_per_decade]

        # Format 4: wikipedia_{decade}.json -- JSON dict with 'articles' array
        if not texts:
            j = base / f"wikipedia_{decade}.json"
            if j.exists():
                data = json.loads(j.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "articles" in data:
                    texts = [
                        a["text"] for a in data["articles"]
                        if isinstance(a, dict) and "text" in a
                    ][:max_per_decade]

        corpus[decade] = texts
        status = f"{len(texts)} texts" if texts else "NO SOURCE FOUND"
        print(f"  load_corpus: {decade} -> {status}")

    return corpus


def run_research_pipeline(config: dict) -> dict:
    """
    Execute the temporal inference pipeline on real historical corpus data.

    Args:
        config: Dict with keys:
            tokenizers           -- list of HuggingFace tokenizer names
            data_dir             -- path to real_historical_texts root
            output_dir           -- where to write per-tokenizer JSON results
            decades              -- list of decade strings, or None for all available
            use_semantic_shift   -- passed through to TemporalDistributionInference
            run_evaluation       -- whether to call run_evaluation() after inference
            max_texts_per_decade -- cap passed to load_corpus()
    Returns:
        Dict keyed by tokenizer name; each value has 'distribution' and
        optionally 'evaluation' sub-dicts.
    """
    CORPUS_ROOT = Path(__file__).parent.parent / "data" / "real_historical_texts"
    data_dir   = config.get("data_dir") or str(CORPUS_ROOT)
    if not Path(data_dir).is_absolute():
        data_dir = str(CORPUS_ROOT)
    output_dir = config["output_dir"]
    tokenizers = config["tokenizers"]
    decades    = config.get("decades", None)
    do_eval    = config.get("run_evaluation", True)
    max_texts  = config.get("max_texts_per_decade", 20)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(data_dir, decades, max_per_decade=max_texts)
    if not any(corpus.values()):
        raise RuntimeError(
            f"No texts loaded from '{data_dir}'. "
            "Check that data_dir points to real_historical_texts and "
            "that the requested decades exist there."
        )

    results = {}
    for tok_name in tokenizers:
        print(f"\n--- Running: {tok_name} ---")
        analyzer = TemporalDistributionInference(tok_name)

        # Break the pre-existing mutual recursion between infer_temporal_distribution
        # and infer_distribution_ensemble: the default ensemble methods all call
        # infer_temporal_distribution, which on LP failure calls infer_distribution_ensemble
        # again -- infinite loop.  We replace the ensemble on this instance with a
        # heuristic-only path that has no LP call, so the cycle cannot form.
        import types as _types

        def _safe_ensemble(self_a, decade_patterns, methods=None, weights=None):
            """Heuristic-only ensemble fallback -- avoids re-entering the LP path."""
            try:
                filtered = self_a.remove_top_frequent_tokens(decade_patterns, 10)
                return self_a._infer_distribution_heuristic(filtered)
            except Exception as _e:
                logger.warning(f"Heuristic fallback also failed: {_e}; returning uniform")
                decades_k = list(decade_patterns.keys())
                return {d: 1.0 / len(decades_k) for d in decades_k}

        analyzer.infer_distribution_ensemble = _types.MethodType(
            _safe_ensemble, analyzer
        )

        analysis = analyzer.run_analysis(
            corpus,
            use_semantic_shift=config.get('use_semantic_shift', False),
        )

        eval_results = None
        if do_eval:
            eval_results = run_evaluation(analyzer, corpus)

        results[tok_name] = {
            "distribution": analysis.get("distribution", {}),
            "evaluation":   eval_results,
        }

        safe_name = tok_name.replace("/", "_")
        out_path = Path(output_dir) / f"{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(results[tok_name], f, indent=2, default=str)
        print(f"    Saved -> {out_path}")

    return results


def main():
    """Main execution function"""
    
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                if not check_requirements():
                    print("\n❌ Requirements check failed. Please fix the issues above.")
                    continue
                
                confirm = input("\nRun FAST EXECUTION? This will take 5-15 minutes. (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    run_pipeline("fast")
                    
                    # Ask if they want to view results
                    view = input("\nView results summary? (y/n): ").strip().lower()
                    if view in ['y', 'yes']:
                        view_existing_results()
                
            elif choice == "2":
                if not check_requirements():
                    print("\n❌ Requirements check failed. Please fix the issues above.")
                    continue
                
                print("\n⚠️  WARNING: Complete execution takes 30-60 minutes")
                print("   It includes 30 manual context annotations")
                print("   You'll need to be available for manual input")
                
                confirm = input("\nRun COMPLETE EXECUTION? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    run_pipeline("complete")
                    
                    # Ask if they want to view results
                    view = input("\nView results summary? (y/n): ").strip().lower()
                    if view in ['y', 'yes']:
                        view_existing_results()
                
            elif choice == "3":
                view_existing_results()
                
            elif choice == "4":
                check_requirements()
                
            elif choice == "5":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("\n❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("SEMANTIC SHIFT ENHANCEMENT PIPELINE")
    print("Using your existing semantic_shift_enhancement.py as foundation")
    print("Adding professor requirements with full academic integrity")
    
    main() 