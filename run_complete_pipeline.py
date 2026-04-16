#!/usr/bin/env python3
"""
SEMANTIC SHIFT ENHANCEMENT PIPELINE EXECUTOR
Choose between complete and fast execution modes
"""

import sys
import os
import subprocess
from pathlib import Path

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