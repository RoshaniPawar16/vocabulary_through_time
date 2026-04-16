"""
Evaluation Framework for the Temporal Analysis Pipeline

This module provides tools to quantitatively evaluate the performance of the
temporal inference model. It works by creating synthetic documents with a known
(ground truth) temporal distribution and comparing the model's prediction
against this ground truth.

Key Functions:
- create_validation_set: Generates a test document from a specified mix of decades.
- calculate_mse: Computes Mean Squared Error between two distributions.
- calculate_jsd: Computes Jensen-Shannon Divergence between two distributions.
- run_evaluation: Orchestrates the evaluation process over multiple test cases.
"""
import numpy as np
import random
from typing import Dict, List, Tuple
from scipy.spatial.distance import jensenshannon

def create_validation_set(
    all_data: Dict[str, List[str]],
    mixture: Dict[str, float],
    total_words: int = 2000
) -> Tuple[str, Dict[str, float]]:
    """
    Creates a synthetic document by mixing texts from different decades.

    Args:
        all_data: The full dataset, mapping decades to lists of texts.
        mixture: A dictionary defining the proportion of each decade in the mix.
        total_words: The approximate total number of words for the final text.

    Returns:
        A tuple containing:
        - The synthetic text.
        - The ground truth probability distribution.
    """
    final_text_parts = []
    ground_truth = {**{d: 0.0 for d in all_data}, **mixture}

    for decade, proportion in mixture.items():
        if proportion == 0:
            continue
            
        num_words = int(total_words * proportion)
        source_text = " ".join(all_data[decade])
        source_words = source_text.split()
        
        # Take a random slice of the source text
        if len(source_words) > num_words:
            start_index = random.randint(0, len(source_words) - num_words)
            end_index = start_index + num_words
            final_text_parts.append(" ".join(source_words[start_index:end_index]))
        else:
            final_text_parts.append(" ".join(source_words))

    random.shuffle(final_text_parts)
    return " ".join(final_text_parts), ground_truth

def calculate_mse(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Computes the Mean Squared Error."""
    return np.mean((predicted - truth) ** 2)

def calculate_jsd(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Computes the Jensen-Shannon Divergence."""
    # Add small epsilon to avoid division by zero if any probability is 0
    epsilon = 1e-9
    predicted = predicted + epsilon
    truth = truth + epsilon
    predicted /= np.sum(predicted)
    truth /= np.sum(truth)
    return jensenshannon(predicted, truth) ** 2

def run_evaluation(pipeline, all_data: Dict[str, List[str]]):
    """
    Runs a full evaluation suite on the pipeline.
    """
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION SUITE")
    print("=" * 80)

    all_decades = sorted(all_data.keys())

    test_scenarios = {
        "100% 1880s": {"1880s": 1.0},
        "100% 1960s": {"1960s": 1.0},
        "50/50 1880s/1920s": {"1880s": 0.5, "1920s": 0.5},
        "50/50 1920s/1960s": {"1920s": 0.5, "1960s": 0.5},
        "33/33/33 Mix": {"1880s": 0.34, "1920s": 0.33, "1960s": 0.33},
    }

    results = []

    # This assumes the pipeline has a 'run' method that takes text and returns a distribution dict
    # We will build this pipeline orchestrator next.
    # For now, this structure is ready.
    
    for name, mixture in test_scenarios.items():
        print(f"\n--- Testing Scenario: {name} ---")
        
        # We need a pipeline object that can be run.
        # This part is a placeholder for the main pipeline execution flow.
        # pipeline_runner = pipeline(all_data) # Or however it's instantiated
        # predicted_dist = pipeline_runner.run(test_text)
        
        # For now, let's simulate a placeholder result to demonstrate the framework
        test_text, truth_dist_map = create_validation_set(all_data, mixture)
        
        # This is where we would call the actual pipeline
        # For now, we pass the truth back as a placeholder for the prediction
        predicted_dist_map = pipeline.run_inference_on_text(test_text)
        
        if not predicted_dist_map:
             print("  - Pipeline failed to produce a result for this scenario.")
             continue

        truth_vec = np.array([truth_dist_map.get(d, 0.0) for d in all_decades])
        pred_vec = np.array([predicted_dist_map.get(d, 0.0) for d in all_decades])

        mse = calculate_mse(pred_vec, truth_vec)
        jsd = calculate_jsd(pred_vec, truth_vec)

        results.append({'scenario': name, 'mse': mse, 'jsd': jsd})
        
        print(f"  - Ground Truth: {[(k, f'{v:.2f}') for k, v in truth_dist_map.items() if v > 0]}")
        print(f"  - Predicted:    {[(k, f'{v:.2f}') for k, v in sorted(predicted_dist_map.items()) if v > 0.01]}")
        print(f"  - MSE: {mse:.6f}, JSD: {jsd:.6f}")

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    avg_mse = np.mean([r['mse'] for r in results])
    avg_jsd = np.mean([r['jsd'] for r in results])
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average JSD: {avg_jsd:.6f}")

    return results 