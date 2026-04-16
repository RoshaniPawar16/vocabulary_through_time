import json
import matplotlib.pyplot as plt
import numpy as np

def plot_regularization_performance(file_path):
    """
    Reads the results file and plots the training and validation MSE 
    against different regularization strengths (alpha).
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'.")
        return

    if 'regularization_results' not in data:
        print("Error: 'regularization_results' not found in the results file.")
        return

    results = data['regularization_results']
    
    alphas = sorted([float(a) for a in results.keys()])
    train_mse = [results[str(a)]['train_mse'] for a in alphas]
    val_mse = [results[str(a)]['val_mse'] for a in alphas]

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, train_mse, 'o-', label='Training MSE')
    plt.plot(alphas, val_mse, 'o-', label='Validation MSE')
    
    plt.xscale('log')
    plt.xlabel('Regularization Strength (alpha)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Model Performance vs. Regularization Strength')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    output_filename = 'regularization_performance.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    plot_regularization_performance('results/overfitting_fixed_results.json') 