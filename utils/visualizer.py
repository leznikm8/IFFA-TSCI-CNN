"""
Created on 11 Dec 2025

@author: Dr. Mike

Results Visualization and Reporting

Provides utilities for displaying and analyzing results from the IFFA-TSCI-CNN pipeline.

Components:

1. ResultsVisualizer:
   - Displays validation results in readable format
   - Shows accuracy metrics (mean, std, fold-by-fold)
   - Shows loss metrics
   
2. Plotting Functions:
   - Plot time series data
   - Plot forecasts
   - Compare actual vs predicted
   - Display confidence intervals

Features:
- Clear output formatting
- Statistical summaries
- Matplotlib integration
- Publication-ready figures

Methods:
- plot_validation_results(): Display training/validation metrics
- plot_time_series(): Visualize time series data
- Custom styling for publication-quality plots
"""

import matplotlib.pyplot as plt
import numpy as np


class ResultsVisualizer:
    """Visualizes results from the pipeline"""
    
    @staticmethod
    def plot_validation_results(results):
        """Plot validation results"""
        print("\n[RESULTS SUMMARY]")
        print(f"Validation Method: {results['method']}")
        
        if 'mean_accuracy' in results:
            print(f"  Mean Accuracy: {results['mean_accuracy']:.4f} (�{results['std_accuracy']:.4f})")
            print(f"  Mean Loss: {results['mean_loss']:.4f}")
            print(f"  Individual fold accuracies: {results['fold_accuracies']}")
        else:
            print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"  Test Loss: {results['test_loss']:.4f}")
    
    @staticmethod
    def plot_time_series(ts, title="Time Series"):
        """Plot a time series"""
        plt.figure(figsize=(12, 4))
        plt.plot(ts)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()