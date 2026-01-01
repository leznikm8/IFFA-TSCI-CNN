'''
Created on 14 Dec 2025

@author: Dr. Mike

Main Entry Point: IFFA-TSCI-CNN Complete Pipeline

Executes the full IFFA-TSCI-CNN pipeline for automated ARIMA model selection
and time series forecasting.

Workflow:
1. [STEP 1] Data Generation: Creates 30 synthetic ARIMA time series
2. [STEP 2] Data Preparation: Encodes time series to images (Stages 1-2)
3. [STEP 3] Training & Validation: Trains CNN with validation (Stages 3-5)
4. [STEP 4] Results Analysis: Reports accuracy and metrics
5. [STEP 5] Testing: Predicts ARIMA order for new time series
6. [STEP 6] Forecasting: Generates 12-step forecast with confidence intervals

Configuration:
- All settings in config.py
- Automatically detects GPU availability
- Uses Stratified K-Fold cross-validation by default

Output:
- Training progress logs
- ARIMA parameter search results
- CNN validation metrics
- Forecast values with statistics

This is the standard entry point for running the complete system.
For interactive forecasting with custom data, use main_forecast.py instead.

'''
import numpy as np
import sys
import torch
import pandas as pd
from config import Config
from stages.stage6_pipeline import IFFATSCICNNPipeline
from stages.stage7_forecasting import ForecastingEngine
from utils.data_generator import DataGenerator


def main():
    print("=" * 80)
    print("IFFA-TSCI-CNN: Time Series Forecasting Model Selection & Forecasting")
    print("=" * 80)
    
    print("\n[CONFIG]")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    
    print("\n[STEP 1/6] Data Generation")
    print("-" * 80)
    time_series_data = DataGenerator.generate_data(
        n_samples=Config.N_SAMPLES,
        ts_length=Config.TS_LENGTH,
        seed=Config.RANDOM_SEED
    )
    
    print("\n[STEP 2/6] Data Preparation & Image Encoding")
    print("-" * 80)
    pipeline = IFFATSCICNNPipeline(
        img_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE
    )
    X, y = pipeline.prepare_data(time_series_data)
    
    print("\n[STEP 3/6] Model Training & Validation")
    print("-" * 80)
    results = pipeline.train_and_validate(X, y, Config.VALIDATION_METHOD)
    
    print("\n[STEP 4/6] Results Analysis")
    print("-" * 80)
    print(f"Validation Method: {results['method']}")
    if 'mean_accuracy' in results:
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
    else:
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    
    print("\n[STEP 5/6] Predicting ARIMA Parameters")
    print("-" * 80)
    test_ts = 10 + 5 * np.sin(np.linspace(0, 4*np.pi, Config.TS_LENGTH))
    test_ts = test_ts + np.linspace(0, 2, Config.TS_LENGTH) + np.random.normal(0, 0.3, Config.TS_LENGTH)
    predicted_order, confidence = pipeline.predict(test_ts)
    print(f"Predicted ARIMA Order: {predicted_order}")
    print(f"Confidence: {confidence:.4f}")
    
    print("\n[STEP 6/6] Generating Forecasts")
    print("-" * 80)
    forecasting_engine = ForecastingEngine()
    forecast_result = forecasting_engine.forecast_single_series(
        test_ts, 
        predicted_order, 
        steps=12,
        series_name="Test Time Series",
        start_date='2024-02-01',
        plot=False
    )
    
    print("\n" + "=" * 80)
    print("Project completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
