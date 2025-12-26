'''
Created on 17 Dec 2025

@author: Dr. Mike
'''

import numpy as np
import sys
import torch
import pandas as pd
from config import Config
from stages.stage6_pipeline import IFFATSCICNNPipeline
from stages.stage7_forecasting import ForecastingEngine
from utils.data_generator import DataGenerator


def load_time_series_from_file(filepath: str) -> np.ndarray:
    """Load time series from CSV or text file"""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            # Try to get first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ts = df[numeric_cols[0]].values
            else:
                raise ValueError("No numeric columns found in CSV")
        else:
            # Load as plain text/space-separated
            ts = np.loadtxt(filepath)
        
        return np.asarray(ts, dtype=np.float64)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None


def load_time_series_from_user_input() -> np.ndarray:
    """Get time series from user input"""
    print("\n[INPUT] Enter time series values (space or comma separated):")
    print("Example: 100 105 110 108 115 120")
    
    try:
        user_input = input("> ").strip()
        
        # Try different delimiters
        if ',' in user_input:
            values = [float(x.strip()) for x in user_input.split(',')]
        else:
            values = [float(x.strip()) for x in user_input.split()]
        
        ts = np.array(values, dtype=np.float64)
        
        if len(ts) < 10:
            print("Warning: Time series should have at least 10 values for meaningful analysis")
        
        return ts
    except Exception as e:
        print(f"Error parsing input: {str(e)}")
        return None


def display_menu():
    """Display main menu"""
    print("\n" + "=" * 80)
    print("IFFA-TSCI-CNN: Time Series Forecasting")
    print("=" * 80)
    print("\n[MENU]")
    print("1. Train model on synthetic data and forecast")
    print("2. Load time series from file and forecast")
    print("3. Enter time series manually and forecast")
    print("4. Exit")
    print("-" * 80)


def train_and_get_pipeline():
    """Train the model on synthetic data"""
    print("\n[TRAINING] Generating synthetic data and training model...")
    print("-" * 80)
    
    # Generate data
    print("\n[STEP 1/3] Data Generation")
    time_series_data = DataGenerator.generate_data(
        n_samples=Config.N_SAMPLES,
        ts_length=Config.TS_LENGTH,
        seed=Config.RANDOM_SEED
    )
    
    # Prepare data
    print("\n[STEP 2/3] Data Preparation & Image Encoding")
    pipeline = IFFATSCICNNPipeline(
        img_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE
    )
    X, y = pipeline.prepare_data(time_series_data)
    
    # Train
    print("\n[STEP 3/3] Model Training & Validation")
    results = pipeline.train_and_validate(X, y, Config.VALIDATION_METHOD)
    
    print("\n[TRAINING RESULTS]")
    print(f"Validation Method: {results['method']}")
    if 'mean_accuracy' in results:
        print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
    else:
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    
    return pipeline


def forecast_user_series(pipeline: IFFATSCICNNPipeline, ts: np.ndarray, series_name: str = "User Time Series"):
    """Forecast for user-provided time series"""
    print("\n" + "=" * 80)
    print(f"[FORECASTING] {series_name}")
    print("=" * 80)
    
    print(f"\nTime series info:")
    print(f"  Length: {len(ts)}")
    print(f"  Mean: {ts.mean():.4f}")
    print(f"  Std: {ts.std():.4f}")
    print(f"  Min: {ts.min():.4f}")
    print(f"  Max: {ts.max():.4f}")
    
    # Predict best ARIMA order
    print("\n[STEP 1] Predicting optimal ARIMA parameters...")
    predicted_order, confidence = pipeline.predict(ts)
    
    print(f"\n[PREDICTION RESULT]")
    print(f"  Predicted ARIMA Order: {predicted_order}")
    print(f"  Model Confidence: {confidence:.4f}")
    
    # Generate forecast
    print("\n[STEP 2] Generating forecast...")
    forecasting_engine = ForecastingEngine()
    
    try:
        forecast_result = forecasting_engine.forecast_single_series(
            ts, 
            predicted_order, 
            steps=12,
            series_name=series_name,
            plot=False
        )
        
        print("\n[FORECAST COMPLETE]")
        print(f"✓ 12-step ahead forecast generated successfully")
        
    except Exception as e:
        print(f"\nError during forecasting: {str(e)}")
        print("Trying alternative ARIMA order (1,1,1)...")
        
        try:
            forecast_result = forecasting_engine.forecast_single_series(
                ts, 
                (1, 1, 1),
                steps=12,
                series_name=series_name,
                plot=False
            )
            print("\n[FORECAST COMPLETE - Using fallback order]")
        except Exception as e2:
            print(f"Forecast failed: {str(e2)}")
            return None
    
    return forecast_result


def main():
    """Main interactive program"""
    pipeline = None
    
    while True:
        display_menu()
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            # Train model and forecast on test data
            pipeline = train_and_get_pipeline()
            
            # Generate test time series
            print("\n[GENERATING TEST TIME SERIES]")
            test_ts = 10 + 5 * np.sin(np.linspace(0, 4*np.pi, Config.TS_LENGTH))
            test_ts = test_ts + np.linspace(0, 2, Config.TS_LENGTH) + np.random.normal(0, 0.3, Config.TS_LENGTH)
            
            forecast_user_series(pipeline, test_ts, "Test Time Series")
        
        elif choice == '2':
            # Load from file
            if pipeline is None:
                print("\n[INFO] Model not trained. Training first...")
                pipeline = train_and_get_pipeline()
            
            filepath = input("\nEnter file path: ").strip()
            ts = load_time_series_from_file(filepath)
            
            if ts is not None:
                forecast_user_series(pipeline, ts, f"Time Series from {filepath}")
        
        elif choice == '3':
            # Manual input
            if pipeline is None:
                print("\n[INFO] Model not trained. Training first...")
                pipeline = train_and_get_pipeline()
            
            ts = load_time_series_from_user_input()
            
            if ts is not None:
                forecast_user_series(pipeline, ts, "User Input Time Series")
        
        elif choice == '4':
            print("\n[EXIT] Thank you for using IFFA-TSCI-CNN!")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please enter 1-4.")
        
        # Ask if user wants to continue
        print("\n" + "-" * 80)
        cont = input("Continue? (y/n): ").strip().lower()
        if cont != 'y':
            print("\n[EXIT] Thank you for using IFFA-TSCI-CNN!")
            sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[EXIT] Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)