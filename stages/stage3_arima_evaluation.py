'''
Created on 16 Dec 2025

@author: Dr. Mike
'''
import numpy as np
from config import Config

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    print("Installing statsmodels...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'statsmodels'])
    from statsmodels.tsa.arima.model import ARIMA


class ARIMAParameterOptimizer:
    """Evaluates ARIMA models and finds optimal (p, d, q)"""
    
    def __init__(self, ts: np.ndarray, max_p: int = Config.MAX_P, 
                 max_d: int = Config.MAX_D, max_q: int = Config.MAX_Q):
        """Initialize ARIMA optimizer"""
        self.ts = ts
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.best_order = None
        self.best_mape = float('inf')
        self.results_history = {}
    
    @staticmethod
    def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = actual != 0
        if mask.sum() == 0:
            return float('inf')
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def find_optimal_order(self, test_size: float = Config.ARIMA_TEST_SIZE) -> tuple:
        """Grid search for optimal ARIMA parameters"""
        n = len(self.ts)
        train_size = int(n * (1 - test_size))
        train_ts = self.ts[:train_size]
        test_ts = self.ts[train_size:]
        
        print(f"  Grid searching ARIMA parameters (p={self.max_p}, d={self.max_d}, q={self.max_q})...")
        
        successful = 0
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        model = ARIMA(train_ts, order=(p, d, q))
                        fitted_model = model.fit()
                        forecast = fitted_model.get_forecast(steps=len(test_ts))
                        predictions = forecast.predicted_mean
                        
                        # Handle both Series and array
                        if hasattr(predictions, 'values'):
                            predictions = predictions.values
                        else:
                            predictions = np.asarray(predictions)
                        
                        mape = self.calculate_mape(test_ts, predictions)
                        self.results_history[(p, d, q)] = mape
                        successful += 1
                        
                        if mape < self.best_mape:
                            self.best_mape = mape
                            self.best_order = (p, d, q)
                    except Exception as e:
                        self.results_history[(p, d, q)] = float('inf')
                        continue
        
        if self.best_order is None:
            self.best_order = (1, 1, 1)
            self.best_mape = float('inf')
            print(f"    Warning: No valid ARIMA model found, using default ARIMA{self.best_order}")
        else:
            print(f"    Optimal order: ARIMA{self.best_order} with MAPE: {self.best_mape:.4f}")
        
        return self.best_order, self.best_mape
    
    def get_best_order(self) -> tuple:
        """Return the best ARIMA order found"""
        if self.best_order is None:
            self.find_optimal_order()
        return self.best_order