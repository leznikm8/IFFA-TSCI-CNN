'''
Created on 17 Dec 2025

@author: Dr. Mike
Stage 3: ARIMA Parameter Optimisation (AIC Version)

Alternative implementation using AIC (Akaike Information Criterion) instead of MAPE
for selecting optimal ARIMA(p,d,q) parameters.

Key Differences from MAPE Version:
1. Criterion: AIC instead of MAPE
   - AIC penalizes model complexity
   - Avoids overfitting
   - Lower AIC is better
   
2. Data Usage: Fits on full time series
   - No train/test split needed
   - Uses all available information
   - Faster computation
   
3. Grid Search: Tests all (p,d,q) combinations
   - Same search space as MAPE version
   - AIC calculated for each model
   
Output: Best ARIMA(p,d,q) tuple and AIC value for each time series

To use this version instead of MAPE, change the import in stage6_pipeline.py:
    from .stage3_arima_evaluation_aic import ARIMAParameterOptimizerAIC
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


class ARIMAParameterOptimizerAIC:
    """Evaluates ARIMA models and finds optimal (p, d, q) using AIC"""
    
    def __init__(self, ts: np.ndarray, max_p: int = Config.MAX_P, 
                 max_d: int = Config.MAX_D, max_q: int = Config.MAX_Q):
        """Initialize ARIMA optimizer"""
        self.ts = ts
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.best_order = None
        self.best_aic = float('inf')
        self.results_history = {}
    
    def find_optimal_order(self) -> tuple:
        """Grid search for optimal ARIMA parameters using AIC criterion"""
        print(f"  Grid searching ARIMA parameters using AIC (p={self.max_p}, d={self.max_d}, q={self.max_q})...")
        
        successful = 0
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        # Fit ARIMA on full time series
                        model = ARIMA(self.ts, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        # Get AIC
                        aic = fitted_model.aic
                        self.results_history[(p, d, q)] = aic
                        successful += 1
                        
                        # Update best (lower AIC is better)
                        if aic < self.best_aic:
                            self.best_aic = aic
                            self.best_order = (p, d, q)
                    except Exception as e:
                        self.results_history[(p, d, q)] = float('inf')
                        continue
        
        if self.best_order is None:
            self.best_order = (1, 1, 1)
            self.best_aic = float('inf')
            print(f"    Warning: No valid ARIMA model found, using default ARIMA{self.best_order}")
        else:
            print(f"    Optimal order: ARIMA{self.best_order} with AIC: {self.best_aic:.4f}")
        
        return self.best_order, self.best_aic
    
    def get_best_order(self) -> tuple:
        """Return the best ARIMA order found"""
        if self.best_order is None:
            self.find_optimal_order()
        return self.best_order
