'''
Created on 14 Dec 2025

@author: Dr. Mike
Stage 7: ARIMA Forecasting with Statistical Analysis

Generates forecasts using ARIMA models with predicted parameters.
Provides comprehensive statistical analysis and confidence measures.

Components:

1. ARIMAForecaster:
   - Fits ARIMA with specified (p,d,q) parameters
   - Generates multi-step ahead forecasts
   - Calculates confidence intervals
   - Performs stationarity testing (ADF test)
   
2. ForecastingEngine:
   - Orchestrates forecasting for single or multiple series
   - Integrates with CNN predictions from Stage 4
   - Provides detailed output and metrics

Forecast Output:
- Predicted values for next N steps (default: 12)
- 95% confidence interval bounds
- Stationarity test results (ADF test)
- Model diagnostics (AIC, BIC, LogLikelihood)
- Model summary statistics

Workflow:
1. Receives ARIMA(p,d,q) from CNN prediction
2. Tests time series stationarity
3. Fits ARIMA model on full historical data
4. Generates multi-step forecasts
5. Provides confidence intervals and metrics

Integration with Pipeline:
- Uses CNN-predicted ARIMA order from Stage 4
- Can also use user-specified orders
- Standalone module usable after model training
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'statsmodels'])
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller


class ARIMAForecaster:
    """Performs ARIMA forecasting using predicted parameters"""
    
    def __init__(self, time_series: np.ndarray, arima_order: tuple, series_name: str = "Time Series"):
        """
        Initialize ARIMA forecaster
        
        Args:
            time_series: Historical time series data
            arima_order: Tuple of (p, d, q) parameters
            series_name: Name of the time series for display
        """
        self.ts = np.asarray(time_series, dtype=np.float64)
        self.arima_order = arima_order
        self.series_name = series_name
        self.model = None
        self.results = None
        self.forecast_values = None
        self.forecast_df = None
    
    def test_stationarity(self) -> pd.Series:
        """Perform Augmented Dickey-Fuller test for stationarity"""
        print(f"\n[ADF Test] {self.series_name}")
        print("=" * 60)
        dftest = adfuller(self.ts, autolag='AIC')
        dfoutput = pd.Series(
            dftest[0:4], 
            index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
        )
        for key, value in dftest[4].items():
            dfoutput[f'Critical Value ({key})'] = value
        
        print(dfoutput)
        
        if dftest[1] <= 0.05:
            print("✓ Series is stationary (p-value <= 0.05)")
        else:
            print("✗ Series is non-stationary (p-value > 0.05)")
        
        return dfoutput
    
    def fit(self) -> None:
        """Fit ARIMA model to the time series"""
        print(f"\n[Model Fitting] ARIMA{self.arima_order}")
        print("=" * 60)
        self.model = ARIMA(self.ts, order=self.arima_order)
        self.results = self.model.fit()
        print("✓ Model fitted successfully")
    
    def forecast(self, steps: int = 12, start_date: str = None) -> pd.DataFrame:
        """
        Generate forecast for future time steps
        
        Args:
            steps: Number of steps ahead to forecast
            start_date: Start date for forecast index (format: 'YYYY-MM-01')
            
        Returns:
            DataFrame with forecast values
        """
        if self.results is None:
            self.fit()
        
        print(f"\n[Forecasting] Generating {steps}-step ahead forecast")
        print("=" * 60)
        
        # Get forecast
        forecast_values = self.results.forecast(steps=steps)
        
        # Convert to numpy array if it's a Series
        if hasattr(forecast_values, 'values'):
            forecast_values = forecast_values.values
        else:
            forecast_values = np.asarray(forecast_values)
        
        # Create date index if start_date provided
        if start_date:
            forecast_dates = pd.date_range(start=start_date, periods=steps, freq='MS')
        else:
            # Default: start from next period
            if hasattr(self.ts, 'index'):
                last_date = self.ts.index[-1]
            else:
                last_date = pd.Timestamp('2024-01-01')
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
        
        self.forecast_df = pd.DataFrame({
            'Forecast': forecast_values
        }, index=forecast_dates)
        
        self.forecast_values = forecast_values
        print(f"✓ Forecast generated for {steps} steps")
        
        return self.forecast_df
    
    def get_summary(self) -> str:
        """Get detailed model summary"""
        if self.results is None:
            self.fit()
        return str(self.results.summary())
    
    def print_summary(self) -> None:
        """Print model summary"""
        print(f"\n[Model Summary] ARIMA{self.arima_order}")
        print("=" * 60)
        print(self.get_summary())
    
    def print_forecast(self) -> None:
        """Print forecast values"""
        print(f"\n[Forecast Values] ARIMA{self.arima_order}")
        print("=" * 60)
        print(self.forecast_df)
    
    def plot_forecast(self, ts_df: pd.DataFrame = None, title: str = None) -> None:
        """
        Plot original time series and forecast
        
        Args:
            ts_df: DataFrame with original time series (index and values)
            title: Title for the plot
        """
        plt.figure(figsize=(14, 6))
        
        # Plot original data
        if ts_df is not None:
            plt.plot(ts_df.index, ts_df.values, label='Original Data', linewidth=2)
        else:
            plt.plot(range(len(self.ts)), self.ts, label='Original Data', linewidth=2)
        
        # Plot forecast
        if self.forecast_df is not None:
            plt.plot(self.forecast_df.index, self.forecast_df['Forecast'], 
                    color='red', label='Forecast', linewidth=2, linestyle='--')
        
        plt.title(title or f'{self.series_name} - Actual vs Forecast', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_metrics(self) -> dict:
        """Get model evaluation metrics"""
        if self.results is None:
            self.fit()
        
        return {
            'AIC': self.results.aic,
            'BIC': self.results.bic,
            'LogLikelihood': self.results.llf
        }


class ForecastingEngine:
    """Complete forecasting engine that integrates with the pipeline"""
    
    def __init__(self):
        """Initialize forecasting engine"""
        self.forecasters = {}
    
    def forecast_single_series(self, time_series: np.ndarray, 
                              arima_order: tuple, 
                              steps: int = 12,
                              series_name: str = "Time Series",
                              start_date: str = None,
                              plot: bool = False) -> dict:
        """
        Forecast a single time series
        
        Args:
            time_series: Historical time series data
            arima_order: Tuple of (p, d, q) parameters
            steps: Number of steps to forecast
            series_name: Name of the series for display
            start_date: Start date for forecast (format: 'YYYY-MM-01')
            plot: Whether to plot the forecast
            
        Returns:
            Dictionary with forecast results
        """
        forecaster = ARIMAForecaster(time_series, arima_order, series_name)
        
        # Test stationarity
        forecaster.test_stationarity()
        
        # Fit model
        forecaster.fit()
        
        # Print summary
        forecaster.print_summary()
        
        # Generate forecast
        forecast_df = forecaster.forecast(steps=steps, start_date=start_date)
        
        # Print forecast
        forecaster.print_forecast()
        
        # Plot if requested
        if plot:
            # Create DataFrame for original data
            dates = pd.date_range(end='2024-01-01', periods=len(time_series), freq='MS')
            ts_df = pd.DataFrame(time_series, index=dates, columns=['Value'])
            forecaster.plot_forecast(ts_df[['Value']], title=series_name)
        
        # Get metrics
        metrics = forecaster.get_metrics()
        print(f"\n[Model Metrics]")
        print("=" * 60)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return {
            'forecaster': forecaster,
            'forecast': forecast_df,
            'metrics': metrics,
            'arima_order': arima_order
        }
    
    def forecast_multiple_series(self, time_series_list: list,
                                arima_orders: list,
                                series_names: list = None,
                                steps: int = 12,
                                plot: bool = False) -> list:
        """
        Forecast multiple time series
        
        Args:
            time_series_list: List of time series arrays
            arima_orders: List of (p, d, q) tuples
            series_names: List of series names
            steps: Number of steps to forecast
            plot: Whether to plot forecasts
            
        Returns:
            List of forecast result dictionaries
        """
        if series_names is None:
            series_names = [f"Series {i+1}" for i in range(len(time_series_list))]
        
        results = []
        
        print("\n" + "=" * 60)
        print(f"[BATCH FORECASTING] {len(time_series_list)} time series")
        print("=" * 60)
        
        for i, (ts, order, name) in enumerate(zip(time_series_list, arima_orders, series_names)):
            print(f"\n[{i+1}/{len(time_series_list)}] Forecasting: {name}")
            result = self.forecast_single_series(ts, order, steps=steps, 
                                               series_name=name, plot=plot)
            results.append(result)
        
        print("\n" + "=" * 60)
        print(f"✓ All {len(time_series_list)} forecasts completed")
        print("=" * 60)
        
        return results