import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import mean_squared_error

def simple_forecast(data, periods=3, smoothing_level=0.5, smoothing_trend=0.2):
    """
    Simple time series forecasting for sequences without seasonality.
    
    Args:
        data (list or np.ndarray): Time series data (e.g., [100, 50, 70, ...]).
        periods (int): Number of future values to predict.
        smoothing_level (float): Alpha parameter (0-1).
        smoothing_trend (float): Beta parameter (0-1).
        
    Returns:
        list: Forecasted values.
    """
    data = np.array(data)
    model = Holt(data).fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend
    )
    forecast = model.forecast(periods)
    return forecast.tolist()

def optimize_forecast(data, periods=3):
    """
    Optimizes forecasting by finding the best alpha and beta parameters.
    
    Args:
        data (list or np.ndarray): Time series data.
        periods (int): Number of future values to predict.
        
    Returns:
        list: Optimized forecasted values.
    """
    data = np.array(data)
    best_score = float('inf')
    best_params = {}

    # Grid search for optimal alpha and beta
    for alpha in np.arange(0.1, 1.0, 0.1):
        for beta in np.arange(0.1, 1.0, 0.1):
            try:
                model = Holt(data[:-1]).fit(
                    smoothing_level=alpha, 
                    smoothing_trend=beta
                )
                forecast = model.forecast(1)
                error = mean_squared_error([data[-1]], forecast)
                
                if error < best_score:
                    best_score = error
                    best_params = {'smoothing_level': alpha, 'smoothing_trend': beta}
            except Exception as e:
                # Log the exception if necessary
                continue
    
    # Forecast using the best parameters
    model = Holt(data).fit(**best_params)
    return model.forecast(periods).tolist()

"""
sequence = [100, 50, 70, 110, 105, 80, 90, 120,65,70,90]
forecast = optimize_forecast(sequence, 3)

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(sequence, 'bo-', label='Historical')
plt.plot(
    range(len(sequence), len(sequence) + len(forecast)), 
    forecast, 'ro-', label='Forecast'
)
plt.title("Simple Time Series Forecast")
plt.legend()
plt.grid(True)
plt.show()
"""

