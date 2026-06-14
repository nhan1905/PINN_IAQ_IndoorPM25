import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    valid_idx = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
 
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n--- Evaluation Metrics for {model_name} (Theo ASTM D 5157 & Standard Errors) ---")
    print(f"Mean Squared Error (MSE)    : {metrics['MSE']:.4f}")
    print(f"Root Mean Squared Err (RMSE): {metrics['RMSE']:.4f}")
    print(f"Mean Absolute Error (MAE)   : {metrics['MAE']:.4f}")
    print(f"Mean Abs Percentage (MAPE)  : {metrics['MAPE']:.4f}")
    print(f"Coefficient of Det (R2)     : {metrics['R2']:.4f}")
    print("-" * 65)
    
    return metrics
