import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_astm_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mean_o = np.mean(y_true)
    mean_p = np.mean(y_pred)
    
    var_o = np.var(y_true, ddof=1) # sample variance
    var_p = np.var(y_pred, ddof=1)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Correlation Coefficient (R) 
    R = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Slope of linear regression (b) 
    numerator = np.sum((y_true - mean_o) * (y_pred - mean_p))
    denominator = np.sum((y_true - mean_o) ** 2)
    b = numerator / denominator if denominator != 0 else np.nan
    
    # Intercept (a) 
    a = mean_p - b * mean_o
    
    # Normalized Mean Squared Error (NMSE)
    nmse = mse / (mean_o * mean_p) if (mean_o * mean_p) != 0 else np.nan
    
    # Fractional Bias (FB)
    fb = 2 * (mean_p - mean_o) / (mean_p + mean_o) if (mean_p + mean_o) != 0 else np.nan
    
    # Similar index of bias based on variance (FS)
    fs = 2 * (var_p - var_o) / (var_p + var_o) if (var_p + var_o) != 0 else np.nan
    
    # Coefficient of Determination (R2)
    r2 = r2_score(y_true, y_pred)
    
    # Acceptance Criteria
    criteria = {
        'R >= 0.9': R >= 0.9,
        '0.75 <= b <= 1.25': 0.75 <= b <= 1.25,
        '|a| <= 0.25 * mean_o': abs(a) <= 0.25 * mean_o,
        'NMSE <= 0.25': nmse <= 0.25,
        '|FB| <= 0.25': abs(fb) <= 0.25,
        '|FS| <= 0.5': abs(fs) <= 0.5
    }
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R': R,
        'R2': r2,
        'b': b,
        'a': a,
        'NMSE': nmse,
        'FB': fb,
        'FS': fs,
        'Criteria_Passed': criteria
    }

def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    metrics = calculate_astm_metrics(y_true, y_pred)
    
    print(f"\n--- Evaluation Metrics for {model_name} (Theo ASTM D 5157 & Standard Errors) ---")
    print(f"Mean Squared Error (MSE)    : {metrics['MSE']:.4f}")
    print(f"Root Mean Squared Err (RMSE): {metrics['RMSE']:.4f}")
    print(f"Mean Absolute Error (MAE)   : {metrics['MAE']:.4f}")
    print(f"Mean Abs Percentage (MAPE)  : {metrics['MAPE']:.4f}")
    print(f"Correlation Coefficient (R) : {metrics['R']:.4f} \t(Pass: {metrics['Criteria_Passed']['R >= 0.9']})")
    print(f"Coefficient of Det (R2)     : {metrics['R2']:.4f}")
    print(f"Slope (b)                   : {metrics['b']:.4f} \t(Pass: {metrics['Criteria_Passed']['0.75 <= b <= 1.25']})")
    print(f"Intercept (a)               : {metrics['a']:.4f} \t(Pass: {metrics['Criteria_Passed']['|a| <= 0.25 * mean_o']})")
    print(f"Normalized MSE (NMSE)       : {metrics['NMSE']:.4f} \t(Pass: {metrics['Criteria_Passed']['NMSE <= 0.25']})")
    print(f"Fractional Bias (FB)        : {metrics['FB']:.4f} \t(Pass: {metrics['Criteria_Passed']['|FB| <= 0.25']})")
    print(f"Fractional Var Bias (FS)    : {metrics['FS']:.4f} \t(Pass: {metrics['Criteria_Passed']['|FS| <= 0.5']})")
    print("-" * 65)
    
    return metrics
