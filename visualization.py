import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

def scatter_plot(y_true, y_pred, file_name, save_path, model_mode = "No MB", sensor_name = None, mode_name=None, reset_period=None, show_plot=True):

    # Convert to 1D array to plot the graph
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate the metrics to display on the plot
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=(8, 8))
    
    # Plot the data points
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color='blue', label='Predicted values')
    
    # Linear fit: y_pred = a * y_true + b (x = actual, y = predicted)
    coef = np.polyfit(y_true, y_pred, 1)
    a, b = coef[0], coef[1]
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    x_line = np.linspace(min_val, max_val, 100)
    y_fit = a * x_line + b
    plt.plot(x_line, y_fit, color='red', linestyle='-', lw=2, label=f'Linear fitting')

    # Format the plot (bold title, axis labels, and tick values)
    plt.title(f'{model_mode} Scatter Plot\n{sensor_name} | {mode_name} | Reset Period: {reset_period} (Time Steps)\n(R2: {r2:.4f}, MAE: {mae:.2f})', fontsize=14, fontweight='bold')
    plt.xlabel('Actual values (µg/m3)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted values (µg/m3)', fontsize=14, fontweight='bold')
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)
    eq_text = f'y = {a:.4f}x + {b:.4f}\nR² = {r2:.4f}'
    plt.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=14, fontweight='bold',
             verticalalignment='top')

    plt.legend(prop={'weight': 'bold', 'size': 14})
    plt.grid(True, alpha=0.3)

    # Save the plot
    save_file = os.path.join(save_path, f"{model_mode}_{file_name}_{reset_period}.png")
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    print(f"--- Successfully saved scatter plot to: {save_file} ---")
    if show_plot:
        plt.show()

# Time Series Plot
def line_plot(y_true, y_pred, file_name, save_path, model_mode = "No MB", sensor_name = None, mode_name=None, reset_period=None, show_plot=True):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    plt.figure(figsize=(12, 6)) 
    
    plt.plot(y_true, label='Actual values', color='blue', linewidth=2, alpha=0.8)
    plt.plot(y_pred, label='Predicted values', color='red', linewidth=2, linestyle='--', alpha=0.8)
    
    plt.title(f'{model_mode} Time Series\n{sensor_name} | {mode_name} | Reset Period: {reset_period} (Time Steps)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Concentration (µg/m3)', fontsize=14, fontweight='bold')
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)
    plt.legend(prop={'weight': 'bold', 'size': 14})
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_file = os.path.join(save_path, f"{model_mode}_{file_name}_{reset_period}.png")
    plt.savefig(save_file, dpi=600, bbox_inches='tight')
    
    print(f"--- Successfully saved line plot to: {save_file} ---")
    if show_plot:
        plt.show()
