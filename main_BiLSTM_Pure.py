import os
import sys
import random
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model_design_layer import *
from utils import *
from utils_train_model import *
from utils_eval_model import *
from visualization import *

# USER INPUTS & RUN_ID CONFIGURATION
run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

seed = 42 + run_id
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

project_dir = os.path.dirname(os.path.abspath(__file__))

RESET_PERIOD = 45 # time steps

DATA_PATH  = os.path.join(project_dir, 'PINN_DATA')
MODEL_PATH = os.path.join(project_dir, 'PINN_BEST_MODEL_sustain_check', f'run_{run_id}')
RESULT_PATH = os.path.join(project_dir, 'PINN_RESULT_sustain_check', f'PURE_MODEL_RESULTS_RESET_PERIOD_{RESET_PERIOD}', f'run_{run_id}')

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

train_data_file_name             = 'TRAIN_DATA.csv'
nor_prediction_data_file_name    = 'NOR.csv'
ext_prediction_data_file_name_1  = 'EXT1.csv'
ext_prediction_data_file_name_2  = 'EXT2.csv'
ext_prediction_data_file_name_3  = 'EXT3.csv'

# Base Model name
base_model_file_name = 'BiLSTM'

# Save file name
pure_trained_model_file_name = f'{base_model_file_name}_Pure_Trained_Model.pth'

# Normal Prediction
csv_nor_physical_sensor_evaluation_results_file_name = f'{base_model_file_name}_NOR_Physical_Sensor_Evaluation_Results.csv'
csv_nor_virtual_sensor_evaluation_results_file_name  = f'{base_model_file_name}_NOR_Virtual_Sensor_Evaluation_Results.csv'
csv_nor_virtual_sensor_autoreset_evaluation_results_file_name  = f'{base_model_file_name}_NOR_Virtual_Sensor_Autoreset_Evaluation_Results.csv'

scatter_plot_nor_physical_sensor_file_name = f'{base_model_file_name}_Scatter_Plot_NOR_Physical_Sensor'
scatter_plot_nor_virtual_sensor_file_name = f'{base_model_file_name}_Scatter_Plot_NOR_Virtual_Sensor'
scatter_plot_nor_virtual_sensor_autoreset_file_name = f'{base_model_file_name}_Scatter_Plot_NOR_Virtual_Sensor_Autoreset'

line_plot_nor_physical_sensor_file_name = f'{base_model_file_name}_Line_Plot_NOR_Physical_Sensor'
line_plot_nor_virtual_sensor_file_name = f'{base_model_file_name}_Line_Plot_NOR_Virtual_Sensor'
line_plot_nor_virtual_sensor_autoreset_file_name = f'{base_model_file_name}_Line_Plot_NOR_Virtual_Sensor_Autoreset'

# External Prediction 1
csv_ext_physical_sensor_evaluation_results_file_name_1 = f'{base_model_file_name}_EXT_Physical_Sensor_Evaluation_Results_1.csv'
csv_ext_virtual_sensor_evaluation_results_file_name_1 = f'{base_model_file_name}_EXT_Virtual_Sensor_Evaluation_Results_1.csv'
csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_1 = f'{base_model_file_name}_EXT_Virtual_Sensor_Autoreset_Evaluation_Results_1.csv'

scatter_plot_ext_physical_sensor_file_name_1 = f'{base_model_file_name}_Scatter_Plot_EXT_Physical_Sensor_1'
scatter_plot_ext_virtual_sensor_file_name_1 = f'{base_model_file_name}_Scatter_Plot_EXT_Virtual_Sensor_1'
scatter_plot_ext_virtual_sensor_autoreset_file_name_1 = f'{base_model_file_name}_Scatter_Plot_EXT_Virtual_Sensor_Autoreset_1'

line_plot_ext_physical_sensor_file_name_1 = f'{base_model_file_name}_Line_Plot_EXT_Physical_Sensor_1'
line_plot_ext_virtual_sensor_file_name_1 = f'{base_model_file_name}_Line_Plot_EXT_Virtual_Sensor_1'
line_plot_ext_virtual_sensor_autoreset_file_name_1 = f'{base_model_file_name}_Line_Plot_EXT_Virtual_Sensor_Autoreset_1'

# External Prediction 2
csv_ext_physical_sensor_evaluation_results_file_name_2 = f'{base_model_file_name}_EXT_Physical_Sensor_Evaluation_Results_2.csv'
csv_ext_virtual_sensor_evaluation_results_file_name_2 = f'{base_model_file_name}_EXT_Virtual_Sensor_Evaluation_Results_2.csv'
csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_2 = f'{base_model_file_name}_EXT_Virtual_Sensor_Autoreset_Evaluation_Results_2.csv'

scatter_plot_ext_physical_sensor_file_name_2 = f'{base_model_file_name}_Scatter_Plot_EXT_Physical_Sensor_2'
scatter_plot_ext_virtual_sensor_file_name_2 = f'{base_model_file_name}_Scatter_Plot_EXT_Virtual_Sensor_2'
scatter_plot_ext_virtual_sensor_autoreset_file_name_2 = f'{base_model_file_name}_Scatter_Plot_EXT_Virtual_Sensor_Autoreset_2'

line_plot_ext_physical_sensor_file_name_2 = f'{base_model_file_name}_Line_Plot_EXT_Physical_Sensor_2'
line_plot_ext_virtual_sensor_file_name_2 = f'{base_model_file_name}_Line_Plot_EXT_Virtual_Sensor_2'
line_plot_ext_virtual_sensor_autoreset_file_name_2 = f'{base_model_file_name}_Line_Plot_EXT_Virtual_Sensor_Autoreset_2'

# External Prediction 3
csv_ext_physical_sensor_evaluation_results_file_name_3 = f'{base_model_file_name}_EXT_Physical_Sensor_Evaluation_Results_3.csv'
csv_ext_virtual_sensor_evaluation_results_file_name_3 = f'{base_model_file_name}_EXT_Virtual_Sensor_Evaluation_Results_3.csv'
csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_3 = f'{base_model_file_name}_EXT_Virtual_Sensor_Autoreset_Evaluation_Results_3.csv'

scatter_plot_ext_physical_sensor_file_name_3 = f'{base_model_file_name}_Scatter_Plot_EXT_Physical_Sensor_3'
scatter_plot_ext_virtual_sensor_file_name_3 = f'{base_model_file_name}_Scatter_Plot_EXT_Virtual_Sensor_3'
scatter_plot_ext_virtual_sensor_autoreset_file_name_3 = f'{base_model_file_name}_Scatter_Plot_EXT_Virtual_Sensor_Autoreset_3'

line_plot_ext_physical_sensor_file_name_3 = f'{base_model_file_name}_Line_Plot_EXT_Physical_Sensor_3'
line_plot_ext_virtual_sensor_file_name_3 = f'{base_model_file_name}_Line_Plot_EXT_Virtual_Sensor_3'
line_plot_ext_virtual_sensor_autoreset_file_name_3 = f'{base_model_file_name}_Line_Plot_EXT_Virtual_Sensor_Autoreset_3'

TRAIN_DATA_PATH            = os.path.join(DATA_PATH, train_data_file_name)
PREDICTION_NOR_DATA_PATH   = os.path.join(DATA_PATH, nor_prediction_data_file_name)
PREDICTION_EXT_DATA_PATH_1 = os.path.join(DATA_PATH, ext_prediction_data_file_name_1)
PREDICTION_EXT_DATA_PATH_2 = os.path.join(DATA_PATH, ext_prediction_data_file_name_2)
PREDICTION_EXT_DATA_PATH_3 = os.path.join(DATA_PATH, ext_prediction_data_file_name_3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()
print(f"Using device: {device}")

print()
print("TRAIN_DATA_PATH            =", TRAIN_DATA_PATH)
print("PREDICTION_NOR_DATA_PATH   =", PREDICTION_NOR_DATA_PATH)
print("PREDICTION_EXT_DATA_PATH_1 =", PREDICTION_EXT_DATA_PATH_1)
print("PREDICTION_EXT_DATA_PATH_2 =", PREDICTION_EXT_DATA_PATH_2)
print("PREDICTION_EXT_DATA_PATH_3 =", PREDICTION_EXT_DATA_PATH_3)


# SENSOR DATA STRUCTURE
TIMESTAMP_COLUMN = 'Datetime'

OUTDOOR_COLS = ['OD_PM25', 'OD_PM10', 'OD_TEMP', 'OD_RH']
INDOOR_COLS = ['ID_TEMP', 'ID_RH', 'ID_CO2']

FEATURE_COLUMNS = OUTDOOR_COLS + INDOOR_COLS
TARGET_COLUMNS = ['ID_PM25']

# MODEL PARAMETERS
WINDOW_SIZE = 5
INPUT_DIM = len(FEATURE_COLUMNS) + len(TARGET_COLUMNS) # Input dimension for fine-tuning

OUTPUT_DIM = len(TARGET_COLUMNS)

HIDDEN_DIM = 512

# Training parameters
FINETUNE_EPOCHS = 50

FT_LEARNING_RATE = 5e-5

# Data split ratio 
TRAIN_RATIO = 0.8     
VAL_RATIO = 1 - TRAIN_RATIO
# 
def main():
    clear_GPU_memory()

    # Load data
    df_train_data = pd.read_csv(TRAIN_DATA_PATH)
    df_nor_prediction_data = pd.read_csv(PREDICTION_NOR_DATA_PATH)
    df_ext_prediction_data_1 = pd.read_csv(PREDICTION_EXT_DATA_PATH_1)
    df_ext_prediction_data_2 = pd.read_csv(PREDICTION_EXT_DATA_PATH_2)
    df_ext_prediction_data_3 = pd.read_csv(PREDICTION_EXT_DATA_PATH_3)

    df_train_feature = df_train_data[FEATURE_COLUMNS].copy()
    df_train_target = df_train_data[TARGET_COLUMNS].copy()

    df_nor_prediction_feature = df_nor_prediction_data[FEATURE_COLUMNS].copy()
    df_nor_prediction_target = df_nor_prediction_data[TARGET_COLUMNS].copy()

    df_ext_prediction_feature_1 = df_ext_prediction_data_1[FEATURE_COLUMNS].copy()
    df_ext_prediction_target_1 = df_ext_prediction_data_1[TARGET_COLUMNS].copy()

    df_ext_prediction_feature_2 = df_ext_prediction_data_2[FEATURE_COLUMNS].copy()
    df_ext_prediction_target_2 = df_ext_prediction_data_2[TARGET_COLUMNS].copy()

    df_ext_prediction_feature_3 = df_ext_prediction_data_3[FEATURE_COLUMNS].copy()
    df_ext_prediction_target_3 = df_ext_prediction_data_3[TARGET_COLUMNS].copy()

    train_idx = int(len(df_train_target) * TRAIN_RATIO)

    # Preprocess data
    scaler_x = StandardScaler()
    scaler_outdoor = StandardScaler()
    scaler_indoor = StandardScaler()

    # Fit scaler on training data
    np_train_feature = scaler_x.fit_transform(df_train_feature)
    np_train_target = scaler_indoor.fit_transform(df_train_target)

    df_OD_PM25 = df_train_feature[["OD_PM25"]]  # 2D required by StandardScaler
    _ = scaler_outdoor.fit_transform(df_OD_PM25)

    # Split data for training and validation
    np_train_train_feature = np_train_feature[: train_idx, :]
    np_train_train_target = np_train_target[: train_idx]

    np_train_val_feature = np_train_feature[train_idx :, :]
    np_train_val_target = np_train_target[train_idx :]

    np_nor_prediction_feature = scaler_x.transform(df_nor_prediction_feature)
    np_nor_prediction_target = scaler_indoor.transform(df_nor_prediction_target)

    np_ext_prediction_feature_1 = scaler_x.transform(df_ext_prediction_feature_1)
    np_ext_prediction_target_1 = scaler_indoor.transform(df_ext_prediction_target_1)

    np_ext_prediction_feature_2 = scaler_x.transform(df_ext_prediction_feature_2)
    np_ext_prediction_target_2 = scaler_indoor.transform(df_ext_prediction_target_2)

    np_ext_prediction_feature_3 = scaler_x.transform(df_ext_prediction_feature_3)
    np_ext_prediction_target_3 = scaler_indoor.transform(df_ext_prediction_target_3)

    # For training mondel
    train_dataloader = create_3d_tensor(
                                        data_feature=np_train_train_feature, 
                                        data_target=np_train_train_target, 
                                        input_size=INPUT_DIM, 
                                        window_size=WINDOW_SIZE, 
                                        shuffle=True, 
                                        num_workers=4, 
                                        mode='training'
                                        )
    # For validating the trained model
    val_dataloader = create_3d_tensor(
                                        data_feature=np_train_val_feature, 
                                        data_target=np_train_val_target, 
                                        input_size=INPUT_DIM, 
                                        window_size=WINDOW_SIZE, 
                                        shuffle=False, 
                                        num_workers=4, 
                                        mode='training'
                                        )

    # For testing the validated model (with Normal IAQ Conditions)
    nor_prediction_dataloader = create_3d_tensor(
                                                    data_feature=np_nor_prediction_feature, 
                                                    data_target=np_nor_prediction_target, 
                                                    input_size=INPUT_DIM, 
                                                    window_size=WINDOW_SIZE, 
                                                    shuffle=False, 
                                                    num_workers=4, 
                                                    mode='evaluation'
                                                    )

    # For testing the validated model (with Extreme IAQ Conditions) 
    ext_prediction_dataloader_1 = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_1, 
                                                    data_target=np_ext_prediction_target_1, 
                                                    input_size=INPUT_DIM, 
                                                    window_size=WINDOW_SIZE, 
                                                    shuffle=False, 
                                                    num_workers=4, 
                                                    mode='evaluation'
                                                    )

    ext_prediction_dataloader_2 = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_2, 
                                                    data_target=np_ext_prediction_target_2, 
                                                    input_size=INPUT_DIM, 
                                                    window_size=WINDOW_SIZE, 
                                                    shuffle=False, 
                                                    num_workers=4, 
                                                    mode='evaluation'
                                                    )

    ext_prediction_dataloader_3 = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_3, 
                                                    data_target=np_ext_prediction_target_3, 
                                                    input_size=INPUT_DIM, 
                                                    window_size=WINDOW_SIZE, 
                                                    shuffle=False, 
                                                    num_workers=4, 
                                                    mode='evaluation'
                                                    )                                                                                               

    # Initialize and train BiLSTM model
    base_model = BiLSTM(
                        input_dim=INPUT_DIM, 
                        hidden_dim=HIDDEN_DIM, 
                        seq_len=WINDOW_SIZE
                        ).to(device)

    
    base_regressor = torch.nn.Sequential(
                                            torch.nn.Linear(HIDDEN_DIM * 2, 64),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(64, 1),
                                            ).to(device)

    print(f"--- Base Model and Regressor initialized ---")

  
    # _, _ = training_pure_model(
    #                         model=base_model, 
    #                         regressor=base_regressor,
    #                         device=device, 
    #                         scaler_indoor=scaler_indoor,
    #                         train_loader=train_dataloader, 
    #                         val_loader=val_dataloader, 
    #                         learning_rate=FT_LEARNING_RATE, 
    #                         epochs=FINETUNE_EPOCHS, 
    #                         model_file_name=pure_trained_model_file_name,
    #                         save_path=MODEL_PATH
    #                         )

    # Load the trained model
    trained_model, trained_regressor = load_model(
                                                    model=base_model, 
                                                    regressor=base_regressor, 
                                                    device=device, 
                                                    model_name=pure_trained_model_file_name, 
                                                    save_path=MODEL_PATH
                                                    )
    # Normal Prediction - Physical Sensor
    y_nor_ps_true, y_nor_ps_pred = evaluation_pure_physical_sensor(
                                                                    model=trained_model, 
                                                                    regressor=trained_regressor, 
                                                                    device=device, 
                                                                    test_loader=nor_prediction_dataloader,
                                                                    scaler_indoor=scaler_indoor,
                                                                    result_file_name=csv_nor_physical_sensor_evaluation_results_file_name,
                                                                    save_path=RESULT_PATH
                                                                    )

    scatter_plot(   y_nor_ps_true, 
                    y_nor_ps_pred, 
                    file_name=scatter_plot_nor_physical_sensor_file_name,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Physical Sensor',
                    mode_name='Normal IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )

    line_plot(  y_nor_ps_true, 
                y_nor_ps_pred, 
                file_name=line_plot_nor_physical_sensor_file_name,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Physical Sensor',
                mode_name='Normal IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Normal Prediction - Virtual Sensor
    y_nor_vr_true, y_nor_vr_pred = evaluation_pure_virtual_sensor(
                                                                    model=trained_model, 
                                                                    regressor=trained_regressor, 
                                                                    device=device, 
                                                                    test_loader=nor_prediction_dataloader, 
                                                                    scaler_indoor=scaler_indoor,
                                                                    result_file_name=csv_nor_virtual_sensor_evaluation_results_file_name,
                                                                    save_path=RESULT_PATH
                                                                    )                                   

    scatter_plot(   y_nor_vr_true, 
                    y_nor_vr_pred, 
                    file_name=scatter_plot_nor_virtual_sensor_file_name,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Virtual Sensor',
                    mode_name='Normal IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )
    
    line_plot(  y_nor_vr_true, 
                y_nor_vr_pred, 
                file_name=line_plot_nor_virtual_sensor_file_name,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Virtual Sensor',
                mode_name='Normal IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Normal Prediction - Virtual Sensor - Autoreset
    y_nor_vr_ar_true, y_nor_vr_ar_pred = evaluation_pure_virtual_sensor_autoreset(
                                                                                    model=trained_model, 
                                                                                    regressor=trained_regressor, 
                                                                                    device=device, 
                                                                                    test_loader=nor_prediction_dataloader,
                                                                                    scaler_indoor=scaler_indoor,
                                                                                    result_file_name=csv_nor_virtual_sensor_autoreset_evaluation_results_file_name,
                                                                                    save_path=RESULT_PATH,
                                                                                    reset_period=RESET_PERIOD
                                                                                    )

    scatter_plot(   y_nor_vr_ar_true, 
                    y_nor_vr_ar_pred, 
                    file_name=scatter_plot_nor_virtual_sensor_autoreset_file_name,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Virtual Sensor',
                    mode_name='Normal IAQ Conditions',
                    reset_period=RESET_PERIOD,
                    show_plot=False
                    )

    line_plot(  y_nor_vr_ar_true, 
                y_nor_vr_ar_pred, 
                file_name=line_plot_nor_virtual_sensor_autoreset_file_name,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Virtual Sensor',
                mode_name='Normal IAQ Conditions',
                reset_period=RESET_PERIOD,
                show_plot=False
                )

    # Normal Prediction - Physical Sensor
    y_ext_ps_true_1, y_ext_ps_pred_1 = evaluation_pure_physical_sensor(
                                                                    model=trained_model, 
                                                                    regressor=trained_regressor, 
                                                                    device=device, 
                                                                    test_loader=ext_prediction_dataloader_1,
                                                                    scaler_indoor=scaler_indoor,
                                                                    result_file_name=csv_ext_physical_sensor_evaluation_results_file_name_1,
                                                                    save_path=RESULT_PATH
                                                                    )

    scatter_plot(   y_ext_ps_true_1, 
                    y_ext_ps_pred_1, 
                    file_name=scatter_plot_ext_physical_sensor_file_name_1,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Physical Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )

    line_plot(  y_ext_ps_true_1, 
                y_ext_ps_pred_1, 
                file_name=line_plot_ext_physical_sensor_file_name_1,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Physical Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction - Virtual Sensor
    y_ext_vr_true_1, y_ext_vr_pred_1 = evaluation_pure_virtual_sensor(
                                                                    model=trained_model, 
                                                                    regressor=trained_regressor, 
                                                                    device=device, 
                                                                    test_loader=ext_prediction_dataloader_1, 
                                                                    scaler_indoor=scaler_indoor,
                                                                    result_file_name=csv_ext_virtual_sensor_evaluation_results_file_name_1,
                                                                    save_path=RESULT_PATH
                                                                    )                                   

    scatter_plot(   y_ext_vr_true_1, 
                    y_ext_vr_pred_1, 
                    file_name=scatter_plot_ext_virtual_sensor_file_name_1,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )
    
    line_plot(  y_ext_vr_true_1, 
                y_ext_vr_pred_1, 
                file_name=line_plot_ext_virtual_sensor_file_name_1,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction - Virtual Sensor - Autoreset
    y_ext_vr_ar_true_1, y_ext_vr_ar_pred_1 = evaluation_pure_virtual_sensor_autoreset(
                                                                                    model=trained_model, 
                                                                                    regressor=trained_regressor, 
                                                                                    device=device, 
                                                                                    test_loader=ext_prediction_dataloader_1,
                                                                                    scaler_indoor=scaler_indoor,
                                                                                    result_file_name=csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_1,
                                                                                    save_path=RESULT_PATH,
                                                                                    reset_period=RESET_PERIOD
                                                                                    )

    scatter_plot(   y_ext_vr_ar_true_1, 
                    y_ext_vr_ar_pred_1, 
                    file_name=scatter_plot_ext_virtual_sensor_autoreset_file_name_1,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=RESET_PERIOD,
                    show_plot=False
                    )

    line_plot(  y_ext_vr_ar_true_1, 
                y_ext_vr_ar_pred_1, 
                file_name=line_plot_ext_virtual_sensor_autoreset_file_name_1,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=RESET_PERIOD,
                show_plot=False
                )

    # Normal Prediction - Physical Sensor
    y_ext_ps_true_2, y_ext_ps_pred_2 = evaluation_pure_physical_sensor(
                                                                    model=trained_model, 
                                                                    regressor=trained_regressor, 
                                                                    device=device, 
                                                                    test_loader=ext_prediction_dataloader_2,
                                                                    scaler_indoor=scaler_indoor,
                                                                    result_file_name=csv_ext_physical_sensor_evaluation_results_file_name_2,
                                                                    save_path=RESULT_PATH
                                                                    )

    scatter_plot(   y_ext_ps_true_2, 
                    y_ext_ps_pred_2, 
                    file_name=scatter_plot_ext_physical_sensor_file_name_2,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Physical Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )

    line_plot(  y_ext_ps_true_2, 
                y_ext_ps_pred_2, 
                file_name=line_plot_ext_physical_sensor_file_name_2,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Physical Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction 2 - Virtual Sensor
    y_ext_vr_true_2, y_ext_vr_pred_2 = evaluation_pure_virtual_sensor(
                                                                    model=trained_model, 
                                                                    regressor=trained_regressor, 
                                                                    device=device, 
                                                                    test_loader=ext_prediction_dataloader_2, 
                                                                    scaler_indoor=scaler_indoor,
                                                                    result_file_name=csv_ext_virtual_sensor_evaluation_results_file_name_2,
                                                                    save_path=RESULT_PATH
                                                                    )                                   

    scatter_plot(   y_ext_vr_true_2, 
                    y_ext_vr_pred_2, 
                    file_name=scatter_plot_ext_virtual_sensor_file_name_2,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )
    
    line_plot(  y_ext_vr_true_2, 
                y_ext_vr_pred_2, 
                file_name=line_plot_ext_virtual_sensor_file_name_2,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction 2 - Virtual Sensor - Autoreset
    y_ext_vr_ar_true_2, y_ext_vr_ar_pred_2 = evaluation_pure_virtual_sensor_autoreset(
                                                                                    model=trained_model, 
                                                                                    regressor=trained_regressor, 
                                                                                    device=device, 
                                                                                    test_loader=ext_prediction_dataloader_2,
                                                                                    scaler_indoor=scaler_indoor,
                                                                                    result_file_name=csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_2,
                                                                                    save_path=RESULT_PATH,
                                                                                    reset_period=RESET_PERIOD
                                                                                    )

    scatter_plot(   y_ext_vr_ar_true_2, 
                    y_ext_vr_ar_pred_2, 
                    file_name=scatter_plot_ext_virtual_sensor_autoreset_file_name_2,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=RESET_PERIOD,
                    show_plot=False
                    )

    line_plot(  y_ext_vr_ar_true_2, 
                y_ext_vr_ar_pred_2, 
                file_name=line_plot_ext_virtual_sensor_autoreset_file_name_2,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=RESET_PERIOD,
                show_plot=False
                )

    # Normal Prediction 3 - Physical Sensor
    y_ext_ps_true_3, y_ext_ps_pred_3 = evaluation_pure_physical_sensor(
                                                                    model=trained_model, 
                                                                    regressor=trained_regressor, 
                                                                    device=device, 
                                                                    test_loader=ext_prediction_dataloader_3,
                                                                    scaler_indoor=scaler_indoor,
                                                                    result_file_name=csv_ext_physical_sensor_evaluation_results_file_name_3,
                                                                    save_path=RESULT_PATH
                                                                    )

    scatter_plot(   y_ext_ps_true_3, 
                    y_ext_ps_pred_3, 
                    file_name=scatter_plot_ext_physical_sensor_file_name_3,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Physical Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )

    line_plot(  y_ext_ps_true_3, 
                y_ext_ps_pred_3, 
                file_name=line_plot_ext_physical_sensor_file_name_3,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Physical Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction 3 - Virtual Sensor
    y_ext_vr_true_3, y_ext_vr_pred_3 = evaluation_pure_virtual_sensor(
                                                                    model=trained_model, 
                                                                    regressor=trained_regressor, 
                                                                    device=device, 
                                                                    test_loader=ext_prediction_dataloader_3, 
                                                                    scaler_indoor=scaler_indoor,
                                                                    result_file_name=csv_ext_virtual_sensor_evaluation_results_file_name_3,
                                                                    save_path=RESULT_PATH
                                                                    )                                   

    scatter_plot(   y_ext_vr_true_3, 
                    y_ext_vr_pred_3, 
                    file_name=scatter_plot_ext_virtual_sensor_file_name_3,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )
    
    line_plot(  y_ext_vr_true_3, 
                y_ext_vr_pred_3, 
                file_name=line_plot_ext_virtual_sensor_file_name_1,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction 3 - Virtual Sensor - Autoreset
    y_ext_vr_ar_true_3, y_ext_vr_ar_pred_3 = evaluation_pure_virtual_sensor_autoreset(
                                                                                    model=trained_model, 
                                                                                    regressor=trained_regressor, 
                                                                                    device=device, 
                                                                                    test_loader=ext_prediction_dataloader_3,
                                                                                    scaler_indoor=scaler_indoor,
                                                                                    result_file_name=csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_3,
                                                                                    save_path=RESULT_PATH,
                                                                                    reset_period=RESET_PERIOD
                                                                                    )

    scatter_plot(   y_ext_vr_ar_true_3, 
                    y_ext_vr_ar_pred_3, 
                    file_name=scatter_plot_ext_virtual_sensor_autoreset_file_name_3,
                    save_path=RESULT_PATH,
                    model_mode='PURE',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=RESET_PERIOD,
                    show_plot=False
                    )

    line_plot(  y_ext_vr_ar_true_3, 
                y_ext_vr_ar_pred_3, 
                file_name=line_plot_ext_virtual_sensor_autoreset_file_name_3,
                save_path=RESULT_PATH,
                model_mode='PURE',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=RESET_PERIOD,
                show_plot=False
                )

    clear_GPU_memory()

# Main function
if __name__ == "__main__":
    main()
