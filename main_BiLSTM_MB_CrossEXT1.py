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

RESET_PERIOD = 40

EXT_id1 = 1

if EXT_id1 == 1:
    EXT_id2, EXT_id3 = 2, 3
elif EXT_id1 == 2:
    EXT_id2, EXT_id3 = 1, 3
elif EXT_id1 == 3:
    EXT_id2, EXT_id3 = 1, 2
else:
    EXT_id2, EXT_id3 = None, None

project_dir = os.path.dirname(os.path.abspath(__file__))

DATA_PATH  = os.path.join(project_dir, 'PINN_DATA')
MODEL_PATH = os.path.join(project_dir, 'PINN_BEST_MODEL_sustain_check', f'run_{run_id}')
RESULT_PATH = os.path.join(project_dir, 'PINN_RESULT_sustain_check', f'MB_MODEL_CrossEXT{EXT_id1}_TRAIN_RESULTS_RESET_PERIOD_{RESET_PERIOD}', f'run_{run_id}')

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

train_data_file_name             = 'TRAIN_DATA.csv'
ext_train_data_file_name_1       = f'EXT{EXT_id1}.csv'

nor_prediction_data_file_name    = 'NOR.csv'
ext_prediction_data_file_name_2  = f'EXT{EXT_id2}.csv'
ext_prediction_data_file_name_3  = f'EXT{EXT_id3}.csv'

# Base Model name
base_model_file_name = 'BiLSTM' # Have to change of each model

# Save file name
mb_trained_model_file_name = f'{base_model_file_name}_MB_Trained_Model.pth' # Have to change of each model
mb_ext_trained_model_file_name = f'{base_model_file_name}_MB_EXT{EXT_id1}_Trained_Model.pth' # Have to change of each model

# Normal Prediction
csv_nor_physical_sensor_evaluation_results_file_name = f'{base_model_file_name}_MB_{EXT_id1}_NOR_Physical_Sensor_Evaluation_Results.csv'
csv_nor_virtual_sensor_evaluation_results_file_name  = f'{base_model_file_name}_MB_{EXT_id1}_NOR_Virtual_Sensor_Evaluation_Results.csv'
csv_nor_virtual_sensor_autoreset_evaluation_results_file_name  = f'{base_model_file_name}_MB_{EXT_id1}_NOR_Virtual_Sensor_Autoreset_Evaluation_Results.csv'

scatter_plot_nor_physical_sensor_file_name = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_NOR_Physical_Sensor'
scatter_plot_nor_virtual_sensor_file_name = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_NOR_Virtual_Sensor'
scatter_plot_nor_virtual_sensor_autoreset_file_name = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_NOR_Virtual_Sensor_Autoreset'

line_plot_nor_physical_sensor_file_name = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_NOR_Physical_Sensor'
line_plot_nor_virtual_sensor_file_name = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_NOR_Virtual_Sensor'
line_plot_nor_virtual_sensor_autoreset_file_name = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_NOR_Virtual_Sensor_Autoreset'

csv_ext_physical_sensor_evaluation_results_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_EXT_{EXT_id2}_Physical_Sensor_Evaluation_Results.csv'
csv_ext_virtual_sensor_evaluation_results_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_EXT_{EXT_id2}_Virtual_Sensor_Evaluation_Results.csv'
csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_EXT_{EXT_id2}_Virtual_Sensor_Autoreset_Evaluation_Results.csv'

scatter_plot_ext_physical_sensor_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_EXT_{EXT_id2}_Physical_Sensor'
scatter_plot_ext_virtual_sensor_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_EXT_{EXT_id2}_Virtual_Sensor'
scatter_plot_ext_virtual_sensor_autoreset_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_EXT_{EXT_id2}_Virtual_Sensor_Autoreset'

line_plot_ext_physical_sensor_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_EXT_{EXT_id2}_Physical_Sensor'
line_plot_ext_virtual_sensor_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_EXT_{EXT_id2}_Virtual_Sensor'
line_plot_ext_virtual_sensor_autoreset_file_name_2 = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_EXT_{EXT_id2}_Virtual_Sensor_Autoreset'

# Extreme Prediction 3 
csv_ext_physical_sensor_evaluation_results_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_EXT_{EXT_id3}_Physical_Sensor_Evaluation_Results.csv'
csv_ext_virtual_sensor_evaluation_results_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_EXT_{EXT_id3}_Virtual_Sensor_Evaluation_Results.csv'
csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_EXT_{EXT_id3}_Virtual_Sensor_Autoreset_Evaluation_Results.csv'

scatter_plot_ext_physical_sensor_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_EXT_{EXT_id3}_Physical_Sensor'
scatter_plot_ext_virtual_sensor_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_EXT_{EXT_id3}_Virtual_Sensor'
scatter_plot_ext_virtual_sensor_autoreset_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_Scatter_Plot_EXT_{EXT_id3}_Virtual_Sensor_Autoreset'

line_plot_ext_physical_sensor_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_EXT_{EXT_id3}_Physical_Sensor'
line_plot_ext_virtual_sensor_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_EXT_{EXT_id3}_Virtual_Sensor'
line_plot_ext_virtual_sensor_autoreset_file_name_3 = f'{base_model_file_name}_MB_{EXT_id1}_Line_Plot_EXT_{EXT_id3}_Virtual_Sensor_Autoreset'

TRAIN_DATA_PATH            = os.path.join(DATA_PATH, train_data_file_name)
TRAIN_EXT_DATA_PATH_1      = os.path.join(DATA_PATH, ext_train_data_file_name_1)
PREDICTION_NOR_DATA_PATH   = os.path.join(DATA_PATH, nor_prediction_data_file_name)
PREDICTION_EXT_DATA_PATH_2 = os.path.join(DATA_PATH, ext_prediction_data_file_name_2)
PREDICTION_EXT_DATA_PATH_3 = os.path.join(DATA_PATH, ext_prediction_data_file_name_3)

# 2. MODEL's CONFIGURATION

# 2.1 SENSOR DATA STRUCTURE
TIMESTAMP_COLUMN = 'Datetime'

OUTDOOR_COLS = ['OD_PM25', 'OD_PM10', 'OD_TEMP', 'OD_RH']
INDOOR_COLS = ['ID_TEMP', 'ID_RH', 'ID_CO2']

FEATURE_COLUMNS = OUTDOOR_COLS + INDOOR_COLS
TARGET_COLUMNS = ['ID_PM25']

# 2.2 MODEL PARAMETERS
WINDOW_SIZE = 5
INPUT_DIM = len(FEATURE_COLUMNS) + len(TARGET_COLUMNS)  # Input dimension for fine-tuning

OUTPUT_DIM = len(TARGET_COLUMNS)

HIDDEN_DIM = 512

# Training parameters
TRAINING_EPOCHS = 50
LEARNING_RATE = 5e-5

EXT_TRAINING_EPOCHS = 100
EXT_LEARNING_RATE = 5e-5

# Data split ratio
TRAIN_RATIO = 0.8
VAL_RATIO = 1 - TRAIN_RATIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()
print(f"Using device: {device}")

# Print
print()
print("TRAIN_DATA_PATH =", TRAIN_DATA_PATH)
print("TRAIN_EXT_DATA_PATH_1 =", TRAIN_EXT_DATA_PATH_1)
print("PREDICTION_NOR_DATA_PATH =", PREDICTION_NOR_DATA_PATH)
print("PREDICTION_EXT_DATA_PATH_2 =", PREDICTION_EXT_DATA_PATH_2)
print("PREDICTION_EXT_DATA_PATH_3 =", PREDICTION_EXT_DATA_PATH_3)
print("RESET_PERIOD =", RESET_PERIOD)

def main():
    clear_GPU_memory()

    #  Load data
    #  Load CSV files
    df_train_data = pd.read_csv(TRAIN_DATA_PATH)
    df_train_ext_data = pd.read_csv(TRAIN_EXT_DATA_PATH_1)

    df_prediction_nor_data = pd.read_csv(PREDICTION_NOR_DATA_PATH)
    df_prediction_ext_data_2 = pd.read_csv(PREDICTION_EXT_DATA_PATH_2)
    df_prediction_ext_data_3 = pd.read_csv(PREDICTION_EXT_DATA_PATH_3)

    #  Extract features and targets
    df_train_feature = df_train_data[FEATURE_COLUMNS].copy()
    df_train_target = df_train_data[TARGET_COLUMNS].copy()

    df_train_ext_feature = df_train_ext_data[FEATURE_COLUMNS].copy()
    df_train_ext_target = df_train_ext_data[TARGET_COLUMNS].copy()

    df_prediction_nor_feature = df_prediction_nor_data[FEATURE_COLUMNS].copy()
    df_prediction_nor_target = df_prediction_nor_data[TARGET_COLUMNS].copy()

    df_prediction_ext_feature_2 = df_prediction_ext_data_2[FEATURE_COLUMNS].copy()
    df_prediction_ext_target_2 = df_prediction_ext_data_2[TARGET_COLUMNS].copy()

    df_prediction_ext_feature_3 = df_prediction_ext_data_3[FEATURE_COLUMNS].copy()
    df_prediction_ext_target_3 = df_prediction_ext_data_3[TARGET_COLUMNS].copy()

    fine_tune_train_idx = int(len(df_train_target) * TRAIN_RATIO)

    #  Preprocess data
    #  Initialize scalers
    scaler_x = StandardScaler()
    scaler_x_ext = StandardScaler()

    scaler_outdoor = StandardScaler() # For mass balance equation
    scaler_outdoor_ext = StandardScaler()

    scaler_indoor = StandardScaler() 
    scaler_indoor_ext = StandardScaler()

    #  Fit scaler on whole data for autoencoder
    np_train_feature = scaler_x.fit_transform(df_train_feature)
    np_train_target = scaler_indoor.fit_transform(df_train_target)

    np_train_ext_feature = scaler_x_ext.fit_transform(df_train_ext_feature)
    np_train_ext_target = scaler_indoor_ext.fit_transform(df_train_ext_target)

    np_prediction_nor_feature = scaler_x.transform(df_prediction_nor_feature)
    np_prediction_nor_target = scaler_indoor.transform(df_prediction_nor_target)

    np_prediction_ext_feature_2 = scaler_x_ext.transform(df_prediction_ext_feature_2)
    np_prediction_ext_target_2 = scaler_indoor_ext.transform(df_prediction_ext_target_2)

    np_prediction_ext_feature_3 = scaler_x_ext.transform(df_prediction_ext_feature_3)
    np_prediction_ext_target_3 = scaler_indoor_ext.transform(df_prediction_ext_target_3)

    df_OD_PM25 = df_train_feature[["OD_PM25"]]  # 2D required by StandardScaler
    _ = scaler_outdoor.fit_transform(df_OD_PM25)

    df_OD_PM25_EXT = df_train_ext_feature[["OD_PM25"]]  # 2D required by StandardScaler
    _ = scaler_outdoor_ext.fit_transform(df_OD_PM25_EXT)

    #  Split data for training and validation (for extreme training and no validation)
    np_train_train_feature = np_train_feature[: fine_tune_train_idx, :]
    np_train_train_target = np_train_target[: fine_tune_train_idx]

    np_train_val_feature = np_train_feature[fine_tune_train_idx :, :]
    np_train_val_target = np_train_target[fine_tune_train_idx :]

    #  Create dataloader
    #  For training mondel
    train_nor_dataloader = create_3d_tensor(
                                            data_feature=np_train_train_feature, 
                                            data_target=np_train_train_target, 
                                            input_size=INPUT_DIM, 
                                            window_size=WINDOW_SIZE, 
                                            shuffle=True, 
                                            num_workers=4, 
                                            mode='fine-tuning'
                                            )
    
    train_ext_dataloader = create_3d_tensor(
                                            data_feature=np_train_ext_feature, 
                                            data_target=np_train_ext_target, 
                                            input_size=INPUT_DIM, 
                                            window_size=WINDOW_SIZE, 
                                            shuffle=True, 
                                            num_workers=4, 
                                            mode='fine-tuning'
                                            )
    #  For validating the trained model
    val_nor_dataloader = create_3d_tensor(
                                            data_feature=np_train_val_feature, 
                                            data_target=np_train_val_target, 
                                            input_size=INPUT_DIM, 
                                            window_size=WINDOW_SIZE, 
                                            shuffle=False, 
                                            num_workers=4, 
                                            mode='fine-tuning'
                                            )

    #  For testing the validated model (with Normal IAQ Conditions)
    nor_prediction_dataloader = create_3d_tensor(
                                                    data_feature=np_prediction_nor_feature, 
                                                    data_target=np_prediction_nor_target, 
                                                    input_size=INPUT_DIM, 
                                                    window_size=WINDOW_SIZE, 
                                                    shuffle=False, 
                                                    num_workers=4, 
                                                    mode='evaluation'
                                                    )

    #  For testing the validated model (with Extreme IAQ Conditions)
    ext_prediction_dataloader_2 = create_3d_tensor(
                                                    data_feature=np_prediction_ext_feature_2, 
                                                    data_target=np_prediction_ext_target_2, 
                                                    input_size=INPUT_DIM, 
                                                    window_size=WINDOW_SIZE, 
                                                    shuffle=False, 
                                                    num_workers=4, 
                                                    mode='evaluation'
                                                    )

    ext_prediction_dataloader_3 = create_3d_tensor(
                                                    data_feature=np_prediction_ext_feature_3, 
                                                    data_target=np_prediction_ext_target_3, 
                                                    input_size=INPUT_DIM, 
                                                    window_size=WINDOW_SIZE, 
                                                    shuffle=False, 
                                                    num_workers=4, 
                                                    mode='evaluation'
                                                    )

    #  Initialize and train BiLSTM model
    base_model = BiLSTM(
                        input_dim=INPUT_DIM, 
                        hidden_dim=HIDDEN_DIM, 
                        seq_len=WINDOW_SIZE
                        ).to(device)

    base_regressor = torch.nn.Sequential(
                                            torch.nn.Linear(HIDDEN_DIM * 2, 64),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(64, 4),
                                            ).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear) and m.out_features == 4:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(m.bias, 0.0)

    base_regressor.apply(init_weights)

    print(f"--- Base Model and Regressor initialized ---")

    # _, _ = training_mb_model(
    #                         model=base_model, 
    #                         regressor=base_regressor,
    #                         device=device, 
    #                         scaler_outdoor=scaler_outdoor,
    #                         scaler_indoor=scaler_indoor,
    #                         train_loader=train_nor_dataloader, 
    #                         val_loader=val_nor_dataloader, 
    #                         learning_rate=LEARNING_RATE, 
    #                         epochs=TRAINING_EPOCHS, 
    #                         model_file_name=mb_trained_model_file_name,
    #                         save_path=MODEL_PATH
    #                         )
    
    trained_mb_model, trained_mb_regressor = load_model(
                                                    model=base_model, 
                                                    regressor=base_regressor, 
                                                    device=device, 
                                                    model_name=mb_trained_model_file_name, 
                                                    save_path=MODEL_PATH
                                                    )

    # Train the model for extreme dataset
    # _, _ = training_mb_model(
    #                         model=trained_mb_model, 
    #                         regressor=trained_mb_regressor,
    #                         device=device, 
    #                         scaler_outdoor=scaler_outdoor_ext,
    #                         scaler_indoor=scaler_indoor_ext,
    #                         train_loader=train_ext_dataloader, 
    #                         val_loader=None, 
    #                         learning_rate=EXT_LEARNING_RATE, 
    #                         epochs=EXT_TRAINING_EPOCHS, 
    #                         model_file_name=mb_ext_trained_model_file_name,
    #                         save_path=MODEL_PATH
    #                         )

    trained_ext_mb_model, trained_ext_mb_regressor = load_model(
                                                    model=base_model, 
                                                    regressor=base_regressor, 
                                                    device=device, 
                                                    model_name=mb_ext_trained_model_file_name, 
                                                    save_path=MODEL_PATH
                                                    )
    # Normal Prediction
    # Normal Prediction - Physical Sensor
    y_nor_ps_true, y_nor_ps_pred, _, _, _, _ = evaluation_mb_physical_sensor(
                                                                            model=trained_ext_mb_model, 
                                                                            regressor=trained_ext_mb_regressor, 
                                                                            device=device, 
                                                                            test_loader=nor_prediction_dataloader,
                                                                            scaler_outdoor=scaler_outdoor,
                                                                            scaler_indoor=scaler_indoor,
                                                                            result_file_name=csv_nor_physical_sensor_evaluation_results_file_name,
                                                                            save_path=RESULT_PATH
                                                                            )

    scatter_plot(   y_nor_ps_true, 
                    y_nor_ps_pred, 
                    file_name=scatter_plot_nor_physical_sensor_file_name,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Physical Sensor',
                    mode_name='Normal IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )

    line_plot(  y_nor_ps_true, 
                y_nor_ps_pred, 
                file_name=line_plot_nor_physical_sensor_file_name,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Physical Sensor',
                mode_name='Normal IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Normal Prediction - Virtual Sensor
    y_nor_vr_true, y_nor_vr_pred, _, _, _, _ = evaluation_mb_virtual_sensor(
                                                                            model=trained_ext_mb_model, 
                                                                            regressor=trained_ext_mb_regressor, 
                                                                            device=device, 
                                                                            test_loader=nor_prediction_dataloader, 
                                                                            scaler_outdoor=scaler_outdoor,
                                                                            scaler_indoor=scaler_indoor,
                                                                            result_file_name=csv_nor_virtual_sensor_evaluation_results_file_name,
                                                                            save_path=RESULT_PATH
                                                                            )                                   

    scatter_plot(   y_nor_vr_true, 
                    y_nor_vr_pred, 
                    file_name=scatter_plot_nor_virtual_sensor_file_name,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Virtual Sensor',
                    mode_name='Normal IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )
    
    line_plot(  y_nor_vr_true, 
                y_nor_vr_pred, 
                file_name=line_plot_nor_virtual_sensor_file_name,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Virtual Sensor',
                mode_name='Normal IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Normal Prediction - Virtual Sensor - Autoreset
    y_nor_vr_ar_true, y_nor_vr_ar_pred, _, _, _, _ = evaluation_mb_virtual_sensor_autoreset(
                                                                                            model=trained_ext_mb_model, 
                                                                                            regressor=trained_ext_mb_regressor, 
                                                                                            device=device, 
                                                                                            test_loader=nor_prediction_dataloader,
                                                                                            scaler_outdoor=scaler_outdoor,
                                                                                            scaler_indoor=scaler_indoor,
                                                                                            result_file_name=csv_nor_virtual_sensor_autoreset_evaluation_results_file_name,
                                                                                            save_path=RESULT_PATH,
                                                                                            reset_period=RESET_PERIOD
                                                                                            )

    scatter_plot(   y_nor_vr_ar_true, 
                    y_nor_vr_ar_pred, 
                    file_name=scatter_plot_nor_virtual_sensor_autoreset_file_name,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Virtual Sensor',
                    mode_name='Normal IAQ Conditions',
                    reset_period=RESET_PERIOD,
                    show_plot=False
                    )

    line_plot(  y_nor_vr_ar_true, 
                y_nor_vr_ar_pred, 
                file_name=line_plot_nor_virtual_sensor_autoreset_file_name,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Virtual Sensor',
                mode_name='Normal IAQ Conditions',
                reset_period=RESET_PERIOD,
                show_plot=False
                )

    # Extreme Prediction 2 (ĐÃ SỬA: Map đúng biến cho kết quả EXT2)
    # Extreme Prediction 2 - Physical Sensor
    y_ext_ps_true_2, y_ext_ps_pred_2, _, _, _, _ = evaluation_mb_physical_sensor(
                                                                            model=trained_ext_mb_model, 
                                                                            regressor=trained_ext_mb_regressor, 
                                                                            device=device, 
                                                                            test_loader=ext_prediction_dataloader_2,
                                                                            scaler_outdoor=scaler_outdoor_ext,
                                                                            scaler_indoor=scaler_indoor_ext,
                                                                            result_file_name=csv_ext_physical_sensor_evaluation_results_file_name_2,
                                                                            save_path=RESULT_PATH
                                                                            )

    scatter_plot(   y_ext_ps_true_2, 
                    y_ext_ps_pred_2, 
                    file_name=scatter_plot_ext_physical_sensor_file_name_2,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Physical Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )

    line_plot(  y_ext_ps_true_2, 
                y_ext_ps_pred_2, 
                file_name=line_plot_ext_physical_sensor_file_name_2,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Physical Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction 2 - Virtual Sensor
    y_ext_vr_true_2, y_ext_vr_pred_2, _, _, _, _ = evaluation_mb_virtual_sensor(
                                                                            model=trained_ext_mb_model, 
                                                                            regressor=trained_ext_mb_regressor, 
                                                                            device=device, 
                                                                            test_loader=ext_prediction_dataloader_2, 
                                                                            scaler_outdoor=scaler_outdoor_ext,
                                                                            scaler_indoor=scaler_indoor_ext,
                                                                            result_file_name=csv_ext_virtual_sensor_evaluation_results_file_name_2,
                                                                            save_path=RESULT_PATH
                                                                            )                                   

    scatter_plot(   y_ext_vr_true_2, 
                    y_ext_vr_pred_2, 
                    file_name=scatter_plot_ext_virtual_sensor_file_name_2,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )
    
    line_plot(  y_ext_vr_true_2, 
                y_ext_vr_pred_2, 
                file_name=line_plot_ext_virtual_sensor_file_name_2,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction 2 - Virtual Sensor - Autoreset
    y_ext_vr_ar_true_2, y_ext_vr_ar_pred_2, _, _, _, _ = evaluation_mb_virtual_sensor_autoreset(
                                                                                            model=trained_ext_mb_model, 
                                                                                            regressor=trained_ext_mb_regressor, 
                                                                                            device=device, 
                                                                                            test_loader=ext_prediction_dataloader_2,
                                                                                            scaler_outdoor=scaler_outdoor_ext,
                                                                                            scaler_indoor=scaler_indoor_ext,
                                                                                            result_file_name=csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_2,
                                                                                            save_path=RESULT_PATH,
                                                                                            reset_period=RESET_PERIOD
                                                                                            )

    scatter_plot(   y_ext_vr_ar_true_2, 
                    y_ext_vr_ar_pred_2, 
                    file_name=scatter_plot_ext_virtual_sensor_autoreset_file_name_2,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=RESET_PERIOD,
                    show_plot=False
                    )

    line_plot(  y_ext_vr_ar_true_2, 
                y_ext_vr_ar_pred_2, 
                file_name=line_plot_ext_virtual_sensor_autoreset_file_name_2,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=RESET_PERIOD,
                show_plot=False
                )

    # Extreme Prediction 3 (ĐÃ SỬA: Map đúng biến cho kết quả EXT3)
    # Extreme Prediction 3 - Physical Sensor
    y_ext_ps_true_3, y_ext_ps_pred_3, _, _, _, _ = evaluation_mb_physical_sensor(
                                                                            model=trained_ext_mb_model, 
                                                                            regressor=trained_ext_mb_regressor, 
                                                                            device=device, 
                                                                            test_loader=ext_prediction_dataloader_3,
                                                                            scaler_outdoor=scaler_outdoor_ext,
                                                                            scaler_indoor=scaler_indoor_ext,
                                                                            result_file_name=csv_ext_physical_sensor_evaluation_results_file_name_3,
                                                                            save_path=RESULT_PATH
                                                                            )

    scatter_plot(   y_ext_ps_true_3, 
                    y_ext_ps_pred_3, 
                    file_name=scatter_plot_ext_physical_sensor_file_name_3,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Physical Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )

    line_plot(  y_ext_ps_true_3, 
                y_ext_ps_pred_3, 
                file_name=line_plot_ext_physical_sensor_file_name_3,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Physical Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction 3 - Virtual Sensor
    y_ext_vr_true_3, y_ext_vr_pred_3, _, _, _, _ = evaluation_mb_virtual_sensor(
                                                                            model=trained_ext_mb_model, 
                                                                            regressor=trained_ext_mb_regressor, 
                                                                            device=device, 
                                                                            test_loader=ext_prediction_dataloader_3, 
                                                                            scaler_outdoor=scaler_outdoor_ext,
                                                                            scaler_indoor=scaler_indoor_ext,
                                                                            result_file_name=csv_ext_virtual_sensor_evaluation_results_file_name_3,
                                                                            save_path=RESULT_PATH
                                                                            )                                   

    scatter_plot(   y_ext_vr_true_3, 
                    y_ext_vr_pred_3, 
                    file_name=scatter_plot_ext_virtual_sensor_file_name_3,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=None,
                    show_plot=False
                    )
    
    line_plot(  y_ext_vr_true_3, 
                y_ext_vr_pred_3, 
                file_name=line_plot_ext_virtual_sensor_file_name_3,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=None,
                show_plot=False
                )

    # Extreme Prediction 3 - Virtual Sensor - Autoreset
    y_ext_vr_ar_true_3, y_ext_vr_ar_pred_3, _, _, _, _ = evaluation_mb_virtual_sensor_autoreset(
                                                                                            model=trained_ext_mb_model, 
                                                                                            regressor=trained_ext_mb_regressor, 
                                                                                            device=device, 
                                                                                            test_loader=ext_prediction_dataloader_3,
                                                                                            scaler_outdoor=scaler_outdoor_ext,
                                                                                            scaler_indoor=scaler_indoor_ext,
                                                                                            result_file_name=csv_ext_virtual_sensor_autoreset_evaluation_results_file_name_3,
                                                                                            save_path=RESULT_PATH,
                                                                                            reset_period=RESET_PERIOD
                                                                                            )

    scatter_plot(   y_ext_vr_ar_true_3, 
                    y_ext_vr_ar_pred_3, 
                    file_name=scatter_plot_ext_virtual_sensor_autoreset_file_name_3,
                    save_path=RESULT_PATH,
                    model_mode='MB',
                    sensor_name='Virtual Sensor',
                    mode_name='Extreme IAQ Conditions',
                    reset_period=RESET_PERIOD,
                    show_plot=False
                    )

    line_plot(  y_ext_vr_ar_true_3, 
                y_ext_vr_ar_pred_3, 
                file_name=line_plot_ext_virtual_sensor_autoreset_file_name_3,
                save_path=RESULT_PATH,
                model_mode='MB',
                sensor_name='Virtual Sensor',
                mode_name='Extreme IAQ Conditions',
                reset_period=RESET_PERIOD,
                show_plot=False
                )

    clear_GPU_memory()

# Main function
if __name__ == "__main__":
    main()
