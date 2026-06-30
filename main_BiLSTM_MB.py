# Author: Nguyen Tran Nhat Minh
import os
import sys
import random
import torch
import pandas as pd
import numpy as np
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

WINDOW_SIZE = 45
PREDICTION_STEPS = WINDOW_SIZE

TRAINING_EPOCHS = 100
LEARNING_RATE = 1e-4

LAMBDA_TEACHER = 0.2
LAMBDA_TREND = 0.1
LAMBDA_BIAS = 0.5
HIGH_PM_THRESHOLD = 50.0
HIGH_PM_WEIGHT = 1.0
HORIZON_MAX_WEIGHT = 2.0
EARLY_STOPPING_PATIENCE = 10

project_dir = os.path.dirname(os.path.abspath(__file__))

DATA_PATH  = os.path.join(project_dir, 'PINN_DATA')
MODEL_PATH = os.path.join(project_dir, 'PINN_BEST_MODEL', f'run_{run_id}')
RESULT_PATH = os.path.join(project_dir, 'PINN_RESULT', f'MB_MODEL_EXT_TRAIN_RESULTS_RESET_PERIOD_{PREDICTION_STEPS}', f'run_{run_id}')

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

train_data_file_name            = 'TRAIN_DATA.csv'
nor_prediction_data_file_name   = 'NOR.csv'
ext_prediction_data_file_name_1 = 'EXT1.csv'
ext_prediction_data_file_name_2 = 'EXT2.csv'
ext_prediction_data_file_name_3 = 'EXT3.csv'

# Base Model name
base_model_file_name = 'BiLSTM' 

# Save file name
mb_trained_model_file_name = f'{base_model_file_name}_MB_Trained_Model_{PREDICTION_STEPS}.pth' 

# Naming Conventions for Physical Prediction Mode
csv_nor_physical_sensor_evaluation_results_file_name = f'{base_model_file_name}_MB_NOR_Physical_Sensor_Evaluation_Results.csv'
scatter_plot_nor_physical_sensor_file_name = f'{base_model_file_name}_MB_Scatter_Plot_NOR_Physical_Sensor'
line_plot_nor_physical_sensor_file_name = f'{base_model_file_name}_MB_Line_Plot_NOR_Physical_Sensor'

csv_ext_physical_sensor_evaluation_results_file_name_1 = f'{base_model_file_name}_MB_EXT_Physical_Sensor_Evaluation_Results_1.csv'
scatter_plot_ext_physical_sensor_file_name_1 = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Physical_Sensor_1'
line_plot_ext_physical_sensor_file_name_1 = f'{base_model_file_name}_MB_Line_Plot_EXT_Physical_Sensor_1'

csv_ext_physical_sensor_evaluation_results_file_name_2 = f'{base_model_file_name}_MB_EXT_Physical_Sensor_Evaluation_Results_2.csv'
scatter_plot_ext_physical_sensor_file_name_2 = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Physical_Sensor_2'
line_plot_ext_physical_sensor_file_name_2 = f'{base_model_file_name}_MB_Line_Plot_EXT_Physical_Sensor_2'

csv_ext_physical_sensor_evaluation_results_file_name_3 = f'{base_model_file_name}_MB_EXT_Physical_Sensor_Evaluation_Results_3.csv'
scatter_plot_ext_physical_sensor_file_name_3 = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Physical_Sensor_3'
line_plot_ext_physical_sensor_file_name_3 = f'{base_model_file_name}_MB_Line_Plot_EXT_Physical_Sensor_3'

# Naming Conventions for Virtual Prediction Mode
csv_nor_virtual_sensor_evaluation_results_file_name = f'{base_model_file_name}_MB_NOR_Virtual_Sensor_Evaluation_Results.csv'
scatter_plot_nor_virtual_sensor_file_name = f'{base_model_file_name}_MB_Scatter_Plot_NOR_Virtual_Sensor'
line_plot_nor_virtual_sensor_file_name = f'{base_model_file_name}_MB_Line_Plot_NOR_Virtual_Sensor'

csv_ext_virtual_sensor_evaluation_results_file_name_1 = f'{base_model_file_name}_MB_EXT_Virtual_Sensor_Evaluation_Results_1.csv'
scatter_plot_ext_virtual_sensor_file_name_1 = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Virtual_Sensor_1'
line_plot_ext_virtual_sensor_file_name_1 = f'{base_model_file_name}_MB_Line_Plot_EXT_Virtual_Sensor_1'

csv_ext_virtual_sensor_evaluation_results_file_name_2 = f'{base_model_file_name}_MB_EXT_Virtual_Sensor_Evaluation_Results_2.csv'
scatter_plot_ext_virtual_sensor_file_name_2 = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Virtual_Sensor_2'
line_plot_ext_virtual_sensor_file_name_2 = f'{base_model_file_name}_MB_Line_Plot_EXT_Virtual_Sensor_2'

csv_ext_virtual_sensor_evaluation_results_file_name_3 = f'{base_model_file_name}_MB_EXT_Virtual_Sensor_Evaluation_Results_3.csv'
scatter_plot_ext_virtual_sensor_file_name_3 = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Virtual_Sensor_3'
line_plot_ext_virtual_sensor_file_name_3 = f'{base_model_file_name}_MB_Line_Plot_EXT_Virtual_Sensor_3'

# Naming Conventions for Virtual Prediction Mode with RESET
csv_nor_virtual_sensor_evaluation_results_file_name_ar = f'{base_model_file_name}_MB_NOR_Virtual_Sensor_Evaluation_Results_AR.csv'
scatter_plot_nor_virtual_sensor_file_name_ar = f'{base_model_file_name}_MB_Scatter_Plot_NOR_Virtual_Sensor_AR'
line_plot_nor_virtual_sensor_file_name_ar = f'{base_model_file_name}_MB_Line_Plot_NOR_Virtual_Sensor_AR'

csv_ext_virtual_sensor_evaluation_results_file_name_1_ar = f'{base_model_file_name}_MB_EXT_Virtual_Sensor_Evaluation_Results_1_AR.csv'
scatter_plot_ext_virtual_sensor_file_name_1_ar = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Virtual_Sensor_1_AR'
line_plot_ext_virtual_sensor_file_name_1_ar = f'{base_model_file_name}_MB_Line_Plot_EXT_Virtual_Sensor_1_AR'

csv_ext_virtual_sensor_evaluation_results_file_name_2_ar = f'{base_model_file_name}_MB_EXT_Virtual_Sensor_Evaluation_Results_2_AR.csv'
scatter_plot_ext_virtual_sensor_file_name_2_ar = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Virtual_Sensor_2_AR'
line_plot_ext_virtual_sensor_file_name_2_ar = f'{base_model_file_name}_MB_Line_Plot_EXT_Virtual_Sensor_2_AR'

csv_ext_virtual_sensor_evaluation_results_file_name_3_ar = f'{base_model_file_name}_MB_EXT_Virtual_Sensor_Evaluation_Results_3_AR.csv'
scatter_plot_ext_virtual_sensor_file_name_3_ar = f'{base_model_file_name}_MB_Scatter_Plot_EXT_Virtual_Sensor_3_AR'
line_plot_ext_virtual_sensor_file_name_3_ar = f'{base_model_file_name}_MB_Line_Plot_EXT_Virtual_Sensor_3_AR'


TRAIN_DATA_PATH            = os.path.join(DATA_PATH, train_data_file_name)
PREDICTION_NOR_DATA_PATH   = os.path.join(DATA_PATH, nor_prediction_data_file_name)
PREDICTION_EXT_DATA_PATH_1 = os.path.join(DATA_PATH, ext_prediction_data_file_name_1)
PREDICTION_EXT_DATA_PATH_2 = os.path.join(DATA_PATH, ext_prediction_data_file_name_2)
PREDICTION_EXT_DATA_PATH_3 = os.path.join(DATA_PATH, ext_prediction_data_file_name_3)

# SENSOR DATA STRUCTURE
TIMESTAMP_COLUMN = 'Datetime'
OUTDOOR_COLS = ['OD_PM25', 'OD_PM10', 'OD_TEMP', 'OD_RH']
INDOOR_AUX_COLS = ['ID_TEMP', 'ID_RH', 'ID_CO2']

FEATURE_COLUMNS = OUTDOOR_COLS + INDOOR_AUX_COLS
TARGET_COLUMNS = ['ID_PM25']

# MODEL PARAMETERS
PAST_DIM = len(FEATURE_COLUMNS) + len(TARGET_COLUMNS)
CURRENT_DIM = len(FEATURE_COLUMNS)
HIDDEN_DIM = 512

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 1 - TRAIN_RATIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

    # Convert to Numpy
    np_train_feature = df_train_feature.values
    np_train_target = df_train_target.values
    np_nor_prediction_feature = df_nor_prediction_feature.values
    np_nor_prediction_target = df_nor_prediction_target.values
    
    np_ext_prediction_feature_1 = df_ext_prediction_feature_1.values
    np_ext_prediction_target_1 = df_ext_prediction_target_1.values
    
    np_ext_prediction_feature_2 = df_ext_prediction_feature_2.values 
    np_ext_prediction_target_2 = df_ext_prediction_target_2.values 
    
    np_ext_prediction_feature_3 = df_ext_prediction_feature_3.values 
    np_ext_prediction_target_3 = df_ext_prediction_target_3.values 

    train_split_idx = int(len(df_train_target) * TRAIN_RATIO)
    np_train_train_feature = np_train_feature[: train_split_idx, :]
    np_train_train_target = np_train_target[: train_split_idx]
    np_train_val_feature = np_train_feature[train_split_idx :, :]
    np_train_val_target = np_train_target[train_split_idx :]

    context_mean = np_train_train_feature.mean(axis=0)
    context_std = np_train_train_feature.std(axis=0)
    context_std[context_std < 1e-6] = 1.0

    indoor_mean = np_train_train_target.mean(axis=0)
    indoor_std = np_train_train_target.std(axis=0)
    indoor_std[indoor_std < 1e-6] = 1.0

    # Create dataloaders
    train_nor_dataloader = create_3d_tensor(
                                            data_feature=np_train_train_feature, 
                                            data_target=np_train_train_target, 
                                            input_size=PAST_DIM, 
                                            window_size=WINDOW_SIZE,
                                            pred_steps=PREDICTION_STEPS,
                                            stride=1,
                                            num_workers=2, 
                                            mode='training'
                                            )
    
    val_nor_dataloader = create_3d_tensor(
                                            data_feature=np_train_val_feature, 
                                            data_target=np_train_val_target, 
                                            input_size=PAST_DIM, 
                                            window_size=WINDOW_SIZE,
                                            pred_steps=PREDICTION_STEPS,
                                            stride=PREDICTION_STEPS,
                                            num_workers=2, 
                                            mode='evaluation'
                                            )

    nor_prediction_dataloader = create_3d_tensor(
                                                data_feature=np_nor_prediction_feature, 
                                                data_target=np_nor_prediction_target, 
                                                input_size=PAST_DIM, 
                                                window_size=WINDOW_SIZE,
                                                pred_steps=1,
                                                stride=1,
                                                num_workers=2, 
                                                mode='evaluation'
                                                )

    nor_prediction_dataloader_ar = create_3d_tensor(
                                                data_feature=np_nor_prediction_feature, 
                                                data_target=np_nor_prediction_target, 
                                                input_size=PAST_DIM, 
                                                window_size=WINDOW_SIZE,
                                                pred_steps=PREDICTION_STEPS,
                                                stride=PREDICTION_STEPS,
                                                num_workers=2, 
                                                mode='evaluation'
                                                )

    ext_prediction_dataloader_1 = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_1, 
                                                    data_target=np_ext_prediction_target_1, 
                                                    input_size=PAST_DIM, 
                                                    window_size=WINDOW_SIZE,
                                                    pred_steps=1,
                                                    stride=1,
                                                    num_workers=2,
                                                    mode='evaluation'
                                                    )

    ext_prediction_dataloader_1_ar = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_1, 
                                                    data_target=np_ext_prediction_target_1, 
                                                    input_size=PAST_DIM, 
                                                    window_size=WINDOW_SIZE,
                                                    pred_steps=PREDICTION_STEPS,
                                                    stride=PREDICTION_STEPS,
                                                    num_workers=2,
                                                    mode='evaluation'
                                                    )                                                    

    ext_prediction_dataloader_2 = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_2, 
                                                    data_target=np_ext_prediction_target_2, 
                                                    input_size=PAST_DIM, 
                                                    window_size=WINDOW_SIZE,
                                                    pred_steps=1,
                                                    stride=1,
                                                    num_workers=2,
                                                    mode='evaluation'
                                                    )

    ext_prediction_dataloader_2_ar = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_2, 
                                                    data_target=np_ext_prediction_target_2, 
                                                    input_size=PAST_DIM, 
                                                    window_size=WINDOW_SIZE,
                                                    pred_steps=PREDICTION_STEPS,
                                                    stride=PREDICTION_STEPS,
                                                    num_workers=2,
                                                    mode='evaluation'
                                                    )                                                    

    ext_prediction_dataloader_3 = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_3, 
                                                    data_target=np_ext_prediction_target_3, 
                                                    input_size=PAST_DIM, 
                                                    window_size=WINDOW_SIZE,
                                                    pred_steps=1,
                                                    stride=1,
                                                    num_workers=2, 
                                                    mode='evaluation'
                                                    )

    ext_prediction_dataloader_3_ar = create_3d_tensor(
                                                    data_feature=np_ext_prediction_feature_3, 
                                                    data_target=np_ext_prediction_target_3, 
                                                    input_size=PAST_DIM, 
                                                    window_size=WINDOW_SIZE,
                                                    pred_steps=PREDICTION_STEPS,
                                                    stride=PREDICTION_STEPS,
                                                    num_workers=2, 
                                                    mode='evaluation'
                                                    )                                                   

    # Initialize and train BiLSTM model
    base_model = BiLSTM(
                        past_dim=PAST_DIM, 
                        hidden_dim=HIDDEN_DIM, 
                        seq_len=WINDOW_SIZE
                        ).to(device)

    base_regressor = MultiHeadRegressor(
                                        current_dim=CURRENT_DIM,
                                        latent_dim=HIDDEN_DIM * 2,
                                        context_mean=context_mean,
                                        context_std=context_std,
                                        indoor_mean=indoor_mean,
                                        indoor_std=indoor_std
                                        ).to(device)

    print(f"--- Base Model and Regressor initialized ---")

    print("\n--- TRAINING >> STARTING... ---")
    _, _ = training_mb_model(
                                model=base_model,
                                regressor=base_regressor,
                                device=device,
                                train_loader=train_nor_dataloader,
                                val_loader=val_nor_dataloader,
                                learning_rate=LEARNING_RATE,
                                epochs=TRAINING_EPOCHS,
                                model_file_name=mb_trained_model_file_name,
                                save_path=MODEL_PATH,
                                lambda_teacher=LAMBDA_TEACHER,
                                lambda_trend=LAMBDA_TREND,
                                lambda_bias=LAMBDA_BIAS,
                                high_pm_threshold=HIGH_PM_THRESHOLD,
                                high_pm_weight=HIGH_PM_WEIGHT,
                                horizon_max_weight=HORIZON_MAX_WEIGHT,
                                early_stopping_patience=EARLY_STOPPING_PATIENCE
                            )
    print("--- TRAINING >> FINISHED --- \n")
    #--------------------------------------

    trained_mb_model, trained_mb_regressor = load_model(
                                                    model=base_model, 
                                                    regressor=base_regressor, 
                                                    device=device, 
                                                    model_name=mb_trained_model_file_name, 
                                                    save_path=MODEL_PATH
                                                    )
    
    print("\n--- EVALUATION >> START ---")
    evaluation_datasets = {
        "NORMAL_PS": {
            "loader": nor_prediction_dataloader,
            "csv_results": csv_nor_physical_sensor_evaluation_results_file_name,
            "scatter_plot": scatter_plot_nor_physical_sensor_file_name,
            "line_plot": line_plot_nor_physical_sensor_file_name,
            "mode_name": "Normal IAQ Conditions"
        },
        "NORMAL_VR": {
            "loader": nor_prediction_dataloader,
            "csv_results": csv_nor_virtual_sensor_evaluation_results_file_name,
            "scatter_plot": scatter_plot_nor_virtual_sensor_file_name,
            "line_plot": line_plot_nor_virtual_sensor_file_name,
            "mode_name": "Normal IAQ Conditions"
        },
        "NORMAL_VR_AR": {
            "loader": nor_prediction_dataloader_ar,
            "csv_results": csv_nor_virtual_sensor_evaluation_results_file_name_ar,
            "scatter_plot": scatter_plot_nor_virtual_sensor_file_name_ar,
            "line_plot": line_plot_nor_virtual_sensor_file_name_ar,
            "mode_name": "Normal IAQ Conditions"
        },
        "EXT1_PS": {
            "loader": ext_prediction_dataloader_1,
            "csv_results": csv_ext_physical_sensor_evaluation_results_file_name_1,
            "scatter_plot": scatter_plot_ext_physical_sensor_file_name_1,
            "line_plot": line_plot_ext_physical_sensor_file_name_1,
            "mode_name": "Extreme IAQ Conditions 1"
        },
        "EXT1_VR": {
            "loader": ext_prediction_dataloader_1,
            "csv_results": csv_ext_virtual_sensor_evaluation_results_file_name_1,
            "scatter_plot": scatter_plot_ext_virtual_sensor_file_name_1,
            "line_plot": line_plot_ext_virtual_sensor_file_name_1,
            "mode_name": "Extreme IAQ Conditions 1"
        },
        "EXT1_VR_AR": {
            "loader": ext_prediction_dataloader_1_ar,
            "csv_results": csv_ext_virtual_sensor_evaluation_results_file_name_1_ar,
            "scatter_plot": scatter_plot_ext_virtual_sensor_file_name_1_ar,
            "line_plot": line_plot_ext_virtual_sensor_file_name_1_ar,
            "mode_name": "Extreme IAQ Conditions 1"
        },
        "EXT2_PS": {
            "loader": ext_prediction_dataloader_2,
            "csv_results": csv_ext_physical_sensor_evaluation_results_file_name_2,
            "scatter_plot": scatter_plot_ext_physical_sensor_file_name_2,
            "line_plot": line_plot_ext_physical_sensor_file_name_2,
            "mode_name": "Extreme IAQ Conditions 2"
        },
        "EXT2_VR": {
            "loader": ext_prediction_dataloader_2,
            "csv_results": csv_ext_virtual_sensor_evaluation_results_file_name_2,
            "scatter_plot": scatter_plot_ext_virtual_sensor_file_name_2,
            "line_plot": line_plot_ext_virtual_sensor_file_name_2,
            "mode_name": "Extreme IAQ Conditions 2"
        },
        "EXT2_VR_AR": {
            "loader": ext_prediction_dataloader_2_ar,
            "csv_results": csv_ext_virtual_sensor_evaluation_results_file_name_2_ar,
            "scatter_plot": scatter_plot_ext_virtual_sensor_file_name_2_ar,
            "line_plot": line_plot_ext_virtual_sensor_file_name_2_ar,
            "mode_name": "Extreme IAQ Conditions 2"
        },
        "EXT3_PS": {
            "loader": ext_prediction_dataloader_3,
            "csv_results": csv_ext_physical_sensor_evaluation_results_file_name_3,
            "scatter_plot": scatter_plot_ext_physical_sensor_file_name_3,
            "line_plot": line_plot_ext_physical_sensor_file_name_3,
            "mode_name": "Extreme IAQ Conditions 3"
        },
        "EXT3_VR": {
            "loader": ext_prediction_dataloader_3,
            "csv_results": csv_ext_virtual_sensor_evaluation_results_file_name_3,
            "scatter_plot": scatter_plot_ext_virtual_sensor_file_name_3,
            "line_plot": line_plot_ext_virtual_sensor_file_name_3,
            "mode_name": "Extreme IAQ Conditions 3"
        },
        "EXT3_VR_AR": {
            "loader": ext_prediction_dataloader_3_ar,
            "csv_results": csv_ext_virtual_sensor_evaluation_results_file_name_3_ar,
            "scatter_plot": scatter_plot_ext_virtual_sensor_file_name_3_ar,
            "line_plot": line_plot_ext_virtual_sensor_file_name_3_ar,
            "mode_name": "Extreme IAQ Conditions 3"
        }
    }

    # for name, data in evaluation_datasets.items():
    # for name in ["NORMAL_PS", "EXT1_PS", "EXT2_PS", "EXT3_PS"]:
    #     data = evaluation_datasets[name]

    #     print(f"\n=== MODEL EVALUATION (PHYSICAL ESTIMATION MODE) OF DATASET >> {name} ===")
    #     y_true, y_pred, _, _, _, _ = evaluation_mb_physical_sensor(
    #         model=trained_mb_model,
    #         regressor=trained_mb_regressor,
    #         device=device,
    #         test_loader=data["loader"],
    #         result_file_name=data["csv_results"],
    #         save_path=RESULT_PATH
    #     )

    #     scatter_plot(
    #         y_true, y_pred,
    #         file_name=data["scatter_plot"],
    #         save_path=RESULT_PATH,
    #         model_mode='MB',
    #         sensor_name='Physical Sensor',
    #         mode_name=data["mode_name"],
    #         show_plot=False
    #     )

    #     line_plot(
    #         y_true, y_pred,
    #         file_name=data["line_plot"],
    #         save_path=RESULT_PATH,
    #         model_mode='MB',
    #         sensor_name='Physical Sensor',
    #         mode_name=data["mode_name"],
    #         show_plot=False
    #     )

    # for name in ["NORMAL_VR", "EXT1_VR", "EXT2_VR", "EXT3_VR"]:
    #     data = evaluation_datasets[name]
    #     print(f"\n=== MODEL EVALUATION (VIRUAL ESTIMATION MODE) OF DATASET >> {name} ===")

    #     y_true, y_pred, _, _, _, _ = evaluation_mb_virtual_sensor(
    #         model=trained_mb_model,
    #         regressor=trained_mb_regressor,
    #         device=device,
    #         test_loader=data["loader"],
    #         result_file_name=data["csv_results"],
    #         save_path=RESULT_PATH
    #     )

    #     scatter_plot(
    #         y_true, y_pred,
    #         file_name=data["scatter_plot"],
    #         save_path=RESULT_PATH,
    #         model_mode='MB',
    #         sensor_name='Virtual Sensor',
    #         mode_name=data["mode_name"],
    #         show_plot=False
    #     )

    #     line_plot(
    #         y_true, y_pred,
    #         file_name=data["line_plot"],
    #         save_path=RESULT_PATH,
    #         model_mode='MB',
    #         sensor_name='Virtual Sensor',
    #         mode_name=data["mode_name"],
    #         show_plot=False
    #     )
    
    for name in ["NORMAL_VR_AR", "EXT1_VR_AR", "EXT2_VR_AR", "EXT3_VR_AR"]:
        data = evaluation_datasets[name]
        print(f"\n=== MODEL EVALUATION (VIRUAL ESTIMATION MODE & RESET) OF DATASET >> {name} ===")

        y_true, y_pred, _, _, _, _ = evaluation_mb_virtual_sensor_autoreset(
            model=trained_mb_model,
            regressor=trained_mb_regressor,
            device=device,
            test_loader=data["loader"],
            result_file_name=data["csv_results"],
            save_path=RESULT_PATH,
            reset_period = PREDICTION_STEPS
        )

        scatter_plot(
            y_true, y_pred,
            file_name=data["scatter_plot"],
            save_path=RESULT_PATH,
            model_mode='MB',
            sensor_name='Virtual Sensor & RESET',
            mode_name=data["mode_name"],
            show_plot=False
        )

        line_plot(
            y_true, y_pred,
            file_name=data["line_plot"],
            save_path=RESULT_PATH,
            model_mode='MB',
            sensor_name='Virtual Sensor & RESET',
            mode_name=data["mode_name"],
            pred_steps=PREDICTION_STEPS,
            show_plot=False
        )
    
    print("\n--- EVALUATION >> FINISHED ---")

    clear_GPU_memory()

if __name__ == "__main__":
    main()
