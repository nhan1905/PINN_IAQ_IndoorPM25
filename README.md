
![Task](https://img.shields.io/badge/Task-Indoor%20PM2.5%20Prediction-e9573f)
![Domain](https://img.shields.io/badge/Domain-Indoor%20Air%20Quality-00c853)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Architecture](https://img.shields.io/badge/Architecture-Physics--Informed%20BiLSTM-1785fb)
![Language](https://img.shields.io/badge/Language-Python%203.10-blueviolet)

<h1 align="center">
Indoor PM2.5 Prediction Using a Mass Balance-Informed Deep Learning Framework
</h1>

<div align="center" style="font-size: 14px;">
Lam Khang Nguyen Duy<sup>a,</sup>^, Minh T.N Nguyen<sup>a,c,</sup>^, Thanh H. Nguyen<sup>b</sup>, Dung D. Le<sup>c</sup>, Phong K. Thai<sup>d</sup>, Hung Hai Pham<sup>e</sup>, Vishal Verma<sup>b</sup>, Nhan Dinh Ngo<sup>a,b,</sup>*
</div>

<br>

<div align="center" style="font-size: 14px;">
<sup>a</sup>VinUni-Illinois Smart Health Center, VinUniversity, Hanoi 100000, Vietnam <br>
<sup>b</sup>Department of Civil & Environmental Engineering, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA <br>
<sup>c</sup>College of Engineering and Computer Science, VinUniversity, Hanoi 100000, Vietnam <br>
<sup>d</sup>Queensland Alliance for Environmental Health Sciences (QAEHS), The University of Queensland, Brisbane, Queensland, Australia <br>
<sup>e</sup>Center for Environmental Intelligence, VinUniversity, Hanoi 100000, Vietnam <br>
</div>

<br>

<div align="center" style="font-size: 14px;">
<strong>Graphic Abstract</strong>
</div>

<br>

<p align="center">
  <img src="Graphic Abstract/Graphic Abstract.png" alt="Graphic Abstract" width="850">
</p>

<br>

<div align="center" style="font-size: 14px;">
<strong>Abstract</strong>
</div>

<br>

<div align="justify">

Monitoring indoor PM2.5 is vital for effective indoor air quality (IAQ) management and public health protection. However, reference-grade monitors are costly, and low-cost sensors often require frequent recalibration. To address these limitations, this study proposes a novel mass balance-informed deep learning framework that integrates a conventional Bidirectional Long Short-Term Memory (BiLSTM) model with a mass balance equation and a RESET mechanism, enabling physics-informed estimation of indoor PM2.5 mass concentrations from indoor and outdoor environmental parameters. The proposed framework was validated using experimental datasets collected under both normal (unaltered) and extreme but controlled indoor pollution scenarios, including incense-burning and candle-burning activities. The results demonstrate that the framework maintained high predictive accuracy (R2 > 0.98) and robustness even under extremely polluted conditions. These findings highlight its practical application for smart building systems by providing a cost-effective and reliable solution for indoor PM2.5 monitoring, reducing dependence on physical PM2.5 sensors, and extending their operational lifespan. Furthermore, by enabling continuous monitoring of indoor PM2.5 concentrations, this approach supports timely and proactive interventions, thereby contributing to improved occupant health protection.

# 1. Introduction

This repository contains the source code, data templates, and instructions required to reproduce the results of our manuscript. We propose a mass balance-informed deep learning framework incorporating a Bidirectional Long Short-Term Memory (BiLSTM) encoder for predicting indoor PM2.5 concentrations under both normal and extreme indoor air quality (IAQ) conditions.

A central feature of the methodology is the **RESET** mechanism. Predictions are generated recursively within a five-step block and are then reinitialized from an observed indoor PM2.5 value at the beginning of the next evaluation block, limiting long-horizon error propagation.

# 2. File Explanation

- **`README.md`**: Provides the project overview, repository structure, environment requirements, data preparation steps, execution commands, and troubleshooting guidance.
- **`LICENSE`**: Specifies the terms under which the repository content may be used, modified, and redistributed.
- **`Graphic Abstract/Graphic Abstract.png`**: Contains the graphical abstract displayed at the beginning of this README.
- **`TRAIN_DATA_TEMPLATE.csv`**: Provides the required column layout for the training dataset.
- **`NOR_TEMPLATE.csv`**: Provides the required column layout for the independent test dataset collected under normal indoor air quality conditions.
- **`EXT1_TEMPLATE.csv`**: Provides the required column layout for the first extreme-condition test dataset.
- **`EXT2_TEMPLATE.csv`**: Provides the required column layout for the second extreme-condition test dataset.
- **`EXT3_TEMPLATE.csv`**: Provides the required column layout for the third extreme-condition test dataset.
- **`main_BiLSTM_Pure.py`**: Configures, trains, loads, evaluates, and visualizes the purely data-driven BiLSTM baseline across the normal and three extreme-condition datasets.
- **`main_BiLSTM_MB.py`**: Configures, trains, loads, evaluates, and visualizes the mass balance-informed BiLSTM model using the recursive five-step prediction and RESET workflow.
- **`model_design_layer.py`**: Defines the shared BiLSTM encoder, the prediction head for the pure baseline, and the multi-head regressor that estimates the mass-balance parameters AER, P, K, and S.
- **`mass_balance_equation.py`**: Implements the discretized indoor PM2.5 mass-balance update, physically bounded parameter transformations, and the data-plus-physics training loss.
- **`utils.py`**: Provides GPU-memory cleanup, VRAM-aware batch-size selection, sliding-window tensor and DataLoader construction, and checkpoint loading utilities.
- **`utils_train_model.py`**: Implements the training and validation loops for both the pure BiLSTM and mass balance-informed BiLSTM models, including learning-rate scheduling, gradient clipping, and best-checkpoint saving.
- **`utils_eval_model.py`**: Implements multi-step evaluation for both models, exports predictions and estimated physical parameters to CSV, and computes overall evaluation metrics.
- **`metrics.py`**: Computes MSE, RMSE, MAE, MAPE, and R2 after filtering invalid values.
- **`visualization.py`**: Provides the scatter-plot and time-series line-plot functions called by the two main scripts to compare observed and predicted indoor PM2.5 values.

# 3. Requirements for Re-implementation

To execute the models, ensure that the environment includes the following components:

- **Programming language**: Python 3.10.
- **Deep learning framework**: PyTorch with optional CUDA acceleration.
- **Hardware**: An NVIDIA GPU is recommended for efficient training; CPU execution is also supported but may be considerably slower.
- **Required libraries**: `pandas`, `numpy`, `scikit-learn`, and `torch`.

# 4. Data Preparation

Place the runtime datasets inside a folder named `PINN_DATA` in the repository root. The main scripts expect the following file names:

- `PINN_DATA/TRAIN_DATA.csv`
- `PINN_DATA/NOR.csv`
- `PINN_DATA/EXT1.csv`
- `PINN_DATA/EXT2.csv`
- `PINN_DATA/EXT3.csv`

The provided template files define the required schema. Populate each template with the corresponding measurements and save or rename it using the runtime file name listed above.

## 4.1 Required columns

Each CSV file must contain a `Datetime` column and the following variables:

- **Outdoor parameters**: `OD_PM25`, `OD_PM10`, `OD_TEMP`, `OD_RH`
- **Indoor auxiliary parameters**: `ID_TEMP`, `ID_RH`, `ID_CO2`
- **Prediction target**: `ID_PM25`

The current implementation assumes five-minute data because the mass-balance time step is fixed at `dt = 1/60` h.

# 5. Step-by-Step Re-implementation

The repository uses a modular implementation in which model definitions, physical equations, data utilities, training routines, evaluation routines, and visualization functions are separated into dedicated files.

Both entry-point scripts accept an optional integer `run_id`. The random seed is set to `42 + run_id`, and results are saved in run-specific subdirectories.

## 5.1 Running the pure data-driven model

To train and evaluate the purely data-driven BiLSTM baseline, run:

```bash
python main_BiLSTM_Pure.py 1
```

## 5.2 Running the mass balance-informed model

To train and evaluate the proposed model, run:

```bash
python main_BiLSTM_MB.py 1
```

Replace `1` with another integer to create a separate seeded run. Omitting the argument uses `run_id = 1`.

# 6. Output and Visualization

The scripts automatically create the following output directories:

- **`PINN_BEST_MODEL/run_<run_id>`**: Stores the best encoder and regressor checkpoint selected using validation loss.
- **`PINN_RESULT/.../run_<run_id>`**: Stores evaluation CSV files and the scatter and line plots generated for the normal and extreme-condition datasets.

For the mass balance-informed model, the evaluation CSV files also include the estimated AER, P, K, and S values. The pure baseline uses `NaN` placeholders for these physical parameters.

# 7. Troubleshooting

- **Missing `visualization.py`**: Both main scripts import `visualization.py`. Confirm that this file is present in the repository before running either entry point.
- **Missing data files**: Ensure that the five runtime CSV files use the exact names listed in Section 4 and are located inside `PINN_DATA`.
- **CUDA out of memory**: The code dynamically selects a batch size from 32 to 512 according to available VRAM. If memory errors persist, reduce the allowed batch-size candidates or the batch-size safety factor in `utils.py`.
- **Multiprocessing issues on Windows**: If a DataLoader worker error occurs, set `num_workers=0` in the calls to `create_3d_tensor`.
- **Model selection**: The current implementation saves the checkpoint with the lowest validation loss. It does not implement an early-stopping termination criterion, so all configured epochs are executed unless training is interrupted.

# 8. Conclusions

By embedding a discretized physical mass-balance update within the BiLSTM framework and applying periodic RESET initialization, the proposed approach limits recursive error propagation while retaining sensitivity to temporal variability under both normal and extreme indoor pollution conditions.

</div>
