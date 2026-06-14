
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

Monitoring of indoor PM2.5 is vital for effective indoor air quality (IAQ) management and public health control. However, reference-grade monitors are costly, and low-cost sensors (LCS) often require frequent re-calibration. This study aims to develop a mass balance-informed framework that indirectly predicts indoor PM2.5 mass concentrations using multiple environmental parameters. We proposed a novel mass balance-informed Bidirectional Long Short-Term Memory (MB-informed BiLSTM) model. To overcome error propagation in multi-step predicting, we introduced a novel RESET mechanism that periodically reinitializes predictions with observed data. This framework was validated across normal and extreme indoor air pollution scenarios. Our results demonstrate that the MB-informed BiLSTM framework equipped with the RESET feature maintained high predictive accuracy (R2 ≃ 0.99) even under extremely-polluted conditions. Our approach has significant implications for smart building systems, such as Digital Twin platforms, by enabling a cost-effective and reliable solution for indoor PM2.5 monitoring. Furthermore, it supports “duty-cycling” strategies to extend low-cost sensor longevity, thereby enabling proactive, health-protective interventions by more accurate forecasting of indoor pollutants.

# 1. Introduction

This repository contains the source codes and detailed instructions for successfully reproducing the results of our manuscript. We propose a mass balance-informed deep learning framework incorporating Bidirectional Long Short-Term Memory (BiLSTM) for predicting indoor PM2.5 concentrations under both normal and extreme indoor air quality (IAQ) conditions.

A novel feature of our methodology is the introduction of a **RESET** mechanism, designed to reduce error propagation during extended recursive algorithmic predictions.

# 2. Requirements for Re-implementation

To successfully execute the models, please ensure your environment meets the following specifications utilized in our study:

- **Programming Language**: Python 3.10.
- **Deep Learning Framework**: PyTorch framework with CUDA 13.0 acceleration, version 2.10.0+cu130.
- **Hardware**: We recommend a workstation equipped with an NVIDIA GPU, such as NVIDIA H100 80GB HBM3, to handle model training efficiently.
- **Required Libraries**: `pandas`, `numpy`, `scikit-learn`, `torch`.

# 3. Data Preparation

Your quantitative datasets must be formatted as `.csv` files and placed inside a folder named `PINN_DATA` in the root directory.

Ensure you have the following files prepared as referenced in the code:

- `TRAIN_DATA_TEMPLATE.csv`
- `NOR_TEMPLATE.csv`
- `EXT1_TEMPLATE.csv`
- `EXT2_TEMPLATE.csv`
- `EXT3_TEMPLATE.csv`

**Data Structure**:

The CSV files must contain a `Datetime` column and the following parameters:

- **Outdoor Parameters**: `OD_PM25`, `OD_PM10`, `OD_TEMP`, `OD_RH`
- **Indoor Parameters**: `ID_TEMP`, `ID_RH`, `ID_CO2`
- **Target Parameter**: `ID_PM25`

# 4. Step-by-Step Re-implementation

The repository is structured using an Object-Oriented Programming (OOP) methodology. Core logic is modularized into files such as `model_design_layer.py` and `loss_function.py`.

## 4.1 Running the Pure Data-Driven Model

To evaluate the baseline model, namely the purely data-driven BiLSTM without mass balance constraints, execute the following command:

```bash
python main_BiLSTM_Pure.py
````

## 4.2 Running the Mass Balance-Informed Model

To train and evaluate the physics-informed model incorporating the discretized mass balance equation, execute:

```bash
python main_BiLSTM_MB.py
```

# 5. Output and Visualization

Once the runs are complete, the scripts will automatically generate directories to store your outputs:

* **`PINN_BEST_MODEL`**: Stores the trained `.pth` model weights.
* **`PINN_RESULT`**: Stores the evaluation `.csv` files and scatter/line plots comparing actual and predicted values for both Physical Sensor and Virtual Sensor, or Algorithmic, modes.

For further statistical analysis and custom plotting, you may utilize software such as OriginLab Pro 2024b, which was used for data visualization in our study.

# 6. Troubleshooting

* **CUDA Out of Memory**: If you run into memory issues during the `create_3d_tensor` steps, try reducing the `batch_size` or clearing your GPU cache. The code includes a `clear_GPU_memory()` function that runs automatically, but local environments may vary.
* **Missing File Errors**: Ensure all five `.csv` data files are strictly named and located within the `PINN_DATA` directory.
* **Overfitting**: Early stopping based on validation loss is applied to prevent overfitting. If you customize the `TRAINING_EPOCHS`, monitor the output logs carefully.

# 7. Conclusions

By embedding physical mass balance constraints and a periodic RESET mechanism into the BiLSTM framework, this approach successfully limits recursive error propagation and allows the model to maintain sensitivity to temporal variability under extreme indoor pollution levels.

</div>
```
