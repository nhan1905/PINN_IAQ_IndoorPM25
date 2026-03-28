![Task](https://img.shields.io/badge/Task-Indoor%20PM2.5%20Prediction-e9573f)
![Domain](https://img.shields.io/badge/Domain-Indoor%20Air%20Quality-00c853)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Architecture](https://img.shields.io/badge/Architecture-Physics--Informed%20BiLSTM-1785fb)
![Language](https://img.shields.io/badge/Language-Python%203.10-blueviolet)

Hi, welcome to our manuscript repository!

<h1 align="center">
Indoor PM2.5 Prediction Using a Mass Balance-Informed Deep Learning Framework
</h1>
<div align="center">

</div>
<div align="center" style="font-size: 14px;">
Lam Khang Nguyen Duy*<sup>a</sup>, Minh Nguyen Tran Nhat*<sup>a,c</sup>, Thanh H. Nguyen<sup>b</sup>, Dung D. Le<sup>c</sup>, Phong Thai Khanh<sup>d</sup>, Hung Hai Pham<sup>e</sup>, Nhan Dinh Ngo<sup>a,b,*</sup>, Vishal Verma<sup>b</sup> [cite: 2]
</div>
<br>

[cite_start]<sup>a</sup>VinUni-Illinois Smart Health Center, VinUniversity, Hanoi 100000, Vietnam [cite: 3]<br>
[cite_start]<sup>b</sup>Department of Civil & Environmental Engineering, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA [cite: 4]<br>
[cite_start]<sup>c</sup>College of Engineering and Computer Science, VinUniversity, Hanoi 100000, Vietnam [cite: 5]<br>
[cite_start]<sup>d</sup>Queensland Alliance for Environmental Health Sciences (QAEHS), The University of Queensland, Brisbane, Queensland, Australia [cite: 6]<br>
[cite_start]<sup>e</sup>Center for Environmental Intelligence, VinUniversity, Hanoi 100000, Vietnam [cite: 7]<br>
  

# 1. Introduction
[cite_start]This repository contains the source codes and detailed instructions for successfully reproducing the results of our manuscript[cite: 388]. [cite_start]We propose a mass balance-informed deep learning framework (incorporating Bidirectional LSTM) for predicting indoor PM2.5 concentrations under both normal and extreme indoor air quality (IAQ) conditions[cite: 67, 140]. 



[cite_start]A novel feature of our methodology is the introduction of a "RESET" mechanism, designed to reduce error propagation during extended recursive algorithmic predictions[cite: 69, 220].

# 2. Requirements for Re-implementation
To successfully execute the models, please ensure your environment meets the following specifications utilized in our study:
- [cite_start]**Programming Language**: Python (version 3.10)[cite: 380].
- [cite_start]**Deep Learning Framework**: PyTorch framework with CUDA 13.0 acceleration (version 2.10.0+cu130)[cite: 380].
- [cite_start]**Hardware**: We recommend a workstation equipped with an NVIDIA GPU (e.g., NVIDIA H100 80GB HBM3 or equivalent) to handle the model training efficiently[cite: 387]. 
- **Required Libraries**: `pandas`, `numpy`, `scikit-learn`, `torch`.

# 3. Data Preparation
Your quantitative datasets must be formatted as `.CSV` files and placed inside a folder named `PINN_DATA` in the root directory.
Ensure you have the following files prepared as referenced in the code:
- `TRAIN_DATA.csv`
- `NOR.csv`
- `EXT1.csv`
- `EXT2.csv`
- `EXT3.csv`

**Data Structure**:
[cite_start]The CSV files must contain a `Datetime` column and the following parameters[cite: 129, 131]:
- **Outdoor Parameters**: `OD_PM25`, `OD_PM10`, `OD_PM1`, `OD_TEMP`, `OD_RH`
- **Indoor Parameters**: `ID_TEMP`, `ID_RH`, `ID_CO2`
- **Target Parameter**: `ID_PM25`

# 4. Step-by-Step Re-implementation

The repository is structured using an Object-Oriented Programming (OOP) methodology. Core logic is modularized into files like `model_design_layer.py` and `loss_function.py`. 

## 4.1 Running the Pure Data-Driven Model
To evaluate the baseline (purely data-driven BiLSTM without mass balance constraints), execute the following command:

```bash
python main_BiLSTM_Pure.py 1
```
[cite_start]*(Note: `1` represents the `run_id`, which sets the random seed for reproducibility[cite: 384]. You can change this integer to test different seeds).*

## 4.2 Running the Mass Balance-Informed Model
[cite_start]To train and evaluate the physics-informed model incorporating the discretized mass balance equation[cite: 151, 157]:

```bash
python main_BiLSTM_MB.py 1
```

## 4.3 Running the Cross-Extreme Fine-Tuned Models
[cite_start]To evaluate the model's ability to generalize across diverse high-concentration scenarios, we implemented a cross fine-tuning step[cite: 117, 120]. You can run models fine-tuned on specific extreme datasets:

```bash
# Fine-tune on Extreme Scenario 1, test on 2 & 3
python main_BiLSTM_MB_CrossEXT1.py 1

# Fine-tune on Extreme Scenario 2, test on 1 & 3
python main_BiLSTM_MB_CrossEXT2.py 1

# Fine-tune on Extreme Scenario 3, test on 1 & 2
python main_BiLSTM_MB_CrossEXT3.py 1
```

# 5. Output and Visualization
Once the runs are complete, the scripts will automatically generate directories to store your outputs:
- **`PINN_BEST_MODEL_sustain_check/run_<id>/`**: Stores the trained `.pth` model weights.
- **`PINN_RESULT_sustain_check/run_<id>/`**: Stores the evaluation `.csv` files and scatter/line plots comparing actual vs. predicted values for both Physical Sensor and Virtual Sensor (Algorithmic) modes.

[cite_start]*Note: For further statistical analysis and custom plotting, you may utilize software such as OriginLab Pro 2024b, which was used for data visualization in our study[cite: 377].*

# 6. Troubleshooting
- **CUDA Out of Memory**: If you run into memory issues during the `create_3d_tensor` steps, try reducing the `batch_size` or clearing your GPU cache. The code includes a `clear_GPU_memory()` function that runs automatically, but local environments may vary.
- **Missing File Errors**: Ensure all 5 `.csv` data files are strictly named and located within the `PINN_DATA` directory.
- [cite_start]**Overfitting**: Early stopping based on validation loss is applied to prevent overfitting[cite: 385]. If you customize the `TRAINING_EPOCHS`, monitor the output logs carefully.

# 7. Conclusions
[cite_start]By embedding physical mass balance constraints and a periodic RESET mechanism into the BiLSTM framework, this approach successfully limits recursive error propagation and allows the model to maintain sensitivity to temporal variability under extreme indoor pollution levels[cite: 647, 779].
```
