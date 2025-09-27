## Communication-Efficient Multimodal Federated Learning: Joint Modality and Client Selection

[![python](https://img.shields.io/badge/Python_3.10-306998?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/release/python-31012/)
[![License: MIT](https://img.shields.io/badge/license-MIT-750014.svg)](https://opensource.org/licenses/MIT) 
[![arXiv](https://img.shields.io/badge/arXiv-2401.16685-b31b1b.svg)](https://arxiv.org/abs/2401.16685)


## 🔥 Our Framework

TL, DR: In this repo, we provide the implementation of **multimodal Federated learning with joint Modality and Client selection** (MFedMC), a novel methodology for multimodal federated learning.

<div align="center">
    <img src="figures/MFedMC.png" alt="overview" style="width:60%;"/>
</div>


## 🖥️ Prerequisites

Install the required packages via:
```bash
pip install -r requirements.txt
```

Alternatively, ensure the following dependencies are installed:
```plaintext
python == 3.10.12
torch == 2.6.0
numpy == 1.26.4
scikit-learn == 1.5.1
shap == 0.42.1
h5py == 3.9.0
shap == 0.42.1
```

## 🗂️ Folder Structure
```
MFedMC/
│   README.md
│   requirements.txt
│
├─── ActionSense/
│   └─── dataset/
│       │   ActionSense_dataset.hdf5
│   │   main.py
│   │   utils_data.py
│   │   utils_train.py
│   │   options.py
│
│   # other datasets
```

- **`ActionSense/`**: Code for the ActionSense dataset
  - `main.py`: Main script for training and evaluating the MFedMC framework.
  - `utils_data.py`: Data loading, preprocessing, and data partitioning utilities.
  - `utils_train.py`: Functions related to model training.
  - `options.py`: Configuration settings.


## 🏃‍♂ Run Code

Run our framework with the following command:
```bash
python ActionSense/main.py
```
