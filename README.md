## Communication-Efficient Multimodal Federated Learning: Joint Modality and Client Selection

[![python](https://img.shields.io/badge/Python_3.10-306998?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/release/python-31012/)
[![License: MIT](https://img.shields.io/badge/license-MIT-750014.svg)](https://opensource.org/licenses/MIT) 
[![arXiv](https://img.shields.io/badge/arXiv-2401.16685-b31b1b.svg)](https://arxiv.org/abs/2401.16685)


## 🔥 Our Framework

TL;DR: In this repo, we provide the implementation of **Multimodal Federated Learning with Joint Modality and Client Selection** (MFedMC), a novel methodology for multimodal federated learning that decouples traditional holistic fusion approaches into separate global modality encoders and local fusion modules.

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
│   │   main.py
│   │   dataset.py
│   │   federated.py
│   │   models.py
│   │   options.py
│   │   utils.py
│
│   # other datasets
```

- **`ActionSense/`**: Core modules for the ActionSense dataset framework. Note: different datasets might have different network architectures and data loaders, so they are separated by folder.
  - `main.py`: Entry point for initializing data, generalized global modality encoders, and running the MFedMC federated learning loop.
  - `dataset.py`: Data loading, restructuring, and partitioning utilities (e.g., Dirichlet, Stratified) to handle dataset and modality heterogeneities.
  - `federated.py`: Core MFedMC algorithm, including local training of modality encoders, personalized local fusion module training, computation of Shapley values, and the joint modality and client selection strategies based on priority metrics.
  - `models.py`: PyTorch neural network definitions for modality encoders (e.g., Eye, EMG, Tactile, IMU).
  - `options.py`: Hyperparameters and configuration settings.
  - `utils.py`: Helper functions for math, normalizations, and SHAP value aggregations.


## 🏃‍♂ Run Code

Run our framework with the following command:
```bash
cd ActionSense
python main.py
```
