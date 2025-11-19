# RUNet: A Zero-Calibration Framework for Cross-Domain EEG Decoding via Riemannian and Unsupervised Representation Learning
This repository contains the code for RUNet implemented with PyTorch.
More details in paper: RUNet: A Zero-Calibration Framework for Cross-Domain EEG Decoding via Riemannian and Unsupervised Representation Learning

## Repository Structure
- `compare_model/`: Comparative models from literature
- {main.py}: Main training pipeline (pretraining + fine-tuning)
- {model.py}: Base model architecture with MSCM module
- { load_data.py}: Data loading and augmentation utilities
- { NTXentLoss.py}: Contrastive loss for pretraining
- {utils.py}: Key modules (PCOM, AIT, LEM)
- `config.yaml`: Hyperparameter configuration
- `requirements.txt`: Dependencies
  
## Hyperparameters
| Stage         | Batch Size | Learning Rate | Epochs | Temperature (τ) |
|---------------|------------|---------------|--------|-----------------|
| Pretraining   | 64         | 0.001         | 20     | 0.5             |
| Fine-tuning   | 32         | 0.0009        | 100/30 | -               |
- Temporal convolution kernels: 15 (KT1), 55 (KT2), 75 (KT3)
- Spatial convolution kernels (KE): 3 (for most datasets), 1 (for BCIC-IV-2b)
- PCOM module: α (initial=0.5, learnable), fusion weights w (initial=1, learnable)
- RALM module: EMA parameter α=0.1
- Numerical stability constant ε=1×10⁻⁶
  
## Environment requirements
The requirements.txt file provides the environment requirements.

## Data Availability
- The BCIC-IV-2a dataset can be downloaded in the following link: https://www.bbci.de/competition/iv/.
- The BCIC-IV-2b dataset can be downloaded in the following link: https://www.bbci.de/competition/iv/.
- The PhysioNet dataset can be downloaded in the following link: https://www.physionet.org/content/eegmmidb/1.0.0/

## Implementations of ShallowNet, EEGNet, FBCNet, EEG-conformer, ATCNet, TransNet, MSVTNet, DMSANet and SST-DPN
- ShallowNet is provided at https://github.com/robintibor/braindecode/  
- EEGNet is provided at https://github.com/vlawhern/arl-eegmodels
- FBCNet is provided at https://github.com/ravikiran-mane/FBCNet
- EEG-conformer is provided at https://github.com/eeyhsong/EEG-Conformer
- ATCNet is provided at https://github.com/Altaheri/EEG-ATCNet
- TransNet, is provided at https://github.com/Ma-Xinzhi/EEG-TransNet
- DMSANet is provided at https://github.com/xingxin-99/DMSANet
- SST-DPN is provided at https://github.com/hancan16/SST-DPN
