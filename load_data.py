import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np

'''
base_contrast_source_path: Pretraining contrastive learning data input path
base_train_path: Fine-tuning phase training data input path
base_test_path: Test data input path

Data Storage Format: All data are independently stored as .mat files per subject. 
- Contrastive learning data filenames follow the "A0X" naming convention (where X denotes the subject index)
- Training data filenames follow the "A0XT" naming convention (where X denotes the subject index)
- Test data filenames follow the "A0XE" naming convention (where X denotes the subject index)

Data Dimensionality Specification: Each file corresponds to a tensor with dimensions (Number of Samples, Number of Channels, Number of Temporal Features) -> (N, C, T)
'''

base_contrast_source_path = r'path/to/train/A{}.mat'
base_train_path = r'path/to/PreTraining/A{}T.mat'
base_test_path = r'path/to/test/A{}E.mat'


def load_contrast_data(subject_id):
    """Load and enhance the EEG data for contrastive learning."""
    source_path = base_contrast_source_path.format(str(subject_id).zfill(2))
    source_data = sio.loadmat(source_path)['data']

    augmented_data = []
    labels = []

    for i, sample in enumerate(source_data):
        # Augmentation Method 1: Noise + Scaling
        noise = np.random.randn(*sample.shape) * 0.1  # noise level
        scale_factor = np.random.uniform(0.8, 1.2)  # Zoom range
        aug1 = (sample + noise) * scale_factor

        # Enhancement Method 2: Zooming + Time Mask
        scale_factor2 = np.random.uniform(0.8, 1.2)
        scaled = sample * scale_factor2

        # Time mask (randomly masking certain time points)
        time_mask = np.random.rand(sample.shape[-1]) > 0.1  # 10% of the time points have been blocked.
        aug2 = scaled * time_mask[np.newaxis, :]

        augmented_data.extend([aug1, aug2])
        labels.extend([i, i])  # The enhancement of the same original sample is classified as a positive sample pair.

    return torch.tensor(np.array(augmented_data), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
def load_task_data(subject_id):
    """Task data loading"""
    source_path = base_train_path.format(str(subject_id).zfill(2))
    source_data = sio.loadmat(source_path)
    X_train = torch.tensor(source_data['x_data'], dtype=torch.float32)
    y_train = torch.tensor(source_data['y_data'].reshape(-1), dtype=torch.long)

    test_path = base_test_path.format(str(subject_id).zfill(2))
    test_data = sio.loadmat(test_path)
    X_test = torch.tensor(test_data['x_data'], dtype=torch.float32)
    y_test = torch.tensor(test_data['y_data'].reshape(-1), dtype=torch.long)

    return (X_train, y_train), (X_test, y_test)

#
