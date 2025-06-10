import os
import sys
import scipy.io as sio
import numpy as np

# Path to check
dataset_path = os.path.expanduser('~/Desktop/G-CoMVKM/dataset/data_synthetic.mat')

print(f"Checking if dataset exists at: {dataset_path}")
print(f"File exists: {os.path.exists(dataset_path)}")

if os.path.exists(dataset_path):
    try:
        print("Trying to load the dataset...")
        mat_data = sio.loadmat(dataset_path)
        print("Dataset successfully loaded!")
        
        # Print the keys in the .mat file
        print("Keys in the .mat file:", mat_data.keys())
        
        if 'data' in mat_data:
            data = mat_data.get('data')
            print("Data shape:", data.shape)
            
            # If it's a cell array in MATLAB, convert to list of numpy arrays
            if hasattr(data, 'dtype') and data.dtype == 'object':
                print("Data is a cell array in MATLAB")
                X = []
                for i in range(data.shape[1]):
                    X.append(data[0, i])
                    print(f"View {i+1} shape:", X[i].shape)
        
        if 'label' in mat_data:
            label = mat_data.get('label').flatten()
            print("Label shape:", label.shape)
            print("Unique labels:", np.unique(label))
    except Exception as e:
        print(f"Error loading the dataset: {e}")
else:
    print("File does not exist. Please check the path.")
