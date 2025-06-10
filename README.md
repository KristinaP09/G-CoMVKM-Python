# G-CoMVKM Python Implementation

## Overview
This is a Python implementation of the Globally Collaborative Multi-View k-Means (G-CoMVKM) clustering algorithm, originally developed by Kristina P. Sinaga. The algorithm integrates a collaborative transfer learning framework with entropy-regularized feature-view reduction, enabling dynamic elimination of uninformative components. This method achieves clustering by balancing local view importance and global consensus.

## Installation

### Requirements
- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- seaborn

You can install the dependencies using pip:

```bash
pip install numpy scipy matplotlib scikit-learn seaborn
```

## Structure
The project is organized as follows:

```
G-CoMVKM-Python/
├── main.py                 # Main script to run demonstrations
├── g_comvkm.py             # Implementation of the G-CoMVKM algorithm
├── demo_2V2D2C.py          # Demo script for the 2V2D2C dataset
├── utils/
│   ├── __init__.py
│   └── data_loader.py      # Utilities for loading datasets
└── evaluation/
    ├── __init__.py
    └── metrics.py          # Implementation of evaluation metrics
```

## Usage
To run the demo on the 2V2D2C (2 Views, 2 Dimensions, 2 Clusters) dataset:

```bash
python main.py --dataset 2V2D2C
```

## Algorithm Parameters
- `n_clusters`: Number of clusters to form
- `gamma`: Exponent parameter to control the weights of V (typically in range [0,1])
- `theta`: Coefficient parameter to control the weights of W (typically > 0)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `verbose`: Whether to print progress information
- `random_state`: Random seed for reproducibility

## Example
```python
from g_comvkm import GCoMVKM
from utils.data_loader import load_synthetic_data

# Load the dataset
X, label = load_synthetic_data()

# Create and fit the model
model = GCoMVKM(
    n_clusters=2,
    gamma=5.0,
    theta=4.0,
    max_iter=100,
    tol=1e-4,
    verbose=True,
    random_state=42
)

model.fit(X)

# Get the cluster assignments
predicted_labels = model.labels_
```

## Evaluation Metrics
The implementation includes several evaluation metrics:
- Normalized Mutual Information (NMI)
- Rand Index (RI)
- Adjusted Rand Index (ARI)
- Error Rate

## Visualizations
The demo script generates several visualizations:
1. Performance distribution across runs
2. Final view weights
3. Dimensionality reduction visualization
4. Clustering visualization
5. Confusion matrix
6. Performance metrics across different initializations
7. Convergence plot

## References
This implementation is based on the MATLAB code by Kristina P. Sinaga. For more details about the algorithm, please refer to the original paper.
