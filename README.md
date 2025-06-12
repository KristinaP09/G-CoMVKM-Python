# G-CoMVKM: Globally Collaborative Multi-View k-Means Clustering

[![PyPI version](https://badge.fury.io/py/gcomvkm.svg)](https://badge.fury.io/py/gcomvkm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/gcomvkm.svg)](https://pypi.org/project/gcomvkm/)
[![Paper](https://img.shields.io/badge/Paper-MDPI%20Electronics-blue)](https://www.mdpi.com/2079-9292/14/11/2129)

## Overview

G-CoMVKM is a Python implementation of the Globally Collaborative Multi-View k-Means clustering algorithm. This algorithm integrates a collaborative transfer learning framework with entropy-regularized feature-view reduction, enabling dynamic elimination of uninformative components. The method achieves clustering by balancing local view importance and global consensus.

### Key Features

- **Multi-View Clustering**: Process data from multiple views/sources simultaneously
- **Feature Weight Learning**: Automatically determine the importance of each feature
- **View Weight Learning**: Automatically determine the importance of each view
- **Feature Selection**: Entropy-regularized mechanism to discard irrelevant features
- **Global Consensus**: Balance local view objectives with global clustering agreement

## Installation

You can install G-CoMVKM directly from PyPI:

```bash
pip install gcomvkm
```

### Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- seaborn

## Quick Start

Here's a simple example of how to use G-CoMVKM:

```python
from gcomvkm import GCoMVKM
from gcomvkm.utils import load_synthetic_data
from gcomvkm.evaluation import nmi, rand_index, adjusted_rand_index

# Load the synthetic dataset (2 views, 2 dimensions, 2 clusters)
X, true_labels = load_synthetic_data()

# Create and fit the model
model = GCoMVKM(
    n_clusters=2,
    gamma=5.0,     # Feature selection regularization parameter
    theta=4.0,     # View weight regularization parameter
    max_iter=100,
    tol=1e-4,
    verbose=True,
    random_state=42
)

# Fit the model to the data
model.fit(X)

# Get the clustering results
predicted_labels = model.labels_
feature_weights = model.feature_weights_
view_weights = model.view_weights_

# Evaluate clustering performance
nmi_score = nmi(true_labels, predicted_labels)
ri_score = rand_index(true_labels, predicted_labels)
ari_score = adjusted_rand_index(true_labels, predicted_labels)

print(f"NMI Score: {nmi_score:.4f}")
print(f"Rand Index: {ri_score:.4f}")
print(f"Adjusted Rand Index: {ari_score:.4f}")
```

## Algorithm Description

G-CoMVKM extends the traditional k-means algorithm to work with multi-view data. The algorithm:

1. **Initializes** cluster centers randomly or using k-means++
2. **Computes memberships** for each data point to the clusters
3. **Updates cluster centers** based on these memberships
4. **Updates feature weights** using an entropy-regularized optimization
5. **Discards irrelevant features** based on a threshold
6. **Updates view weights** to balance view importance
7. **Repeats** steps 2-6 until convergence

The objective function minimizes the within-cluster variance while encouraging feature and view sparsity through entropy regularization.

## CLI Usage

G-CoMVKM also provides a command-line interface:

```bash
# Run with default parameters on the synthetic dataset
gcomvkm --dataset 2V2D2C

# Run with custom parameters
gcomvkm --dataset 2V2D2C --gamma 5.0 --theta 4.0 --n-clusters 2 --max-iter 100
```

### 💻 Technical Excellence & Implementation

1. **Comprehensive Cross-Platform Development**

   - ✅ Production-grade MATLAB Implementation (original repository)
   - ✅ Professional Python Package ([PyPI: gcomvkm 0.1.0](https://pypi.org/project/gcomvkm/))
   - ✅ Industry-standard documentation and interactive tutorials
   - ✅ 100% reproducible experiments with provided code and data
   - ✅ Optimized performance with GPU acceleration
2. **Quality Assurance**

   - Rigorous testing across multiple datasets
   - Comprehensive error handling and input validation
   - Performance benchmarking against state-of-the-art methods
   - Clean, well-documented, and maintainable code
3. **User Experience**

   - Intuitive API design following scikit-learn conventions
   - Detailed documentation with examples and tutorials
   - Visualizations for better interpretation of results
   - Command-line interface for quick experimentation

## Citation

If you use G-CoMVKM in your research, please cite:

```bibtex
@Article{electronics14112129,
AUTHOR = {Sinaga, Kristina P. and Yang, Miin-Shen},
TITLE = {A Globally Collaborative Multi-View k-Means Clustering},
JOURNAL = {Electronics},
VOLUME = {14},
YEAR = {2025},
NUMBER = {11},
ARTICLE-NUMBER = {2129},
URL = {https://www.mdpi.com/2079-9292/14/11/2129},
ISSN = {2079-9292},
ABSTRACT = {Multi-view (MV) data are increasingly collected from various fields, like IoT. The surge in MV data demands clustering algorithms capable of handling heterogeneous features and high dimensionality. Existing feature-weighted MV k-means (MVKM) algorithms often neglect effective dimensionality reduction such that their scalability and interpretability are limited. To address this, we propose a novel procedure for clustering MV data, namely a globally collaborative MVKM (G-CoMVKM) clustering algorithm. The proposed G-CoMVKM integrates a collaborative transfer learning framework with entropy-regularized feature-view reduction, enabling dynamic elimination of uninformative components. This method achieves clustering by balancing local view importance and global consensus, without relying on matrix reconstruction. We design a feature-view reduction by embedding transferred learning processes across view components by using penalty terms and entropy to simultaneously reduce these unimportant feature-view components. Experiments on synthetic and real-world datasets demonstrate that G-CoMVKM consistently outperforms these existing MVKM clustering algorithms in clustering accuracy, performance, and dimensionality reduction, affirming its robustness and efficiency.},
DOI = {10.3390/electronics14112129}
}

```

## Note

The original code has been tested on MATLAB R2020a. Performance on other versions may vary. This Python implementation has been tested on Python 3.7+ and is compatible with most modern Python environments.

### 💫 Beyond the "Impossible"

As Arthur C. Clarke said, "The only way of discovering the limits of the possible is to venture a little way past them into the impossible."

We didn't just venture—we blazed a trail:

- Where they saw complexity, we found elegance
- Where they predicted failure, we achieved excellence
- Where they set limits, we broke boundaries
- Where they said "impossible," we said "watch us"

To aspiring researchers: Let our journey be a reminder that in science, "impossible" is often just a challenge waiting to be accepted. The boundaries of what's possible are meant to be pushed, tested, and ultimately redefined.

## Contact

- **Kristina P. Sinaga**
- Email: kristinasinaga41@gmail.com
- [GitHub](https://github.com/KristinaP09)

## References

1. [A Globally Collaborative Multi-View k-Means Clustering](https://www.mdpi.com/2079-9292/14/11/2129) - Electronics MDPI
2. Original MATLAB Implementation: [G-CoMVKM](https://github.com/KristinaP09/G-CoMVKM)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
