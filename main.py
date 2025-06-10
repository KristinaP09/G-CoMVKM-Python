"""
G-CoMVKM Python Implementation
Globally Collaborative Multi-View k-Means Clustering

Author: Kristina P. Sinaga (Original MATLAB implementation)
Python Implementation: [Your Name]

This implementation is based on the MATLAB code by Kristina P. Sinaga.
The original algorithm integrates a collaborative transfer learning framework with
entropy-regularized feature-view reduction, enabling dynamic elimination of
uninformative components.
"""

import os
import sys
import argparse
from demo_2V2D2C import run_demo
import traceback

def main():
    """
    Main function to run the G-CoMVKM algorithm demonstration
    """
    parser = argparse.ArgumentParser(
        description='Globally Collaborative Multi-View k-Means Clustering (G-CoMVKM) Python Implementation'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='2V2D2C',
        choices=['2V2D2C'],
        help='Dataset to use for demonstration (default: 2V2D2C)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.dataset == '2V2D2C':
            print("Running G-CoMVKM on 2V2D2C dataset...")
            run_demo()
        else:
            print(f"Dataset '{args.dataset}' is not supported.")
            print("Supported datasets: 2V2D2C")
    except Exception as e:
        print(f"Error running the demo: {e}")
        print(traceback.format_exc())
    
if __name__ == "__main__":
    main()
