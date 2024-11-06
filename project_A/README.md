# Dataset Distillation: A Data-Efficient Learning Framework

## Overview

This project explores dataset distillation techniques by comparing two methods, DataDAM and PAD, for creating compact synthetic datasets. Dataset distillation compresses data into smaller, information-rich sets that allow for faster training with less computational load.

Using MNIST as a benchmark, this study examines DataDAM’s Attention Matching and PAD’s alignment-based approach, with PAD customized for MNIST through a data selection strategy inspired by DeepCore. DataDAM is also applied to the larger MHIST dataset to test its adaptability to more complex data.

## Project Structure
```
ECE1512_2024F_ProjectRepo_SwapnilPatel/project_A/
├── report/                             # Report directory
|   ├── figures/                        # contains all the figures used in the report
├── src/
│   ├── task1.ipynb                     # Jupyter notebook for Task 1 implementation
│   ├── task1_application.ipynb         # Jupyter notebook for Task 1 application (NAS)
│   ├── task2/
│   │   ├── PAD/                        # Adapted PAD method for MNIST dataset
│   ├── task2.ipynb                     # Jupyter notebook to visualize data and results for Task 2
|   ├── network.py                      # Contains all networks used in this project
|   ├── utils.py                        # Contains various utilities function used in this project.
└── README.md                           # Project README file
```

### Hardware Information
All experiements and results prented in this repository were conducted on following hardware: 
```
CPU : AMD EPYC 7B13
PyTorch Version: 2.2.1
CUDA Device: NVIDIA GeForce RTX 4090
CUDA Compute Capability: 8.9
Total Memory: 23.65 GB
```
