# LDP-noise-addition

## Overview
This repository contains a Python script for adding Local Differential Privacy (LDP) noise to a dataset. The script demonstrates two types of noise addition methods: Laplacian noise and randomized response. It provides functions to add these noise types to both numerical and categorical columns of the dataset.

## Script Details
- The script provides functions to add Laplacian noise and randomized response noise to numerical and categorical columns, respectively.
- It utilizes the `numpy` and `pandas` libraries for numerical operations and data manipulation.
- Make sure to specify the privacy parameter `eps` according to the desired level of privacy for the dataset.
- The noisy dataset is saved as a new CSV file with the suffix `_eps_<epsilon_value>.csv` in the `Data/` directory.

## Usage
1. Clone the repository to your local machine:
   ```
   git clone https://github.com/your_username/LDP-noise-addition.git
   ```
2. Install the required dependencies:
   ```
   pip install numpy pandas
   ```
3. Place your dataset in the `Data/` directory. The script assumes the dataset is in CSV format.
4. Update the script (`add_ldp_noise.py`) with the correct path to your dataset and modify the categorical/numerical columns.
5. Run the script:
   ```
   python add_ldp_noise.py
   ```
6. The script will generate a new CSV file with the noisy dataset in the `Data/` directory.

## Dataset
The dataset used in this project is the "Adult Census Income" dataset, available from [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income). It contains various demographic and financial attributes of individuals.
