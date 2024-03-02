import numpy as np
import math
import pandas as pd

def laplace_noise_col(val, sensitivity, eps):
    """
    Add Laplace noise to a single value.

    Parameters:
    - val: The original value.
    - sensitivity: The sensitivity of the data.
    - eps: The privacy parameter epsilon.

    Returns:
    - noisy_val: The value with added Laplace noise.
    """
    lambda_ = sensitivity / eps
    noise = np.random.laplace(loc=0, scale=lambda_)
    return round(val + noise)

def add_laplacian_noise(df, eps, col_name):
    """
    Add Laplace noise to a column in a DataFrame.

    Parameters:
    - df: The DataFrame containing the data.
    - eps: The privacy parameter epsilon.
    - col_name: The name of the column to which noise will be added.

    Returns:
    - noisy_df: The DataFrame with added Laplace noise.
    """
    sensitivity = df[col_name].max() - df[col_name].min()
    df[col_name] = df.apply(lambda x: laplace_noise_col(x[col_name], sensitivity, eps), axis=1)
    return df

def randomized_response(choices, p, q, val):
    """
    Apply randomized response mechanism to a single value.

    Parameters:
    - choices: List of possible choices.
    - p: Probability of keeping the original value.
    - q: Probability of switching to another value.
    - val: The original value to be perturbed.

    Returns:
    - new_val: The perturbed value.
    """
    rand_val = np.random.uniform(0, 1)
    new_val = val
    new_choices = choices.copy()
    new_choices.remove(val)
    if rand_val > p:
        rand_q = np.random.uniform(0, 1)
        index = math.floor(rand_q * len(new_choices))
        new_val = new_choices[index]
    return new_val

def add_randomized_response_noise(df, eps, col_name):
    """
    Apply randomized response mechanism to a column in a DataFrame.

    Parameters:
    - df: The DataFrame containing the data.
    - eps: The privacy parameter epsilon.
    - col_name: The name of the column to which noise will be added.

    Returns:
    - noisy_df: The DataFrame with added randomized response noise.
    """
    choices = np.array(df[col_name].to_numpy())
    choices = np.unique(choices)
    n = len(choices)
    p = np.exp(eps) / (np.exp(eps) + n - 1)  # Keep the same value with probability p
    q = 1 / (np.exp(eps) + n - 1)            # Switch to any other choice with probability q

    label = df[col_name].to_numpy()
    new_label = []
    for l in label:
        new_label.append(randomized_response(choices.tolist(), p, q, l))
    df[col_name] = new_label
    return df

# Set privacy parameter epsilon
eps = 5

# Read the DataFrame from a CSV file
df = pd.read_csv("Data/training.csv", sep=",", index_col=0)

# Add Laplacian noise to numerical columns
columns_to_noise = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
for col in columns_to_noise:
    df = add_laplacian_noise(df, eps, col)

# Add randomized response noise to categorical columns
categorical_columns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "class_attribute"]
for col in categorical_columns:
    df = add_randomized_response_noise(df, eps, col)

# Save the noisy DataFrame to a CSV file
df.to_csv("Data/training_eps_" + str(eps) + ".csv", sep=',')
