import os
import jax.numpy as jnp
import pandas as pd
from urllib.request import urlretrieve
import zipfile

def download_and_extract_f16():
    
    # Load the dataset
    url = "https://data.4tu.nl/file/b6dc643b-ecc6-437c-8a8a-1681650ec3fe/5414dfdc-6e8d-4208-be6e-fa553de9866f"
    data_dir = "./data/f16/"
    zip_path = os.path.join(data_dir, "F16GVT_Files.zip")
    extracted_folder = os.path.join(data_dir, "F16GVT_Files")
    os.makedirs(data_dir, exist_ok=True)
    
    if os.path.exists(extracted_folder):
        print("Data alread loaded.")
        return extracted_folder
    
    print(f"Downloading data from {url}...")
    urlretrieve(url, zip_path)
    print("Done!")
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Done!")
    
    return extracted_folder


def load_io_data(filepath, header=True):
    """Load the data, filter missing values."""
    df = pd.read_csv(filepath, header=0 if header else None)
    # df = df.dropna(inplace=False)
    return df.values


def load_f16():
    
    # Get the files of interest
    folder = "./data/f16/F16GVT_Files/BenchmarkData/"
    files = [
        "F16Data_FullMSine_Level1.csv",
        "F16Data_FullMSine_Level2_Validation.csv",
        "F16Data_FullMSine_Level3.csv",
        "F16Data_FullMSine_Level4_Validation.csv",
        "F16Data_FullMSine_Level5.csv",
        "F16Data_FullMSine_Level6_Validation.csv",
        "F16Data_FullMSine_Level7.csv"
    ]
    file_paths = [os.path.join(folder, f) for f in files]
    
    # Use dataset 4 for standardisation
    ref_data = load_io_data(file_paths[3])
    mu = jnp.mean(ref_data[:, :5], axis=0)
    sigma = jnp.std(ref_data[:, :5], axis=0)
    
    def process_file(filepath):
        data = load_io_data(filepath)   # Get data
        data = data[:, :5]              # Select relevant columns
        data = (data - mu) / sigma      # Standardization
        return data[:, :2], data[:, 2:] # Split into the 2 inputs and 3 measurements
    
    # Store data with shape (time, batches, ...)
    datasets = [process_file(fp) for fp in file_paths] 
    u_train = jnp.swapaxes(jnp.array([datasets[i][0] for i in [0, 2, 4, 6]]), 1, 0)
    y_train = jnp.swapaxes(jnp.array([datasets[i][1] for i in [0, 2, 4, 6]]), 1, 0)
    u_val = jnp.swapaxes(jnp.array([datasets[i][0] for i in [1, 3, 5]]), 1, 0)
    y_val = jnp.swapaxes(jnp.array([datasets[i][1] for i in [1, 3, 5]]), 1, 0)
    
    return (u_train, y_train), (u_val, y_val)
