import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def process_mahalanobis_binary():
    file_path = 'csv_characterization/features_binary.csv'
    df = load_data(file_path)
    return None