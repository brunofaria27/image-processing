import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def process_mahalanobis_multiclass():
    file_path = 'csv_characterization/features_multiclass.csv'
    df = load_data(file_path)
    return None
