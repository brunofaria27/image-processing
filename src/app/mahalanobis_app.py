import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score

def mahalanobis_distance(x, mean, inv_covariance_matrix):
    x_minus_mean = x - mean
    return np.sqrt(np.dot(np.dot(x_minus_mean, inv_covariance_matrix), x_minus_mean.T))

def predict_mahalanobis(data, ids):
    classes = data["Class"].unique()

    class_covariance_matrices = {}
    class_means = {}
    for c in classes:
        class_data = data[data["Class"] == c].drop("Class", axis=1)
        
        try:
            covariance_matrix = LedoitWolf().fit(class_data).covariance_
            class_covariance_matrices[c] = np.linalg.inv(covariance_matrix)
            class_means[c] = np.mean(class_data, axis=0)
        except np.linalg.LinAlgError:
            return f'A classe {c} não tem dados suficientes para poder fazer a distância de Mahalanobis'

    results = []
    for idx, sample in data.iterrows():
        true_label = sample["Class"]
        distances = {c: mahalanobis_distance(sample[1:], class_means[c], class_covariance_matrices[c]) for c in classes}
        predicted_label = min(distances, key=distances.get)
        mahalanobis_distance_value = distances[predicted_label]
        results.append([ids[idx], true_label, mahalanobis_distance_value, predicted_label])

    result_columns = ['ID', 'True Class', 'Mahalanobis Distance', 'Predicted Class']
    result_df = pd.DataFrame(results, columns=result_columns)
    accuracy = accuracy_score(result_df['True Class'], result_df['Predicted Class'])

    return result_df, accuracy

def process_mahalanobis_multiclass(ids_images):
    file_path = 'csv_characterization/features_multiclass.csv'
    df_multiclass = pd.read_csv(file_path)
    table, accuracy = predict_mahalanobis(df_multiclass, ids_images)
    print(table)
    print(f'Accuracy: {accuracy}')
    return table, accuracy
    
def process_mahalanobis_binary(ids_images):
    file_path = 'csv_characterization/features_binary.csv'
    df_binary = pd.read_csv(file_path)
    table, accuracy = predict_mahalanobis(df_binary, ids_images)
    print(table)
    print(f'Accuracy: {accuracy}')
    return table, accuracy
