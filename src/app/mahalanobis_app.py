import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score

def mahalanobis_distance(x, mean, inv_covariance_matrix):
    x_minus_mean = x - mean
    return np.sqrt(np.dot(np.dot(x_minus_mean, inv_covariance_matrix), x_minus_mean.T))

def predict_mahalanobis(data):
    classes = data["Class"].unique()

    class_covariance_matrices = {}
    class_means = {}
    for c in classes:
        class_data = data[data["Class"] == c].drop("Class", axis=1)
        
        # TODO: Ver quando tem 1 dado para classe está certo dar erro
        try:
            covariance_matrix = LedoitWolf().fit(class_data).covariance_
            class_covariance_matrices[c] = np.linalg.inv(covariance_matrix)
            class_means[c] = np.mean(class_data, axis=0)
        except np.linalg.LinAlgError:
            return f'A classe {c} não tem dados suficientes para poder fazer a distância de Mahalanobis'

    results = []
    true_labels = []
    for _, sample in data.iterrows():
        true_label = sample["Class"]
        true_labels.append(true_label)

        distances = {c: mahalanobis_distance(sample[1:], class_means[c], class_covariance_matrices[c]) for c in classes}
        predicted_label = min(distances, key=distances.get)
        mahalanobis_distance_value = distances[predicted_label]
        results.append([true_label, mahalanobis_distance_value, predicted_label])

    predicted_labels = [result[2] for result in results]
    accuracy = accuracy_score(true_labels, predicted_labels)

    return pd.DataFrame(results, columns=['True Class', 'Mahalanobis Distance', 'Predicted Class']), accuracy

def process_mahalanobis_multiclass():
    file_path = 'csv_characterization/features_multiclass.csv'
    df_multiclass = pd.read_csv(file_path)
    table, accuracy = predict_mahalanobis(df_multiclass)
    print(table)
    print(f'Accuracy: {accuracy}')
    return table, accuracy
    
def process_mahalanobis_binary():
    file_path = 'csv_characterization/features_binary.csv'
    df_binary = pd.read_csv(file_path)
    table, accuracy = predict_mahalanobis(df_binary)
    print(table)
    print(f'Accuracy: {accuracy}')
    return table, accuracy
