import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, confusion_matrix

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

def predict_mahalanobis_all(data, ids):
    classes = data["Class"].unique()

    class_covariance_matrices = {}
    class_means = {}
    for c in classes:
        class_data = data[data["Class"] == c].drop(["Class", "ID"], axis=1)
        
        try:
            covariance_matrix = LedoitWolf().fit(class_data).covariance_
            class_covariance_matrices[c] = np.linalg.inv(covariance_matrix)
            class_means[c] = np.mean(class_data, axis=0)
        except np.linalg.LinAlgError:
            return f'A classe {c} não tem dados suficientes para poder fazer a distância de Mahalanobis'

    results = []
    for idx, sample in data[data['ID'].isin(ids)].iterrows():
        true_label = sample["Class"]
        distances = {c: mahalanobis_distance(sample.drop(["ID", "Class"]), class_means[c], class_covariance_matrices[c]) for c in classes}
        predicted_label = min(distances, key=distances.get)
        mahalanobis_distance_value = distances[predicted_label]
        results.append([sample['ID'], true_label, mahalanobis_distance_value, predicted_label])

    result_columns = ['ID', 'True Class', 'Mahalanobis Distance', 'Predicted Class']
    result_df = pd.DataFrame(results, columns=result_columns)
    accuracy = accuracy_score(result_df['True Class'], result_df['Predicted Class'])

    return result_df, accuracy

def predict(train_data, test_data):
    classes = train_data["Class"].unique()

    class_covariance_matrices = {}
    class_means = {}
    for c in classes:
        class_data = train_data[train_data["Class"] == c].drop("Class", axis=1)
        covariance_matrix = LedoitWolf().fit(class_data).covariance_
        class_covariance_matrices[c] = np.linalg.inv(covariance_matrix)
        class_means[c] = np.mean(class_data, axis=0)

    predicted_labels = []
    for _, sample in test_data.iterrows():
        distances = {c: mahalanobis_distance(sample[1:], class_means[c], class_covariance_matrices[c]) for c in classes}
        predicted_label = min(distances, key=distances.get)
        predicted_labels.append(predicted_label)

    # Evaluate accuracy
    accuracy = accuracy_score(test_data["Class"], predicted_labels)
    print(f'Acurácia: {accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(test_data["Class"], predicted_labels, labels=classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.title(f'Accuracy: {accuracy * 100:.2f}%')
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def process_mahalanobis_multiclass(ids_images):
    file_path = 'csv_characterization/features_multiclass.csv'
    df_multiclass = pd.read_csv(file_path)
    table, accuracy = predict_mahalanobis(df_multiclass, ids_images)
    print(f'Accuracy: {accuracy}')
    return table, accuracy
    
def process_mahalanobis_binary(ids_images):
    file_path = 'csv_characterization/features_binary.csv'
    df_binary = pd.read_csv(file_path)
    table, accuracy = predict_mahalanobis(df_binary, ids_images)
    print(f'Accuracy: {accuracy}')
    return table, accuracy

def process_mahalanobis_multiclass_exception(ids_images):
    file_path = 'csv_characterization/all_features_multiclass.csv'
    df_multiclass_ex = pd.read_csv(file_path)
    table, accuracy = predict_mahalanobis_all(df_multiclass_ex, ids_images)
    print(f'Accuracy: {accuracy}')
    return table, accuracy
    
def process_mahalanobis_binary_exception(ids_images):
    file_path = 'csv_characterization/all_features_binary.csv'
    df_binary_ex = pd.read_csv(file_path)
    table, accuracy = predict_mahalanobis_all(df_binary_ex, ids_images)
    print(f'Accuracy: {accuracy}')
    return table, accuracy

def process_mahalanobis_all_images():
    train_data_binary = pd.read_csv("csv_characterization/train_features_binary.csv")
    test_data_binary = pd.read_csv("csv_characterization/test_features_binary.csv")
    train_data_multiclass = pd.read_csv("csv_characterization/train_features_multiclass.csv")
    test_data_multiclass = pd.read_csv("csv_characterization/test_features_multiclass.csv")
    predict(train_data_multiclass, test_data_multiclass)
    predict(train_data_binary, test_data_binary)