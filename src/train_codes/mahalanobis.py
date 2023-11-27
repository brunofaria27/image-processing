import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, confusion_matrix

def mahalanobis_distance(x, mean, inv_covariance_matrix):
    x_minus_mean = x - mean
    return np.sqrt(np.dot(np.dot(x_minus_mean, inv_covariance_matrix), x_minus_mean.T))

def predict(train_data, test_data, multiclass=False):
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
        distances = {c: mahalanobis_distance(sample[:-1], class_means[c], class_covariance_matrices[c]) for c in classes}
        predicted_label = min(distances, key=distances.get)
        predicted_labels.append(predicted_label)

    # Avaliação da acurácia
    accuracy = accuracy_score(test_data["Class"], predicted_labels)
    print(f'Acurácia: {accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(test_data["Class"], predicted_labels, labels=classes)
    df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    if multiclass: df.to_csv('confusion_matriz_mahalanobis/matriz_confusao_multiclass.csv', index=False)
    else: df.to_csv('confusion_matriz_mahalanobis/matriz_confusao_binary.csv', index=False)

def main():
    train_data_binary = pd.read_csv("csv_characterization/train_features_binary.csv")
    test_data_binary = pd.read_csv("csv_characterization/test_features_binary.csv")
    train_data_multiclass = pd.read_csv("csv_characterization/train_features_multiclass.csv")
    test_data_multiclass = pd.read_csv("csv_characterization/test_features_multiclass.csv")
    predict(train_data_binary, test_data_binary)
    predict(train_data_multiclass, test_data_multiclass, multiclass=True)

if __name__ == "__main__":
    main()