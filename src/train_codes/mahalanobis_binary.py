import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_encoder = LabelEncoder()
    train_df['Class'] = label_encoder.fit_transform(train_df['Class'])
    test_df['Class'] = label_encoder.transform(test_df['Class'])

    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    return X_train, y_train, X_test, y_test

def calculate_mahalanobis_distance(x, mean, inv_cov_matrix):
    mean_diff = x - mean
    return np.sqrt(np.dot(mean_diff, np.dot(inv_cov_matrix, mean_diff)))

def main():
    train_path = 'csv_characterization/train_features_binary.csv'
    test_path = 'csv_characterization/test_features_binary.csv'

    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_path, test_path)

    cov_matrix = LedoitWolf().fit(X_train).covariance_
    inv_cov_matrix = np.linalg.pinv(cov_matrix)

    train_mean = X_train.mean()
    test_mean = X_test.mean()

    train_distances = X_train.apply(lambda x: calculate_mahalanobis_distance(x, train_mean, inv_cov_matrix), axis=1)
    test_distances = X_test.apply(lambda x: calculate_mahalanobis_distance(x, test_mean, inv_cov_matrix), axis=1)

    threshold = 0.5
    y_train_pred = (train_distances > threshold).astype(int)
    y_test_pred = (test_distances > threshold).astype(int)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)

    print(f'Train Accuracy: {train_accuracy:.2f}')
    print(f'Test Accuracy: {test_accuracy:.2f}')
    print('\nConfusion Matrix - Train:')
    print(train_conf_matrix)
    print('\nConfusion Matrix - Test:')
    print(test_conf_matrix)

if __name__ == "__main__":
    main()
