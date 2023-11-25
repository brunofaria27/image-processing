import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# TODO: Tentar arrumar

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

def main():
    train_path = 'csv_characterization/train_features_multiclass.csv'
    test_path = 'csv_characterization/test_features_multiclass.csv'

    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_path, test_path)

    class_models = {}
    for class_label in y_train.unique():
        class_data = X_train[y_train == class_label]
        model = GaussianMixture(n_components=1, covariance_type='full')
        model.fit(class_data)
        class_models[class_label] = model

    train_likelihoods = np.array([class_models[class_label].score_samples(X_train) for class_label in y_train.unique()]).T
    test_likelihoods = np.array([class_models[class_label].score_samples(X_test) for class_label in y_train.unique()]).T

    y_train_pred = np.argmax(train_likelihoods, axis=1)
    y_test_pred = np.argmax(test_likelihoods, axis=1)

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
