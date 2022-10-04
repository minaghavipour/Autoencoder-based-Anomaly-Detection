import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import csv


def load_data(file_path, mode):
    input_data = []
    output_classes = []
    with open(file_path, 'r', newline='') as csvfile:
        log_reader = csv.reader(csvfile, delimiter=',')
        next(log_reader)
        for row in log_reader:
            if mode == "train":
                input_data.append(row[:-1])
                output_classes.append(row[-1])
            else:
                input_data.append(row)
    input_data = np.array(input_data, dtype=int)
    return input_data, output_classes


def pre_processing(output_classes):
    encoder = preprocessing.LabelEncoder()
    output_classes = 1 - encoder.fit_transform(output_classes).astype(int)  # 1: anomaly  0: normal
    return encoder, output_classes


def split_data(X, y):
    y_1 = y[y == 1]
    y_0 = y[y == 0]

    X_0 = X[y == 0]
    X_1 = X[y == 1]

    num_class_0_examples = X_0.shape[0]
    permutation = np.random.permutation(num_class_0_examples)
    indx_90 = int(num_class_0_examples * 0.9)

    X_0_train = X_0[permutation[:indx_90]]
    X_0_val = X_0[permutation[indx_90:]]

    y_0_train = y_0[permutation[:indx_90]]
    y_0_val = y_0[permutation[indx_90:]]

    X_val = np.concatenate([X_1, X_0_val], axis=0)
    y_val = np.concatenate([y_1, y_0_val], axis=0)

    X_train = X_0_train.copy()
    y_train = y_0_train.copy()

    X_val = X_val.copy()
    y_val = y_val.copy()
    return X_train, X_val, y_train, y_val


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def prepare_data(X, scaler, device, fit=False):
    X = scaler.fit_transform(X) if fit else scaler.transform(X)
    X = torch.from_numpy(X.astype(np.float32))
    X = X.to(device)
    return X


def save_result(input_data, result, result_file):
    data = [row for row in zip(input_data, result)]
    np.savetxt(result_file, data, delimiter=',', fmt='%s')


def save_components(model, scalar, encoder, threshold, components_file):
    joblib.dump([model, scalar, encoder, threshold], components_file)


def load_components(components_file):
    model, scalar, encoder, threshold = joblib.load(components_file)
    return model, scalar, encoder, threshold


def compute_accuracy(pred_CLASS, y_T):
    CR = classification_report(y_T, pred_CLASS)
    CM = confusion_matrix(y_T, pred_CLASS)

    plt.figure("Confusion Matrix")
    sns.heatmap(CM, annot=True, fmt="d", center=10000)
    plt.xlabel("Predictions")
    plt.ylabel("True labels")
    print()

    TPR = CM[1, 1] / (CM[1, 1] + CM[1, 0])
    val_f1_score = f1_score(y_T, pred_CLASS)
    print("TPR = ", TPR)
    print("F1 score =", val_f1_score)