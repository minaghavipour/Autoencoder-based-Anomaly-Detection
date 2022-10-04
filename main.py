import matplotlib.pyplot as plt
import numpy as np
import timeit
import sys
import os

import torch
from utils import load_data, pre_processing, split_data, save_components, load_components, compute_accuracy, \
    save_result, prepare_data
from classifier import AutoEncoder, predict_Anomalies
from sklearn.model_selection import train_test_split


# def main(mode):
mode = "predict"
threshold = 0.27
components_file = "AE_components.pkl"
result_file = "AE_result.csv"
device = None
is_GPU_ON = True if input("Is GPU on? (y/n) \n") == "y" else False
if is_GPU_ON:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
file_path = input("Please enter the path to your csv-file \n")
name, extension = os.path.splitext(file_path)
if extension == '.csv':
    start = timeit.default_timer()
    input_data, output_classes = load_data(file_path, mode)
    if mode == "train":
        encoder, output_classes = pre_processing(output_classes)
        X_train, X_val, y_train, y_val = split_data(input_data, output_classes)

        train_pred, val_AE_pred, scaler_x, model = AutoEncoder(X=X_train, y=X_train,
                                                                                num_epochs=200,
                                                                                batch_size=30,
                                                                                learning_rate=0, reg_lambda=0,
                                                                                hidden_units=50, Batch=True,
                                                                                device=device)
        y = y_val.copy()
        X = prepare_data(X_val, scaler_x, device)

        X_train_T, X_val_T, y_train_T, y_val_T = train_test_split(X, y, test_size=0.2, stratify=y)
        train_pred_T, _ = model.predict(X_train_T)
        # sq_difference_train = predict_Anomalies(train_pred_T, X_train_T, y_train_T, threshold=threshold)
        # show_result(sq_difference_train, threshold, y_train_T)

        pred_class = predict_Anomalies(X_val_T, y_val_T, threshold, model, mode)
        compute_accuracy(pred_class, y_val_T)
        save_components(model, scaler_x, encoder, threshold, components_file)
    elif mode == "predict":
        model, scaler_x, encoder, threshold = load_components(components_file)
        X = prepare_data(input_data, scaler_x, device)
        pred_class = predict_Anomalies(X, [], threshold, model, mode)
        result = encoder.inverse_transform(pred_class)
        save_result(input_data, pred_class, result_file)
    stop = timeit.default_timer()
    print("Run Time: %.2f" % (stop - start))
    plt.show()
else:
    print("File appears not to be in CSV format")
    exit(-1)


# if len(sys.argv) == 2 and sys.argv[1] in ["train", "predict"]:
#     main(sys.argv[1])
# else:
#     print('Please run the code in the following format: main.py  [train/predict]')
#     exit(-1)
