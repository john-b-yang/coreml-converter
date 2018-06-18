# Basics
import sys
import numpy as np
import pandas as pd

# Sklearn Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Custom Classes
from mlfunction.helper import getMLModels

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

def conversion(data_file, split_ratio, user_models, y_data_column):
    split_ratio = float(split_ratio) if float(split_ratio )< 1 else int(split_ratio)/100

    # Step 1: Import user data, put into data frame, create test/train split data
    raw_data = open(data_file, 'r')
    data_frame = pd.read_csv(data_file, index_col=0) # Removing index column
    y_data = data_frame[[y_data_column]]
    x_data = data_frame.drop(axis=1, labels=[y_data_column])

    # TODO: Add option to randomize rows

    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, train_size=split_ratio, random_state=0)

    # Step 2: Build testing pipeline based on user selected data sets
    requested = [x.lower() for x in user_models.split()]

    pipelines, filtered_requests = getMLModels(requested)

    # Step 3: Test Selected Models, Save results in txt file, Output Ranking
    output_file = open('output.txt', 'w')
    return_value = ""

    for index in range(len(pipelines)):
        pipeline = pipelines[index]

        pipeline.fit(X_train, Y_train)
        Y_predict = pipeline.predict(X_test)

        report = classification_report(Y_test, Y_predict)
        matrix = confusion_matrix(Y_test, Y_predict)

        return_value += str("Model: %s\n" % filtered_requests[index])
        return_value += str("Classification Report: \n%s\n" % report)
        return_value += str("Confusion Matrix: \n%s\n\n\n" % matrix)

    output_file.write(return_value)
    output_file.close()
    return return_value

if __name__ == '__main__':
    data_file_path = input("Enter path pointing to data file: ")
    ratio = input("Enter test split percentage: ")
    requested_models = input("Enter models to test, separated by spaces: ")
    y_data_col = input("Enter column name of Y/Label data: ")

    print("{0}, {1}, {2}, {3}".format(data_file_path, ratio, requested_models, y_data_col))
    conversion(data_file_path, ratio, requested_models, y_data_col)
