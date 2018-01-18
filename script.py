# Basics
import sys
import numpy as np
import pandas as pd

# Sklearn Models
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Sklearn Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def conversion(data_file, split_ratio, user_models, y_data_column):
    split_ratio = float(split_ratio) if float(split_ratio )< 1 else int(split_ratio)/100

    # Step 1: Import user data, put into data frame, create test/train split data
    raw_data = open(data_file, 'r')
    data_frame = pd.read_csv(data_file, index_col=0) # Removing index column
    y_data = data_frame[[y_data_column]]
    x_data = data_frame.drop(axis=1, labels=[y_data_column])

    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, train_size=split_ratio, random_state=0)

    # Step 2: Build testing pipeline based on user selected data sets
    requested_models = user_models.split()
    requested_models = [x.lower() for x in requested_models]

    available_models = ['svc', 'knn', 'decision-trees', 'random-forest', 'gradient-boosted']

    svc = Pipeline([('clf', SVC())])
    knn = Pipeline([('clf', KNeighborsClassifier(n_neighbors=3))])
    decision_trees = Pipeline([('clf', DecisionTreeClassifier())])
    random_forest = Pipeline([('clf', RandomForestClassifier())])
    gradient_boosted = Pipeline([('clf', GradientBoostingClassifier())])

    available_pipelines = [svc, knn, decision_trees, random_forest, gradient_boosted]

    # Logic:
    # 1. Check if requested model is part of available models
    # 2. If available, append the model's name to an array
    # 3. Then, append the model's corresponding pipeline to the requested_pipelines array
    # *Note* requested_models_filtered & requested_pipelines should be same length
    requested_pipelines = []
    requested_models_filtered = []
    for model in requested_models:
        if model in available_models:
            requested_models_filtered.append(model)
            requested_pipelines.append(available_pipelines[available_models.index(model)])

    # Step 3: Test Selected Models, Save results in txt file, Output Ranking
    output_text_file = open('output.txt', 'w')
    count = 0
    for index in range(len(requested_pipelines)):
        pipeline = requested_pipelines[index]

        pipeline.fit(X_train, Y_train)
        Y_predict = pipeline.predict(X_test)

        report = classification_report(Y_test, Y_predict)
        matrix = confusion_matrix(Y_test, Y_predict)

        output_text_file.write("Model: %s\n" % requested_models_filtered[index])
        output_text_file.write("Classification Report: \n%s\n" % report)
        output_text_file.write("Confusion Matrix: \n%s\n" % matrix)
        output_text_file.write("\n\n")
        count = count + 1

    output_text_file.close()

def convertToCoreML():
    return 'not complete'

if __name__ == '__main__':
    data_file_path = input("Enter path pointing to data file: ")
    ratio = input("Enter test split percentage: ")
    requested_models = input("Enter models to test, separated by spaces: ")
    y_data_col = input("Enter column name of Y/Label data: ")

    print("{0}, {1}, {2}, {3}".format(data_file_path, ratio, requested_models, y_data_col))
    conversion(data_file_path, ratio, requested_models, y_data_col)
