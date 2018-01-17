# Basics
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
from sklearn.model_seletion import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Step 1: Import user data, put into data frame, create test/train split data
raw_data = open('MARK: User data', 'r')

data_frame = pd.DataFrame(raw_data)

# Assumes formatting of 'Y' labels as first column, 'X' data vector as everything else
x_data = data_frame.iloc[0:, 1:]
y_data = data_frame.iloc[0:, :1]

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, train_size='MARK: Split %', random_state=0)

# Step 2: Build testing pipeline based on user selected data sets
requested_models = ['MARK: User models']
requested_pipelines = []

# Names of Models + Corresponding Pipelines
available_models = ['SVC', 'KNN', 'Decision Trees', 'Random Forest', 'Gradient Boosted']

SVC = Pipeline([('clf', SVC())])
KNN = Pipeline([('clf', KNeighborsClassifier(n_neighbors=3))])
decision_trees = Pipeline([('clf', DecisionTreeClassifier())])
random_forest = Pipeline([('clf', RandomForestClassifier())])
gradient_boosted = Pipeline([('clf', GradientBoostingClassifier())])
available_pipelines = [SVC, KNN, decision_trees, random_forest, gradient_boosted]

# Logic:
# 1. Check if requested model is part of available models
# 2. If available, append the model's name to an array
# 3. Then, append the model's corresponding pipeline to the requested_pipelines array
# *Note* requested_models_filtered & requested_pipelines should be same length
requested_models_filtered = []
for model in requested_models:
    if model in available_models:
        requested_models_filtered.append(model)
        requested_pipelines.append(available_pipelines[available_models.index(model)])

# Step 3: Test Selected Models, Save results in txt file, Output Ranking
output_text_file = open('output.txt', 'w')

for index in range(len(requested_pipelines)):
    pipeline = requested_pipelines[index]

    pipeline.fit(X_train, Y_train)
    Y_predict = pipeline.predict(X_test)

    report = classification_report(Y_test, Y_predict)
    matrix = confusion_matrix(Y_test, Y_predict)

    output_text_file.write("Model: %s\n" % requested_models_filtered[index])
    output_text_file.write("Classification Report: %s\n" % report)
    output_text_file.write("Confusion Matrix: %s\n" % matrix)
    count = count + 1

output_text_file.close()

# Step 4: Conversion to CoreML model
