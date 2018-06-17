import coremltools

# Responsible for converting selected ML models + datasets => CoreML file
def convertToCoreML():


# Retrieve correct ML Models based on user query
#
# @params userQuery: Array containing strings corresponding to user requested ML Models
# @return requested_pipelines:
# @return models_filtered: Returning list of Pipeline objects containing requested ML models
def getMLModels(requested_models):
    svc = Pipeline([('clf', SVC())])
    knn = Pipeline([('clf', KNeighborsClassifier(n_neighbors=3))])
    decision_trees = Pipeline([('clf', DecisionTreeClassifier())])
    random_forest = Pipeline([('clf', RandomForestClassifier())])
    gradient_boosted = Pipeline([('clf', GradientBoostingClassifier())])

    available_models = ["svc", "knn", "decision-trees", "random-forest", "gradient-boosted"]
    available_pipelines = [svc, knn, decision_trees, random_forest, gradient_boosted]

    # 1. Check if requested model is part of available models
    # 2. If available, append model's name to return array
    # 3. Append model's corresponding pipeline to requested pipelines
    pipelines = []
    models_filtered = []
    for model in requested_models:
        model = model.strip('"')
        if (str(model) in available_models):
            models_filtered.append(model)
            index = available_models.index(model)
            pipelines.append(available_pipelines[index])

    return (pipelines, models_filtered)

if __name__ == '__main__':
    # Unit Tests
    print("Todo")
