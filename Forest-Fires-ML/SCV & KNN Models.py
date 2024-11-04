import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, validation_curve, learning_curve, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,  classification_report
from sys import exit

# ML models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

## Define definitions
# Visualize dataframe
def visualize(dataframe):
    print(dataframe)

# Grid search for best parameters in data
def searchParameters(model, paramGrid, X_train, y_train, cv):
    # Perform grid search
    grid_search = GridSearchCV(model, paramGrid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Return best parameters found
    return grid_search.best_params_

# Create confusion matrix
def cConfusionMatrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for SVC")
    #plt.title("Confusion Matrix for KNN")
    plt.show(block=False)

# Create validation curve
def cValidationCurve(X_train, y_train, model):
    #SVC
    param_range = np.logspace(-3, 2, 10)
    param_name = "C"
    # KNN
    #param_range = np.arange(1, 31)
    #param_name = "n_neighbors"

    # Calculate validation curve
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=-1)
    
    # Calculate mean scores
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores_mean, label='Training score', marker='o')
    plt.plot(param_range, test_scores_mean, label='Testing score', marker='o')
    plt.xlabel(f'Parameter {param_name}')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Curve for SVC')
    #plt.title(f'Validation Curve for KNN')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

# Create learning curve
def cLearningCurve(X_train, y_train, model):
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy')

    # Calculate mean scores
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training score', marker='o')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score', marker='o')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve for SVC')
    #plt.title('Learning Curve for KNN')
    plt.legend()
    plt.grid(True)
    plt.show()

## Import, manipulate, split data
# Import forest fire dataset
algerian_forest_fires = pd.read_csv("Algerian_forest_fires_dataset_Merge.csv")

# Drop columns
algerian_forest_fires = algerian_forest_fires.drop(columns=['day', 'month', 'year', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'])

# Map and replace "classes" to numeric values
class_mapping = {'not fire': 0, 'fire': 1}
algerian_forest_fires['classes'] = algerian_forest_fires['classes'].str.strip()
algerian_forest_fires['classes'] = algerian_forest_fires['classes'].map(class_mapping)

# Shuffle the dataset
shuffled_index = list(algerian_forest_fires.index)
random.shuffle(shuffled_index)
algerian_forest_fires = algerian_forest_fires.loc[shuffled_index]

# Split features and labels
X_data = algerian_forest_fires.drop(columns=['classes'])
y_data = algerian_forest_fires['classes']

# Datasplit
split = int(0.30 * len(algerian_forest_fires))
testingSet = algerian_forest_fires[:split]
trainingSet = algerian_forest_fires[split:]

# Divide training data into sets, x and y, where x values is data without labels, and y values are the labels
X_training = trainingSet.drop(columns=['classes'])
y_training = trainingSet['classes']
# Divide testing data into sets, x and y, where x values is data without labels, and y values are the labels
X_testing = testingSet.drop(columns=['classes'])
y_testing = testingSet['classes']

## Model
# Define parameter grids
paramGridSVC = {'C': [0.1, 1, 10, 100], # Regularisation parameter, controls the trade-off between achieving a low error on the training data and minimizing the norm of the weights
               'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], # Specifies the kernel type to be used in the algorithm
               #'gamma': [0.01, 0.1, 1, 10], # Defines how far the influence of a single training example reaches
               'gamma': [0.01],
               'shrinking': [True, False]} # Whether to use the shrinking heuristic to speed up the optimization process.

paramGridKNN = {'n_neighbors': [3, 5, 7, 9, 11], # Number of neighbors to use
               'weights': ['uniform', 'distance'], # Weight function used in prediction
               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], # Algorithm used to compute the nearest neighbors
               'metric': ['euclidean', 'manhattan', 'minkowski']} # Distance metric to use for the tree

# ML modle setup
model = SVC()
#model = KNeighborsClassifier()

# Parameter tuning
parameters = searchParameters(model, paramGridSVC, X_data, y_data, 2)
#parameters = searchParameters(model, paramGridKNN, X_data, y_data, 2)

# Output best parameters
print("Best parameters: " + str(parameters))

# Define model with best parameters
model = SVC(**parameters)
#model = KNeighborsClassifier(**parameters)

# Train model
model.fit(X_training, y_training)

# Make predictions
y_prediction = model.predict(X_testing)

# Calculate accuracy
accuracy = accuracy_score(y_testing, y_prediction)

# Perform cross-validation
cv_results = cross_validate(model, X_training, y_training, cv=5, return_train_score=True, scoring='accuracy')

# Calculate the mean testing and training accuracy
mean_train_accuracy = cv_results['train_score'].mean()
std_train_accuracy = cv_results['train_score'].std()

mean_test_accuracy = cv_results['test_score'].mean()
std_test_accuracy = cv_results['test_score'].std()

# Count labels
correct = 0
incorrect = 0
for actual, predicted in zip(y_testing, y_prediction):
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

# Return results
print("\nForrest fire SVC prediction model results")
#print("\nForrest fire KNN prediction model results")
print(f"Testing accuracy: ", round(accuracy*100, 2), "%", sep="")
train_pred = model.predict(X_training)
train_acc = accuracy_score(y_training, train_pred)
print(f'Training accuracy: {train_acc:.2%}')
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print("\nCross-validation results:")
print(f"Mean Testing Accuracy: {mean_test_accuracy * 100:.2f}%")
print(f"Standard Deviation of Testing Accuracy: {std_test_accuracy:.2f}")
print(f"Mean Training Accuracy: {mean_train_accuracy * 100:.2f}%")
print(f"Standard Deviation of Training Accuracy: {std_train_accuracy:.2f}")

#Output Diagrams
# Confusion matrix
class_names = ['not fire', 'fire']
cConfusionMatrix(y_testing, y_prediction, class_names)

# Validation curve
cValidationCurve(X_training, y_training, model)

# Learning curve
cLearningCurve(X_training, y_training, model)

print("\n\n")
print(classification_report(y_testing, y_prediction, target_names=class_names))
