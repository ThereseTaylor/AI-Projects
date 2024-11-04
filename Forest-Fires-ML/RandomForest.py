# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, validation_curve, cross_val_score, GridSearchCV, learning_curve
from scipy.stats import randint

# Tree Visualisation
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

#import and process data
data = pd.read_csv('dataset.csv')
data = data.dropna()
data.isnull().sum()
df = pd.DataFrame(data)
df['Classes'] = df['Classes'].str.strip()
df['Classes'] = df['Classes'].replace({'fire': 1, 'not fire': 0})

y = df['Classes']
x = df.drop(['Classes', 'day', 'month', 'year', ' RH', 'Ws','FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'], axis=1)
print(x)

#Divide the dataset 80/20 for training/testing
trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2, random_state=101)

#Train the model
rf_Model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=12, min_samples_leaf=1, random_state= 101)
rf_Model.fit(trainX, trainY)
# Print the train accuracy scores
train_pred = rf_Model.predict(trainX)
train_acc = accuracy_score(trainY, train_pred)
print(f'Training set accuracy: {train_acc}')
# Print the test accuracy scores
test_pred = rf_Model.predict(testX)
test_acc = accuracy_score(testY, test_pred)
print(f'Test set accuracy: {test_acc}')

#Show the classification report
print('\nClassification Report:')
print(classification_report(testY, test_pred))

#Show the feature importance diagram
features = x.columns
importances = rf_Model.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Generate confusion matrix
cm = confusion_matrix(testY, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not fire', 'fire'])
# Plot confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=102),
    x, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy"
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.title("Learning Curve with RandomForest")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.plot(train_sizes, train_scores_mean, label="Training score", color="darkorange", lw=2)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=2)
plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="navy", lw=2)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=2)
plt.legend(loc="best")
plt.show()

""" rf_Model = RandomForestClassifier(random_state=101)

# Define hyperparameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 15],
    'min_samples_leaf': [1, 2]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_Model, param_grid, cv=5)
grid_search.fit(trainX, trainY)

# Get the best model from grid search
best_rf_Model = grid_search.best_estimator_
print(best_rf_Model)

train_cv_scores = cross_val_score(best_rf_Model, trainX, trainY, cv=5)
print("Cross-Validation Scores:", train_cv_scores)
print("Mean CV Score:", train_cv_scores.mean())

test_cv_scores = cross_val_score(best_rf_Model, testX, testY, cv=5)
print("Cross-Validation Scores:", test_cv_scores)
print("Mean CV Score:", test_cv_scores.mean())

# Evaluate on training set
train_pred = best_rf_Model.predict(trainX)
train_acc = accuracy_score(trainY, train_pred)
print(f'Training set accuracy: {train_acc}')

# Evaluate on test set
test_pred = best_rf_Model.predict(testX)
test_acc = accuracy_score(testY, test_pred)
test_pres = precision_score(testY, test_pred)
print(f'Test set accuracy: {test_acc}')
 """