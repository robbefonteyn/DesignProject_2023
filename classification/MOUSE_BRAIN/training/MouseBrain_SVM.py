print("Mouse Brain Model Training")


import numpy as np
import scanpy as sp
import pandas as pd
import pickle

import sklearn as sk
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# Set seed
seed = 2023  # DO NOT CHANGE!

# Print library versions
print(f"sklearn version: {sk.__version__}")
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"scanpy version: {sp.__version__}")

BRAIN = sp.read_h5ad("./MouseData/mouse_brain.h5ad")

# Filtering + Train/Test Split
cutoff = 0.001

cell_types, type_numbers = np.unique(BRAIN.obs['predicted.id'], return_counts=True)
bad_types = cell_types[type_numbers / len(BRAIN.obs['predicted.id'])<cutoff]

bad_types_mask = np.invert(np.isin(BRAIN.obs['predicted.id'], bad_types))
X = BRAIN.X[bad_types_mask]
Y = BRAIN.obs['predicted.id'][bad_types_mask]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)

n_train = X_train.shape[0]
n_test = X_test.shape[0]
print(f"{n_train} train samples\n{n_test} test samples\n{n_train/(n_train+n_test)*100:.2f}% of samples used for training")

# Model training
n_folds = 5 

cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state = seed)

param_grid = {
    'C': [10**x for x in np.arange(-5, 0.5, 0.5)]
}

lsvc = svm.LinearSVC()
gridsearch = GridSearchCV(lsvc, param_grid, n_jobs=-1, cv=cv, verbose=5, 
                          return_train_score=True)

gridsearch.fit(X_train, Y_train)

results = gridsearch.cv_results_
train_score = results['mean_train_score'][gridsearch.best_index_]
validation_score = results['mean_test_score'][gridsearch.best_index_]

print(f'Average training accuracy across folds: {train_score:.3}')
print(f'Average validation accuracy across folds: {validation_score:.3}')
print(f'Best hyperparams: {gridsearch.best_params_}')

# Save best model
with open('models/MouseBrain_lsvm.pkl', 'wb') as f:
    pickle.dump(gridsearch.best_estimator_, f)

with open('models/MouseBrain_lsvm.pkl', 'rb') as f:
    best_model = pickle.load(f)
    
test_predictions =  best_model.predict(X_test)
train_predictions =  best_model.predict(X_train)

from sklearn.metrics import accuracy_score

print(f"Train accuracy: {accuracy_score(Y_train, train_predictions)}")
print(f"Test accuracy: {accuracy_score(Y_test, test_predictions)}")


from sklearn.metrics import balanced_accuracy_score

print(f"Balanced Train Accuracy: {balanced_accuracy_score(Y_train, train_predictions)}")
print(f"Balanced Test Accuracy: {balanced_accuracy_score(Y_test, test_predictions)}")
