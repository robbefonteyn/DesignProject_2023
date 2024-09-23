print("BMMC RNAseq Model Training RF")

import numpy as np
import scanpy as sp
import pandas as pd
import pickle

import sklearn as sk
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

seed = 2023 # DO NOT CHANGE!

print(f"sklearn version: {sk.__version__}")
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"scanpy version: {sp.__version__}")


BMMC = sp.read_h5ad("GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")

donors, donor_sample_count = np.unique(BMMC.obs['DonorID'], return_counts=True)

test_donors_idx = [19593, 28045, 28483]
test_filter = np.isin(BMMC.obs['DonorID'], test_donors_idx)
train_filter = np.invert(test_filter)

n_train = np.sum(train_filter)
n_test = np.sum(test_filter)


d =  BMMC.d =  BMMC.X
X_test = d[test_filter]
X_train = d[train_filter]

Y_test = BMMC.obs['cell_type'][test_filter]
Y_train = BMMC.obs['cell_type'][train_filter]

train_donors = BMMC.obs[train_filter]['DonorID']

print(f"The sparsity of the train matrix is {(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])) * 100:.2f}%")



n_folds = 5 # change?

cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state = seed)


param_grid = {
    'min_samples_leaf': [1, 3, 5, 7, 9]
}


rf = RandomForestClassifier()
gridsearch = GridSearchCV(rf,param_grid, n_jobs=-1, cv=cv, verbose=5, 
                          return_train_score=True)

gridsearch.fit(X_train, Y_train, groups=train_donors)



results = gridsearch.cv_results_
train_score = results['mean_train_score'][gridsearch.best_index_]
validation_score = results['mean_test_score'][gridsearch.best_index_]

print(f'Average training accuracy across folds: {train_score:.3}')
print(f'Average validation accuracy across folds: {validation_score:.3}')

print(f'Best hyperparams: {gridsearch.best_params_}')


# save best model
with open('RF_RNAseq_best_model.pkl','wb') as f:
    pickle.dump(gridsearch.best_estimator_,f)

    
# load model
with open('RF_RNAseq_best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
    
test_predictions =  best_model.predict(X_test)
train_predictions =  best_model.predict(X_train)


from sklearn.metrics import accuracy_score

print(f"Train accuracy: {accuracy_score(Y_train, train_predictions)}")
print(f"Test accuracy: {accuracy_score(Y_test, test_predictions)}")


from sklearn.metrics import balanced_accuracy_score

print(f"Balanced Train Accuracy: {balanced_accuracy_score(Y_train, train_predictions)}")
print(f"Balanced Test Accuracy: {balanced_accuracy_score(Y_test, test_predictions)}")
