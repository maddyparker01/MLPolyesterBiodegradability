import numpy as np
import pandas as pd
from featurisation.feature_transformers import get_data, get_fransen_data, getMolDescriptors, \
    FeatureMaskTransformer
from joblib import dump
from numpy import mean, std
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_validate, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline

# GET SOURCE DATA ----------------------------------------------------
mols, base_y = get_fransen_data()
features = getMolDescriptors().transform(mols)

# GET TARGET DATA ----------------------------------------------------
mols_new, y = get_data()
features_new = getMolDescriptors().transform(mols_new)

# REMOVE FEATURES ----------------------------------------------------
# Check if values in df2 are outside the range of df1
min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
out_of_range = (features_new < min_vals) | (features_new > max_vals)

# get list of descriptors to remove
all_descriptors = np.array([desc[0] for desc in Descriptors._descList])
descriptors_removed = all_descriptors[np.array([desc != 0 for desc in out_of_range.sum(axis=0)])]
print(descriptors_removed)

# START MODEL ----------------------------------------------------

# configure the cross-validation procedure
groups = pd.read_csv('groups.csv', header=None)
groups = np.array(groups).reshape(-1)
cv_inner = LeaveOneGroupOut()
cv_outer = LeaveOneGroupOut()

# define the model
pipeline = Pipeline([
    ('fingerprint', getMolDescriptors()),
    ('mask', FeatureMaskTransformer(descriptors_removed)),
    ('feature_selection', VarianceThreshold()),
    ('model', RandomForestClassifier(random_state=42))
])

# define search space
space = dict()
space['model__n_estimators'] = [100, 200, 500]
space['model__max_features'] = [2, 5, 10]
space['model__max_depth'] = [2, 4, 6]

# define search
search = RandomizedSearchCV(pipeline, space, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],  # n_jobs=-1,
                            cv=cv_inner, refit='accuracy', verbose=2)

# execute the nested cross-validation (for model evaluation)
scores = cross_validate(search, mols, base_y, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                        cv=cv_outer, n_jobs=-1, verbose=2, fit_params={'groups': groups}, groups=groups,
                        return_train_score=True)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores['test_accuracy']), std(scores['test_accuracy'])))
print('Precision: %.3f (%.3f)' % (mean(scores['test_precision']), std(scores['test_precision'])))
print('Recall: %.3f (%.3f)' % (mean(scores['test_recall']), std(scores['test_recall'])))
print('F1: %.3f (%.3f)' % (mean(scores['test_f1']), std(scores['test_f1'])))
print('ROC AUC: %.3f (%.3f)' % (mean(scores['test_roc_auc']), std(scores['test_roc_auc'])))

# Save the trained model to a file
final_model = search.fit(mols, base_y, groups=groups)
best_model = final_model.best_estimator_
model_filename = '../models/chain_step1.joblib'
dump(best_model, model_filename)
