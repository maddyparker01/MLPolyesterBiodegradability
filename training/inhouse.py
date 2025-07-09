from featurisation.feature_transformers import get_data, getFingerprints
from joblib import dump
from numpy import mean, std
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# GET TARGET DATA ----------------------------------------------------
mols_new, y = get_data()

# START MODEL ----------------------------------------------------
# define the model
pipeline = Pipeline([
    ('fingerprint', getFingerprints()),
    ('feature_selection', VarianceThreshold()),
    ('model', RandomForestClassifier(random_state=42))
])

# define search space
space = dict()
space['fingerprint__fpSize'] = [2048, 4096, 8192]
space['fingerprint__maxPath'] = [4, 5, 6]
space['model__n_estimators'] = [100, 200, 500]
space['model__max_features'] = [2, 5, 10]
space['model__max_depth'] = [2, 4, 6]

# define search
search = RandomizedSearchCV(pipeline, space, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],  n_jobs=1,
                            cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True), refit='accuracy', verbose=2, random_state=42)

# execute the nested cross-validation (for model evaluation)
scores = cross_validate(search, mols_new, y, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                        cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True), n_jobs=-1, verbose=2, return_train_score=True)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores['test_accuracy']), std(scores['test_accuracy'])))
print('Precision: %.3f (%.3f)' % (mean(scores['test_precision']), std(scores['test_precision'])))
print('Recall: %.3f (%.3f)' % (mean(scores['test_recall']), std(scores['test_recall'])))
print('F1: %.3f (%.3f)' % (mean(scores['test_f1']), std(scores['test_f1'])))
print('ROC AUC: %.3f (%.3f)' % (mean(scores['test_roc_auc']), std(scores['test_roc_auc'])))

# Save the trained model to a file
final_model = search.fit(mols_new, y)
best_model = final_model.best_estimator_
model_filename = '../models/inhouse.joblib'
dump(best_model, model_filename)
