import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.calibration import CalibrationDisplay

from featurisation.feature_transformers import get_data, inside_validity_domain

# GET TARGET DATA ----------------------------------------------------
df_new = pd.read_csv("../data/inhouse_data.csv")
mols_new, y = get_data()

# get model info
best_model = load('../models/inhouse.joblib')
fpSize = best_model.named_steps['fingerprint'].fpSize
maxPath = best_model.named_steps['fingerprint'].maxPath
variance_threshold_step = best_model.named_steps['feature_selection']
selected_features_mask = variance_threshold_step.get_support()
feature_names = [str(feature) for feature, selected in zip(range(fpSize), selected_features_mask) if selected]
feature_names = [int(bit) for bit in feature_names]
feature_names_np = np.array([int(bit) for bit in feature_names])

# ANALYSIS ----------------------------------------------------
# fingerprint generation
fpgen = AllChem.GetRDKitFPGenerator(fpSize=fpSize, maxPath=maxPath)
fps = []
for i in range(len(mols_new)):
    fp = fpgen.GetFingerprint(mols_new[i])
    fps.append(fp)

# get tanimoto similarity to 5 nearest neighbours
distance_scores = []
for i in range(len(fps)):
    target = fps[i]
    not_i = fps[:i] + fps[i + 1:]  # List of all but the i-th fingerprint

    i_scores = DataStructs.BulkTanimotoSimilarity(target, not_i)  # compare i to all-but i
    k_nearest = np.argsort(i_scores)[-5:]
    closest5 = [i_scores[i] for i in k_nearest]
    distance_scores.append(np.mean(closest5))

print('Mean 5NN similarity: ', np.mean(distance_scores))

# START MODEL ----------------------------------------------------
# define the model
"""pipeline = Pipeline([
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
search = RandomizedSearchCV(pipeline, space, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], n_jobs=1,
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), refit='accuracy',
                            verbose=2, random_state=42)

cv = LeaveOneOut()
X = mols_new
pred_scores = []
for fold, (train, test) in enumerate(cv.split(mols_new, y)):
    # fit model
    final_model = search.fit(X[train], y[train])
    best_model = final_model.best_estimator_

    # predict proba
    pred = best_model.predict_proba(X[test])[:, 1][0]
    # pred = random.rand()
    pred_scores.append(pred)

pd.DataFrame(data={"pred": pred_scores}).to_csv("LOO_predictions.csv")"""

pred_scores = pd.read_csv("LOO_predictions.csv")['pred']

scores = pd.DataFrame({'dist': distance_scores,
                       'pred': pred_scores,
                       'y': y})

# remove entries where test set is outside validity domain
scores = scores[[inside_validity_domain(mol) for mol in df_new['Trimer']]]

# group entries using the defined bins
bins = [0, 0.8, 0.9, 1]
scores['bin'] = pd.cut(scores['dist'], bins=bins, labels=False)

# plot results
markers = ["^", "s", "o"]
fig = plt.figure()
ax = plt.subplot()
for i in range(len(bins) - 1):
    binned_data = scores[scores['bin'] == i]
    disp = CalibrationDisplay.from_predictions(binned_data['y'], binned_data['pred'], n_bins=3,
                                               name=f"{bins[i]} - {bins[i + 1]}", marker=markers[i],
                                               ax=ax)

ax.legend(loc='upper left')
plt.show()
