from featurisation.feature_transformers import get_data, getMolDescriptors, getFingerprints
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import scikit_posthocs as sp
from numpy import mean, std
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns

# GET TARGET DATA ----------------------------------------------------
mols_new, y = get_data()

# CONFIG MODEL ----------------------------------------------------
# define structural representations and hyperparams
transformers = {
    'RDF': {
        'transformer': getFingerprints(),
        'space': {
            'fingerprint__fpSize': [2048, 4096, 8192],
            'fingerprint__maxPath': [4, 5, 6]
        }
    },
    'RDD': {
        'transformer': getMolDescriptors(),
        'space': {}
    }
}

# define algorithms and hyperparams
algorithms = {
    'RF': {
        'model': RandomForestClassifier(random_state=42),
        'space': {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [2, 4, 6],
            'model__max_features': [2, 5, 10],
        }
    },
    'NN': {
        'model': MLPClassifier(random_state=42),
        'space': {
            'model__hidden_layer_sizes': [(8,), (16,), (8, 4), (16, 8)],
            'model__max_iter': [500, 1000, 1500]
        }
    }
}

# initialise df for friedman test
cv_results = pd.DataFrame({'cv cycle': range(1, 11)})
cv_results_allmethods = pd.DataFrame()


# START MODEL ----------------------------------------------------
# Loop over transformers
for transformer_name, transformer_info in transformers.items():

    # Loop over algorithms
    for algo_name, algo_config in algorithms.items():

        # Build the pipeline
        pipeline_steps = [
            ('fingerprint', transformer_info['transformer']),  # Replace with actual transformer
            ('feature_selection', VarianceThreshold()),  # Removes low-variance features
        ]

        # Conditionally set the scaler
        if algo_name == 'NN':
            scaler = MinMaxScaler()
            pipeline_steps.append(('scaler', scaler))

        pipeline_steps.append(('model', algo_config['model']))
        pipeline = Pipeline(pipeline_steps)

        # define search space
        space = {**transformer_info['space'], **algo_config['space']}

        # define search
        search = RandomizedSearchCV(pipeline, space, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],  #n_jobs=1,
                                    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42), refit='accuracy', verbose=2)

        # execute the nested cross-validation (for model evaluation)
        scores = cross_validate(search, mols_new, y, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42), n_jobs=-1, verbose=2, return_train_score=True)

        # report performance
        print(f"Results from {algo_name} with {transformer_name}...")
        print('Accuracy: %.3f (%.3f)' % (mean(scores['test_accuracy']), std(scores['test_accuracy'])))
        print('Precision: %.3f (%.3f)' % (mean(scores['test_precision']), std(scores['test_precision'])))
        print('Recall: %.3f (%.3f)' % (mean(scores['test_recall']), std(scores['test_recall'])))
        print('F1: %.3f (%.3f)' % (mean(scores['test_f1']), std(scores['test_f1'])))
        print('ROC AUC: %.3f (%.3f)' % (mean(scores['test_roc_auc']), std(scores['test_roc_auc'])))

        # save results
        #cv_results[f'{algo_name} x {transformer_name}'] = scores['test_accuracy']
        cv_results['method'] = f'{algo_name} x {transformer_name}'
        cv_results['accuracy'] = scores['test_accuracy']
        cv_results['precision'] = scores['test_precision']
        cv_results['recall'] = scores['test_recall']
        cv_results['F1'] = scores['test_f1']
        cv_results['ROC AUC'] = scores['test_roc_auc']
        cv_results_allmethods = pd.concat([cv_results_allmethods, cv_results], ignore_index=True)


# PLOT RESULTS ----------------------------------------------------

sns.set_context('notebook')
sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
sns.set_style('white')
figure, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16, 8))

for i, stat in enumerate(['accuracy', 'ROC AUC']):
    friedman = pg.friedman(cv_results_allmethods, dv=stat, within="method", subject="cv cycle")['p-unc'].values[0]
    print(f"p={friedman:.03f}")

    ax = sns.boxplot(y=stat, x="method", ax=axes[i], data=cv_results_allmethods, palette=['lightblue', 'lightpink', 'lightgreen', 'lightsalmon'], linewidth=3, linecolor='k')
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.set_xlabel("")
    ax.set_ylabel(stat, fontsize=20)
    ax.set_ylim(0.5, 1)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=20)
plt.tight_layout()
plt.show()

# posthoc conover friedman
for i, stat in enumerate(['accuracy', 'ROC AUC']):
    pc = sp.posthoc_conover_friedman(cv_results_allmethods, y_col=stat, group_col="method", block_col="cv cycle",
                                     p_adjust="holm", melted=True)
    print(stat.upper().replace("_", " "))
    print(pc)
    print()
