import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import load
from rdkit.Chem import Descriptors

from featurisation.feature_transformers import get_data


def make_shap_waterfall_plot(shap_values, column_list, num_display=20):
    feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    column_list = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order = feature_ratio[np.argsort(feature_ratio)[::-1]]
    cum_sum = np.cumsum(feature_ratio_order)

    # get top features
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]

    num_height = 0
    if (num_display >= 20) & (len(column_list) >= 20):
        num_height = (len(column_list) - 20) * 0.4

    fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
    ax1.plot(cum_sum[::-1], column_list[::-1].astype(str), c='black', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(np.arange(0,num_display), feature_ratio_order[::-1],  alpha=0.6, color='black')

    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1)+1, 5))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max()), 0.5))
    ax1.set_xlabel('Cumulative %', fontsize=16)
    ax2.set_xlabel('Composition %', fontsize=16)
    ax1.set_ylabel('Fingerprint Bit', fontsize=16)
    ax1.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="both", labelsize=12)
    plt.ylim(-1, len(column_list))
    plt.tight_layout()
    plt.show()


# GET TARGET DATA ----------------------------------------------------
# get assay data
df_new = pd.read_csv("../data/inhouse_data.csv")
mols_new, y = get_data()

# get model info
best_model = load('../models/inhouse_descriptors.joblib')
variance_threshold_step = best_model.named_steps['feature_selection']
selected_features_mask = variance_threshold_step.get_support()
feature_names = [str(feature) for feature, selected in zip([item[0] for item in Descriptors._descList], selected_features_mask) if selected]
feature_names_np = np.array(feature_names)


# ANALYSIS ----------------------------------------------------
X = best_model[:-1].transform(mols_new)
model = best_model.named_steps['model']

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X, approximate=False, check_additivity=False)

# Shap plot
shap.summary_plot(shap_values[:,:,1], best_model[:-1].transform(mols_new), feature_names=feature_names, color_bar_label='Feature Value')
plt.show()

# Shap waterfall plot
make_shap_waterfall_plot(shap_values[:,:,1], feature_names_np) #get features from pipeline
