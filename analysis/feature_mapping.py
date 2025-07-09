import numpy as np
import pandas as pd
import shap
from joblib import load
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps
import io
from PIL import Image

from featurisation.feature_transformers import get_data


def get_atom_contribution(mol, fpSize, maxPath):
    atom_contribution = [0] * mol.GetNumAtoms()  # init list of length number of atoms in mol

    fpgen = AllChem.GetRDKitFPGenerator(fpSize=fpSize, maxPath=maxPath)
    ao = AllChem.AdditionalOutput()
    ao.CollectBitPaths()
    fingerprint = fpgen.GetFingerprint(mol, additionalOutput=ao)

    bits = fingerprint.GetOnBits()
    shapsum = 0
    for bit in bits:  # bit name vs index
        if bit not in feature_names:  # remove bits removed in variancethreshold step
            continue
        bit_idx = feature_names.index(bit)

        envs = ao.GetBitPaths()[bit]  # lists of bond indices for each env
        atoms = set()
        for env in envs:  # iterate over all collided envs - these could be multiple occurences of same substruc or different substrucs
            for bidx in env:  # iterate over bond indices
                atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
                atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())

        for atom in atoms:
            atom_contribution[atom] += shap_values[bit_idx] / len(atoms)
        shapsum += shap_values[bit_idx]

    molsum = sum([x for x in atom_contribution])

    atom_contribution = [x[0] for x in atom_contribution]

    return atom_contribution


def plot_map(mol, atom_contribution):
    d = Draw.MolDraw2DCairo(1000, 1000)

    # Coloring atoms of element 0 to 100 black
    d.drawOptions().updateAtomPalette({i: (0, 0, 0, 1) for i in range(100)})
    cps = Draw.ContourParams()
    cps.fillGrid = True
    cps.gridResolution = 0.02
    cps.extraGridPadding = 1.2
    coolwarm = ((0.017, 0.50, 0.850, 0.5),
                (1, 1, 1, 0.5),
                (1, 0.25, 0.0, 0.5)
                )

    cps.setColourMap(coolwarm)

    SimilarityMaps.GetSimilarityMapFromWeights(mol, atom_contribution, contourLines=5,
                                               draw2d=d, contour_params=cps, sigma_f=0.4)

    d.FinishDrawing()
    return d


def show_png(data):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img


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

pred_scores = pd.read_csv("LOO_predictions.csv")['pred']

for i in range(len(mols_new)):
    #get mol
    mol = mols_new[i]

    # SHAP
    fp = best_model[:-1].transform(mol)
    model = best_model.named_steps['model']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(fp, approximate=False, check_additivity=False)
    shap_values = shap_values[:,:,1].T

    # decompose shap and plot
    atom_contribution = get_atom_contribution(mol, fpSize, maxPath)
    res = plot_map(mol, atom_contribution)
    img = show_png(res.GetDrawingText())
    img.show()
