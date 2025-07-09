import io

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image, ImageDraw, ImageFont
from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

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
    ax2.barh(np.arange(0, num_display), feature_ratio_order[::-1], alpha=0.6, color='black')

    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1) + 1, 5))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max()), 0.5))
    ax1.set_xlabel('Cumulative %', fontsize=16)
    ax2.set_xlabel('Composition %', fontsize=16)
    ax1.set_ylabel('Fingerprint Bit', fontsize=16)
    ax1.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="both", labelsize=12)
    plt.ylim(-1, len(column_list))
    plt.tight_layout()
    plt.show()


def get_collisions(bits):
    bit_data = []
    fragments = []
    for bit in bits:
        bit_fragments = []
        for n in range(len(df_new['Trimer'])):
            mol = Chem.MolFromSmiles(df_new['Trimer'][n])
            fpgen = AllChem.GetRDKitFPGenerator(fpSize=fpSize, maxPath=maxPath)
            ao = AllChem.AdditionalOutput()
            ao.CollectBitPaths()
            fingerprint = fpgen.GetFingerprint(mol, additionalOutput=ao)
            bi = ao.GetBitPaths()
            if bit in list(fingerprint.GetOnBits()):
                envs = bi[bit]
                atoms = set()
                for bidx in envs[0]:
                    atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
                    atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
                fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), bondsToUse=envs[0])
                if fragment not in bit_fragments:
                    bit_fragments.append(fragment)
                    bit_data.append((mol, bit, bi))
                    fragments.append(fragment)

    return bit_data, fragments


def get_images(bit_data):
    drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    drawOptions.prepareMolsBeforeDrawing = False
    images = []
    for mol, bit, bi in bit_data:
        svg = Draw.DrawRDKitBit(mol, bit, bi, drawOptions=drawOptions)
        img = svg_to_pil(svg)
        images.append(img)
    return images


def get_mol_svgs(bit_data):
    images = []
    for i, (mol, bit, bi) in enumerate(bit_data):
        drawer = rdMolDraw2D.MolDraw2DSVG(1200, 1200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()
        images.append(img)
    return images


def svg_to_pil(svg_string):
    png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
    return Image.open(io.BytesIO(png_data))


def make_image_grid(images, bit_data):
    # Determine grid dimensions
    bits = []
    for i, (mol, bit, bi) in enumerate(bit_data):
        bits.append(bit)

    rows = len(set(bits))  # row for each bit
    cols = max(bits.count(bit) for bit in bits)

    # Calculate individual image dimensions
    max_width = max(img.width for img in images) + 20
    max_height = max(img.height for img in images)

    # Create a blank grid image
    grid_width = cols * max_width
    grid_height = rows * max_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Paste each image into the grid
    for i, img in enumerate(images):
        x = (bits[:i + 1].count(bits[i]) - 1) * max_width + 10
        y = (len(set(bits[:i + 1])) - 1) * max_height
        grid_image.paste(img, (x, y))

    # Optionally, add labels to the grid
    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.load_default(size=20)
    for i, (mol, bit, bi) in enumerate(bit_data):
        x = (bits[:i + 1].count(bits[i]) - 1) * max_width
        y = (len(set(bits[:i + 1])) - 1) * max_height
        draw.text((x, y), f"Bit {bit}", fill='black', font=font)

    # bbox = grid_image.getbbox()
    # grid_image = grid_image.crop(bbox)

    plt.imshow(grid_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return grid_image


# GET TARGET DATA ----------------------------------------------------
# get assay data
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
X = best_model[:-1].transform(mols_new)
model = best_model.named_steps['model']

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X, approximate=False, check_additivity=False)

# Shap plot
shap.summary_plot(shap_values[:, :, 1], best_model[:-1].transform(mols_new), feature_names=feature_names,
                  color_bar_label='Bit Value')
plt.show()

# get list of top features
vals = np.abs(shap_values[:, :, 1]).mean(0)  # shap vals for all features
top_bits = feature_names_np[np.argsort(vals)][-9:][::-1]

# visualise
bit_data, fragments = get_collisions(top_bits)
images = get_images(bit_data)
grid_image = make_image_grid(images, bit_data)
svgs = get_mol_svgs(bit_data)

# Shap waterfall plot
make_shap_waterfall_plot(shap_values[:, :, 1], feature_names_np)  # get features from pipeline
