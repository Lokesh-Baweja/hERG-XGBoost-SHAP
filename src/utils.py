import shap
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

def get_shap_explanation(model, X_instance, explainer=None):
    if explainer is None:
        explainer = shap.Explainer(model)
    shap_values = explainer(X_instance)
    return shap_values

def plot_molecule_and_waterfall(smiles, shap_values, feature_names=None, top_n=10):
    mol = Chem.MolFromSmiles(smiles)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Draw molecule
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        axs[0].imshow(img)
        axs[0].axis('off')
        axs[0].set_title("Molecule Structure")
    else:
        axs[0].text(0.5, 0.5, "Invalid SMILES", ha='center')
        axs[0].axis('off')

    # SHAP waterfall
    shap.plots.waterfall(shap_values[0], max_display=top_n, show=False, ax=axs[1])
    axs[1].set_title("SHAP Waterfall Plot")

    plt.tight_layout()
    plt.show()
