import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, MACCSkeys, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D

def generate_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr, mol
    return np.zeros((167,)), None

def plot_waterfall_shap(model, instance_array, background_df, feature_names):
    explainer = shap.Explainer(model, background_df)
    shap_values = explainer(instance_array)
    shap.plots.waterfall(shap_values[0], max_display=10)

