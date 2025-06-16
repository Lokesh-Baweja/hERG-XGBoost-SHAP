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



def plot_waterfall_shap1(model, instance_array, background_df, feature_names):
    explainer = shap.Explainer(model, background_df)
    shap_values = explainer(instance_array)  # SHAP values for one or more samples

    # --- Get the SHAP value array for the first sample
    shap_values_array = shap_values[0].values         # SHAP values (1D numpy array)
    input_bits = shap_values[0].data                  # MACCS bit vector (same shape)
    
    # Optional: convert to numpy array if needed
    input_bits = np.array(input_bits)

    # --- Filter: SHAP > 0 and bit == 1
    positive_bit_mask = (shap_values_array > 0) & (input_bits == 1)
    filtered_shap = shap_values_array[positive_bit_mask]
    filtered_names = np.array(feature_names)[positive_bit_mask]

    # Sort and select top N
    sorted_idx = np.argsort(filtered_shap)[::-1]
    top_n = min(10, len(sorted_idx))
    top_feats = filtered_names[sorted_idx[:top_n]]
    top_vals = filtered_shap[sorted_idx[:top_n]]
    top_feat_numbers = [int(f.split("_")[1]) for f in top_feats]
    # Print results
    print(f"Top {top_n} SHAP-positive features with bit = 1:")
    for f, v in zip(top_feats, top_vals):
        print(f"{f}: {v:.4f} ↑")

    # --- Plot full SHAP waterfall
    #shap.plots.waterfall(shap_values[0], max_display=10)
    return {
    "shap_values_array": shap_values_array,
    "sorted_idx": sorted_idx,
    "top_features": top_feat_numbers,
    }
