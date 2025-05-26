import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D

def highlight_top_maccs_features(mol, shap_values, feature_names, top_n=10):
    if mol is None:
        raise ValueError("Invalid molecule provided.")

    # Identify top N features by absolute SHAP value
    top_indices = np.argsort(np.abs(shap_values[0].values))[-top_n:]

    # MACCS SMARTS patterns
    smarts_list = MACCSkeys.smartsPatts

    hit_atoms = set()
    for idx in top_indices:
        smarts = smarts_list[idx][0]
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            match = mol.GetSubstructMatch(patt)
            if match:
                hit_atoms.update(match)

    # Draw molecule with highlighted atoms
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    drawer.DrawMolecule(mol, highlightAtoms=list(hit_atoms))
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


def draw_molecule_with_shap_highlights(molecule, shap_values_array, maccs_features, top_n=10):
    """
    Highlight substructures on molecule corresponding to top_n MACCS fingerprint features
    with active bits ('1') and SHAP values coloring the atoms.
    
    Parameters:
    - molecule: RDKit Mol object
    - shap_values_array: numpy array of SHAP values for each feature
    - maccs_features: string or list representing MACCS fingerprint bits ('0'/'1')
    - top_n: number of top features to highlight based on absolute SHAP values
    
    Returns:
    - img: PNG image bytes for display or saving
    """
    MACCSsmartsPatts = MACCSkeys.smartsPatts  # tuple of SMARTS strings
    
    # Get indices of top_n absolute SHAP values
    sorted_idx = np.argsort(np.abs(shap_values_array))[::-1]  # descending order
    top_n_idx = sorted_idx[:top_n]

    highlight_atoms = []
    highlight_bonds = []
    highlight_colors = {}

    for feature in top_n_idx:
        # Check if MACCS feature is active
        if maccs_features[feature] == '1':
            smarts = MACCSsmartsPatts[feature]  # directly access SMARTS string
            substructure = Chem.MolFromSmarts(smarts)
            if substructure:
                match = molecule.GetSubstructMatch(substructure)
                if match:
                    # Add matched atoms
                    highlight_atoms.extend(match)
                    # Color atoms red for positive SHAP
                    color = (1.0, 0.6, 0.6) if shap_values_array[feature] > 0 else (0.6, 0.6, 1.0)
                    for atom in match:
                        highlight_colors[atom] = color
                    
                    # Optionally highlight bonds connected to these atoms
                    # Uncomment if you want bond highlights
                    # for bond in molecule.GetBonds():
                    #     if bond.GetBeginAtomIdx() in match or bond.GetEndAtomIdx() in match:
                    #         highlight_bonds.append(bond.GetIdx())
                    #         highlight_colors[bond.GetIdx()] = color

    # Remove duplicate atoms and bonds
    highlight_atoms = list(set(highlight_atoms))
    # highlight_bonds = list(set(highlight_bonds))


