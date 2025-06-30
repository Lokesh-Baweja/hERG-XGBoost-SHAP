import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Only add a handler if one doesn't exist to prevent duplicate logs in Jupyter
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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

def draw_molecule_with_shap_highlights1(molecule, shap_values_array, maccs_features, top_n=10):
    """
    Highlight substructures on molecule corresponding to top_n MACCS fingerprint features
    with SHAP values > 0 and feature presence = 1.
    
    Parameters:
    - molecule: RDKit Mol object
    - shap_values_array: numpy array of SHAP values for each feature
    - maccs_features: numpy array of MACCS fingerprint bits (0/1)
    - top_n: number of top features to highlight
    
    Returns:
    - img: PNG image bytes
    """
    MACCSsmartsPatts = MACCSkeys.smartsPatts  # tuple of SMARTS strings (length 167)

    # --- Filter features: SHAP > 0 and bit is set (1)
    valid_mask = (shap_values_array > 0) & (maccs_features == 1)
    valid_indices = np.where(valid_mask)[0]
    valid_shap_values = shap_values_array[valid_indices]

    # --- Sort valid SHAP values in descending order
    sorted_idx = valid_indices[np.argsort(valid_shap_values)[::-1]]
    top_n_idx = sorted_idx[:top_n]

    highlight_atoms = []
    highlight_colors = {}
    for feature_idx in top_n_idx:
        print (feature_idx)
        smarts = MACCSsmartsPatts[feature_idx]
        if smarts:
            substructure = Chem.MolFromSmarts(smarts)
            if substructure:
                match = molecule.GetSubstructMatch(substructure)
                if match:
                    color = (1.0, 0.6, 0.6)  # red for positive SHAP
                    highlight_atoms.extend(match)
                    for atom in match:
                        highlight_colors[atom] = color

    # Remove duplicates
    highlight_atoms = list(set(highlight_atoms))

    # --- Draw the molecule with highlights
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    drawer.DrawMolecule(molecule, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()



def draw_molecule_with_shap_highlights2(molecule, shap_values_array, maccs_features, top_n_idx, top_n=10):
    """
    Highlight substructures on molecule corresponding to top_n MACCS fingerprint features
    with SHAP values > 0 and active bits (1).
    
    Returns:
    - img_bytes: PNG image bytes
    """
    MACCSsmartsPatts = MACCSkeys.smartsPatts  # tuple of (SMARTS,) strings


    highlight_atoms = []
    highlight_bonds = []
    highlight_colors = {}

    for feature in top_n_idx:
        smarts = MACCSsmartsPatts[feature][0]
        logger.debug(f"Feature {feature} SMARTS: {smarts}")
        substructure = Chem.MolFromSmarts(smarts)
        if substructure:
            match = molecule.GetSubstructMatch(substructure)
            logger.debug(f"Match for feature {feature}: {match}")
            if not match:
                continue
            highlight_atoms.extend(match)
            for atom in match:
                highlight_colors[atom] = (1.0, 0.6, 0.6)

            for bond in molecule.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if begin in match and end in match:
                    highlight_bonds.append(bond.GetIdx())
                    highlight_colors[bond.GetIdx()] = (1.0, 0.6, 0.6)

    highlight_atoms = list(set(highlight_atoms))
    highlight_bonds = list(set(highlight_bonds))

    logger.debug(f"Highlight atoms: {highlight_atoms}")
    logger.debug(f"Highlight bonds: {highlight_bonds}")
    logger.debug(f"Highlight colors: {highlight_colors}")

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.drawOptions().useBWAtomPalette()
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=highlight_colors,
        highlightBondColors=highlight_colors
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()




def draw_single_maccs_feature(mol, atom_indices, color=(1.0, 0.2, 0.2)):
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.drawOptions().highlightRadius = 0.6
    drawer.drawOptions().fillHighlights = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atom_indices,
        highlightAtomColors={idx: color for idx in atom_indices}
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(BytesIO(png))

def draw_molecule_with_shap_highlights3(molecule, shap_values_array, maccs_features, top_n=10, n_cols=5):
    """
    Draws a grid of 2D molecule images, each highlighting one of the top_n MACCS features with highest SHAP values.
    
    Parameters:
    - molecule: RDKit Mol object
    - shap_values_array: Array of SHAP values corresponding to MACCS features
    - maccs_features: Binary MACCS array (1 for active, 0 for inactive)
    - top_n: Number of top features to highlight (default = 10)
    - n_cols: Number of columns in the subplot grid
    
    Returns:
    - Displays a matplotlib grid of highlighted molecules
    """
    AllChem.Compute2DCoords(molecule)
    MACCSsmartsPatts = smartsPatts

    # Step 1: filter for valid SHAP > 0 and bit is 1
    valid_indices = [
        idx for idx in range(len(shap_values_array))
        if shap_values_array[idx] > 0 and maccs_features[idx] == 1
    ]

    if not valid_indices:
        print("No valid features with SHAP > 0 and fingerprint bit == 1")
        return

    # Step 2: sort and select top_n
    sorted_idx = sorted(valid_indices, key=lambda i: shap_values_array[i], reverse=True)[:top_n]

    # Step 3: prepare colors
    cmap = plt.get_cmap("YlOrRd")
    norm = plt.Normalize(vmin=0, vmax=len(sorted_idx) - 1)

    images, titles = [], []

    for rank, idx in enumerate(sorted_idx):
        smarts = MACCSsmartsPatts[idx][0]
        patt = Chem.MolFromSmarts(smarts)
        if not patt:
            continue

        matches = molecule.GetSubstructMatches(patt)
        if not matches:
            continue

        atom_indices = list(matches[0])  # Show first match only
        color = cmap(norm(rank))[:3]     # RGB color

        img = draw_single_maccs_feature(molecule, atom_indices, color=color)
        images.append(img)
        titles.append(f"MACCS {idx}, SHAP: {shap_values_array[idx]:.2f}")

    # Step 4: Plot grid
    n = len(images)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axs = axs.flatten()

    for ax in axs[n:]:
        ax.axis('off')

    for ax, img, title in zip(axs, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


    from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import pyplot as plt
import numpy as np
from io import BytesIO

def draw_molecule_with_shap_highlights4(molecule, shap_values_array, maccs_features, top_n_idx, top_n=10):
    """
    Highlights the top_n MACCS fingerprint substructures on the molecule,
    colored by SHAP importance (brightest = highest SHAP).
    
    Parameters:
    - molecule: RDKit Mol object
    - shap_values_array: Array of SHAP values (same length as MACCS bits)
    - maccs_features: Binary MACCS fingerprint (1 = bit on)
    - top_n_idx: Indices of top N MACCS features sorted by SHAP (descending)
    - top_n: Number of top features to highlight
    
    Returns:
    - img_bytes: PNG image bytes
    """
    AllChem.Compute2DCoords(molecule)
    MACCSsmartsPatts = MACCSkeys.smartsPatts

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=max(top_n - 1, 1))  # avoid divide-by-zero

    highlight_atoms = []
    highlight_bonds = []
    highlight_atom_colors = {}
    highlight_bond_colors = {}

    for rank, feature in enumerate(top_n_idx[:top_n]):
        if shap_values_array[feature] <= 0 or maccs_features[feature] == 0:
            continue

        smarts = MACCSsmartsPatts[feature][0]
        substructure = Chem.MolFromSmarts(smarts)
        if not substructure:
            continue

        matches = molecule.GetSubstructMatches(substructure)
        if not matches:
            continue

        color = cmap(norm(rank))[:3]  # RGB

        for match in matches:
            highlight_atoms.extend(match)
            for atom in match:
                highlight_atom_colors[atom] = color

            for bond in molecule.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a1 in match and a2 in match:
                    highlight_bonds.append(bond.GetIdx())
                    highlight_bond_colors[bond.GetIdx()] = color

    # Remove duplicates
    highlight_atoms = list(set(highlight_atoms))
    highlight_bonds = list(set(highlight_bonds))

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.drawOptions().useBWAtomPalette()
    drawer.drawOptions().highlightRadius = 0.6
    drawer.drawOptions().fillHighlights = False

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()