import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def generate_fingerprints(smiles_list, radius=2, n_bits=1024):
    """
    Generate MACCS fingerprints for a list of SMILES strings.
    
    Args:
        smiles_list (list): List of SMILES strings.
        radius (int): Radius for Morgan (not used currently).
        n_bits (int): Number of bits for Morgan (not used currently).

    Returns:
        np.ndarray: MACCS fingerprints.
    """
    maccs_fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            maccs_fp = list(MACCSkeys.GenMACCSKeys(mol))  # MACCS (167 bits)
            maccs_fps.append(maccs_fp)
        else:
            print(f"Invalid SMILES: {smi}")
            maccs_fps.append([0] * 167)  # Placeholder

    return np.array(maccs_fps)

