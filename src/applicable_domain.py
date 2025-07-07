import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
import pandas as pd

def load_maccs_training_fps(training_path="../data/processed/training.csv"):
    """
    Loads training SMILES and computes MACCS fingerprints.
    
    Args:
        training_path (str): Path to CSV file with 'smiles' column.
    
    Returns:
        train_fps (List): List of RDKit MACCS fingerprints.
        smiles_list (List): Corresponding SMILES strings.
    """
    df = pd.read_csv(training_path)
    smiles_list = df["smile"].dropna().tolist()

    train_fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            train_fps.append(fp)
    return train_fps, smiles_list


def get_max_maccs_similarity(test_smiles, train_fps):
    """
    Computes the maximum Tanimoto similarity between a test molecule
    and training set fingerprints using MACCS keys.

    Args:
        test_smiles (str): SMILES of the test molecule.
        train_fps (List): List of MACCS fingerprints of training set.

    Returns:
        float: Maximum Tanimoto similarity value.
    """
    mol = Chem.MolFromSmiles(test_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {test_smiles}")
    
    test_fp = MACCSkeys.GenMACCSKeys(mol)
    similarities = [TanimotoSimilarity(test_fp, fp) for fp in train_fps]
    
    return max(similarities) if similarities else 0.0


def get_max_maccs_similarity_with_match(test_smiles, train_fps, train_smiles):
    """
    Computes the max Tanimoto similarity and returns the corresponding
    training molecule with the highest similarity.

    Args:
        test_smiles (str): Test compound SMILES.
        train_fps (List): List of training fingerprints.
        train_smiles (List): Corresponding SMILES strings.

    Returns:
        Tuple[float, str]: (Max similarity, most similar training SMILES)
    """
    mol = Chem.MolFromSmiles(test_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {test_smiles}")
    
    test_fp = MACCSkeys.GenMACCSKeys(mol)
    similarities = [TanimotoSimilarity(test_fp, fp) for fp in train_fps]

    if not similarities:
        return 0.0, None

    max_idx = similarities.index(max(similarities))
    return similarities[max_idx], train_smiles[max_idx]


def check_applicability_domain(test_smiles, train_fps, cutoff=0.4, verbose=True):
    """
    Checks whether a test molecule is within the applicability domain.

    Args:
        test_smiles (str): Test compound SMILES.
        train_fps (List): MACCS fingerprints from training set.
        cutoff (float): Tanimoto similarity threshold.
        verbose (bool): Whether to print the result.

    Returns:
        float: Max Tanimoto similarity.
    """
    max_sim = get_max_maccs_similarity(test_smiles, train_fps)
    
    if verbose:
        if max_sim < cutoff:
            print(f"⚠️  Max MACCS similarity = {max_sim:.2f} < cutoff {cutoff} — prediction may be out-of-domain.")
        else:
            print(f"✅ Max MACCS similarity = {max_sim:.2f} — within applicability domain.")
    
    return max_sim

