import pandas as pd

def load_dataset(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_smiles, train_y = train_df.iloc[:,0].tolist(), train_df.iloc[:,1].values
    test_smiles, test_y = test_df.iloc[:, 0].tolist(), test_df.iloc[:, 1].values

    return train_smiles, train_y, test_smiles, test_y






