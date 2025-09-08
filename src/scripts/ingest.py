import pandas as pd
from pathlib import Path
import joblib

from typing import Literal

from rdkit import Chem
from rdkit.Chem import Descriptors

from src.config.settings import settings
from src.config.columns import TO_DROP_BOIL, TO_DROP_MELT

def ingest_data(target_name: Literal['boiling_points', 'melting_points']) -> None:
    if target_name == "melting_points":
        file_name = "melt_clean.csv"
        target_col = "melt_value"
        to_drop = TO_DROP_MELT

    elif target_name == "boiling_points":
        file_name = "boil_clean.csv"
        target_col = "boil_value"
        to_drop = TO_DROP_BOIL

    data = pd.read_csv(settings.DATA_PATH / file_name)
    desc = data['canonical_smiles'].apply(lambda smi: smiles_to_rdkit_desc(smi, columns_to_drop=to_drop))

    X = pd.DataFrame(desc.to_list())
    y = data[target_col]
    dataset = {
        "X": X,
        "y": y
    }

    processed_path = settings.PROCESSED_DATA_PATH
    processed_path.mkdir(parents=True, exist_ok=True)
    processed_file = processed_path / f'{target_name}_processed.pkl'

    joblib.dump(dataset, processed_file)

def smiles_to_rdkit_desc(smiles: str, columns_to_drop: list = None) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(smiles)
        return None

    if columns_to_drop is None:
        return {nm: fn(mol) for nm, fn in Descriptors._descList}
    else:
        return {nm: fn(mol) for nm, fn in Descriptors._descList if nm not in columns_to_drop}

if __name__ == "__main__":
    ingest_data("melting_points")
    ingest_data("boiling_points")