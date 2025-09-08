import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal

from datasail.sail import datasail

from config.experiment_config import config, Config

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


class DataUtils:
    def __init__(self):
        self._outliers_threshold: float = 3.0
        self._config: Config = config

    def plot_target(self, target: pd.Series) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.add_subplot(sns.boxplot(target, ax=axes[0]))
        fig.add_subplot(sns.histplot(target, ax=axes[1], kde=True, color='green'))

    def check_normality(self, target: pd.Series) -> None:
        mu, sigma = np.mean(target), np.std(target, ddof=1)
        stat, p_value = scipy.stats.kstest(target, 'norm', args=(mu, sigma))

        skew = scipy.stats.skew(target)
        kurtosis = scipy.stats.kurtosis(target)

        print(f"КС-тест = {stat}")
        print(f"p-value = {p_value}")
        print(f"Сдвиг: {skew}")
        print(f"Эксцесс: {kurtosis}")

    def get_outliers_mask(self, target: pd.Series) -> np.ndarray:
        z_scores = scipy.stats.zscore(target)
        outlier_mask = np.abs(z_scores) > self._outliers_threshold
        return outlier_mask
    
    def datasail_split(self, df: pd.DataFrame, technique: Literal["C1e", "I1e"]) -> tuple:
        return datasail(
            techniques=[technique],
            splits=[7, 1, 1],
            names=["train","val", "test"],
            runs=5,
            solver="SCIP",
            threads=config.N_THREADS,
            e_type="M",
            e_data=dict(df[["index", "canonical_smiles"]].values.tolist())
        )

    def standardize_smiles(self, smiles: str) -> str:
        if '[2H]' in smiles:
            print(smiles)
            return smiles
        
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except Exception:
            mol = None
    
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                raise ValueError(f"Value Error: {smiles}")
    
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
                Chem.SanitizeMol(mol)
            except Exception:
                return smiles
    
        try:
            mol = rdMolStandardize.Normalizer().normalize(mol)
        except Exception:
            pass
    
        try:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        except Exception:
            pass
    
        try:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        except Exception:
            pass
    
        try:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        except Exception:
            pass
    
        flag = Chem.SanitizeMol(mol, catchErrors=True)
            
        if flag != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_ALL ^ flag)
            
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            
        return canonical_smiles


    
data_utils = DataUtils()