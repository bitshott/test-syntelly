import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasail.sail import datasail

from config.experiment_config import config, Config



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
    
    def datasail_split(self, df: pd.DataFrame) -> list[dict]:
        splits, _, _ = datasail(
            techniques=["C1e"],
            splits=[8, 2],
            names=["train","test"],
            runs=5,
            solver="SCIP",
            e_type="M",
            threads=config.N_THREADS,
            e_data=dict(df[["index", "smiles"]].values.tolist())
        )
        return splits['C1e']
    
data_utils = DataUtils()