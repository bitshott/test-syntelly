from dataclasses import dataclass
from pathlib import Path
    
    
BASE_DIR = Path(__file__).resolve().parent.parent 


@dataclass
class Config:
    DEVICE: str = 'cuda:0'

    DATA_PATH: Path = BASE_DIR / 'data'
    MODEL_PATH: Path = BASE_DIR / 'models'


    N_THREADS: int = 71
    RANDOM_SEED: int = 42
    N_OPTUNA_TRIALS: int = 30


config = Config()