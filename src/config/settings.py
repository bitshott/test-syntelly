from pathlib import Path
import multiprocessing


class Settings():
    RANDOM_SEED: int = 42
    N_THREADS: int = multiprocessing.cpu_count()
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH: Path = BASE_DIR / "models"
    REPORTS_PATH: Path = BASE_DIR / "reports"
    DATA_PATH: Path = BASE_DIR / "data"
    PROCESSED_DATA_PATH: Path = BASE_DIR / 'data' / 'processed'

settings = Settings()