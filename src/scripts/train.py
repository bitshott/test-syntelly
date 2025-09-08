import pandas as pd
import joblib
import json

from typing import Literal, Type
from datetime import datetime
import platform

import sklearn
from sklearn.ensemble import RandomForestRegressor
import lightgbm
from lightgbm import LGBMRegressor

from src.config.settings import settings


def train(target_name: Literal['boiling_points', 'melting_points'], 
            model_class: Type[RandomForestRegressor] | Type[LGBMRegressor],  
            model_version: int) -> None:
   
    
    model_dir = settings.MODEL_PATH / f'{target_name}_models' / f'v{model_version}'
    model_dir.mkdir(exist_ok=True, parents=True)

    model_file = model_dir / f'{target_name}_model.pkl'
    
    model_params_file = settings.REPORTS_PATH / f'{target_name}_model_params.json'
    model_params = json.load((model_params_file).open())

    dataset = joblib.load(settings.PROCESSED_DATA_PATH / f'{target_name}_processed.pkl')
    X = dataset['X']
    y = dataset['y']

    model = model_class(**model_params, random_state=settings.RANDOM_SEED)
    model.fit(X=X, y=y)

    joblib.dump(model, model_file, compress=('gzip', 3))
     
    metadata = {
        "target_name": target_name,                      
        "model_class": model_class.__name__,               
        "model_version": model_version,                    
        "training_date": datetime.now().isoformat(),      
        "n_samples": X.shape[0],                          
        "n_features": X.shape[1],                        
        "model_params": model_params,                      
        "random_seed": settings.RANDOM_SEED,             
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__ if model_class.__name__=="LGBMRegressor" else None,
        "feature_columns": list(X.columns)               
    }
    
    metadata_file = model_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    train(target_name='boiling_points', model_class=RandomForestRegressor, model_version=2)
    train(target_name='melting_points', model_class=LGBMRegressor, model_version=1)