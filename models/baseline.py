"""
Linear Regression and Random Forest trained on
handcrafted statistical features. These serve as a check that signal
about H is present in the trajectories.

Maybe linear regerssion can give satisfactory predictions.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# from our code 
from utils.preprocessing import extract_features


class BaselineHurstRegressor:

    """
    Use scikit-learn regressor on
    handcrafted features extracted from raw trajectories.

    Good for systematic pipeline. 
    A direct regression can also be used
    
    """

    def __init__(self, 
                 kind: str = "random_forest", **model_kwargs):
        
        if kind == "linear":
            self.model = LinearRegression(**model_kwargs)

        elif kind == "random_forest":
            
            defaults = dict(n_estimators=300, 
                            max_depth=None, 
                            random_state=0, 
                            n_jobs=-1)
            
            defaults.update(model_kwargs)
            
            self.model = RandomForestRegressor(**defaults)


        else:
            raise ValueError(f"Unknown baseline kind: {kind}")

        self.kind = kind
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        feats = extract_features(X)
        feats = self.scaler.fit_transform(feats)
        self.model.fit(feats, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = extract_features(X)
        feats = self.scaler.transform(feats)
        return self.model.predict(feats)
