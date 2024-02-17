import pandas as pd

from typing import Tuple, Union, List

from sklearn.linear_model import LogisticRegression

class DelayModel:

    FEATURES = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET = 'delay'
    
    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # TODO: target -> Series
        # TODO: dropna
        # TODO: missing columns
        features = pd.concat(
            [
                pd.get_dummies(data['OPERA'], prefix='OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
                pd.get_dummies(data['MES'], prefix='MES')
            ], 
            axis = 1
        )

        if target_column:
            return features[self.FEATURES], data[self.TARGET]
        
        return features[self.FEATURES]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        weights = {
            1: sum(target == 0)/len(target),
            0: sum(target == 1)/len(target)
        }

        model = LogisticRegression(class_weight=weights)
        model.fit(features, target)
        
        self._model = model

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        
        return self._model.predict(features).values