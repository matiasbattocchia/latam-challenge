import pandas as pd
import pickle
from typing import Tuple, Union, List

from sklearn.linear_model import LogisticRegression


class DelayModel:

    FEATURES_COL = [
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

    TARGET_COL = 'delay'

    DELAY_THRESHOLD_MIN = 15
    
    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
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
        # TODO: sklearn.preprocessing.OneHotEncoder can do it better.
        features = pd.concat(
            [
                pd.get_dummies(data['OPERA'], prefix='OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
                pd.get_dummies(data['MES'], prefix='MES')
            ],
            axis = 1
        )

        features = features.reindex(columns=self.FEATURES_COL, fill_value=0)

        if target_column:
            target = pd.to_datetime(data['Fecha-O']) - pd.to_datetime(data['Fecha-I']) > pd.Timedelta(minutes=self.DELAY_THRESHOLD_MIN)
            target.replace({True: 1, False: 0}, inplace=True)
            target.name = self.TARGET_COL

            return features, target.to_frame()

        return features

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
        target = target[self.TARGET_COL]
        
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
        
        return self._model.predict(features).tolist()

    def save(
        self,
        path: str
    ) -> None:
        """
        Save the DelayModel so it can be re-used.

        Args:
            path (str): output filename.
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(
        path: str
    ) -> "DelayModel":
        """
        Reload the DelayModel from the saved state so it can be re-used.

        Args:
            path (str): input filename.
        """
        with open(path, "rb") as file:
            return pickle.load(file)
        