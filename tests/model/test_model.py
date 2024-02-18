import pytest
import pandas as pd

from challenge.model import DelayModel
from sklearn.utils.validation import check_is_fitted

        
class TestModel():
        
    FEATURES_COLS = [
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

    TARGET_COL = [
        "delay"
    ]

    
    @pytest.fixture
    def data(self):
        return pd.read_csv(filepath_or_buffer="data/data.csv", low_memory=False)

    
    @pytest.fixture(autouse=True)
    def model(self):
        self.model = DelayModel()

    
    @pytest.fixture
    def features(self):
        return pd.DataFrame([[1,2], [3, 4]])


    @pytest.fixture
    def target(self):
        return pd.DataFrame([0,1], columns=TestModel.TARGET_COL)

    
    def test_model_preprocess_for_training(
        self,
        data
    ):
        features, target = self.model.preprocess(
            data=data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)


    def test_model_preprocess_for_serving(
        self,
        data
    ):
        features = self.model.preprocess(
            data=data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

    
    def test_model_fit(
        self,
        features,
        target
    ):
        self.model.fit(
            features=features,
            target=target
        )

        # will raise an exception if not
        check_is_fitted(self.model._model) 


    def test_model_predict(
        self,
        features,
        target
    ):
        class ModelMock():
            def predict(_, X):
                # should return a numpy array as sklearn does
                return target[self.TARGET_COL[0]].values

        self.model._model = ModelMock()

        predicted_targets = self.model.predict(
            features=features
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)