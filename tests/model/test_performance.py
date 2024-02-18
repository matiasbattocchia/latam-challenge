import unittest
import pandas as pd

from sklearn.metrics import classification_report
from challenge.model import DelayModel

class TestPerformance(unittest.TestCase):

    
    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        self.data = pd.read_csv(filepath_or_buffer="data/data.csv", low_memory=False)

    
    def test_model_performance(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model.predict(
            features=features
        )

        report = classification_report(target, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30