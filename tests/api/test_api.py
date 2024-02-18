import unittest
from mockito import when, ANY
from fastapi.testclient import TestClient
from challenge import app
import numpy as np
from challenge.model import DelayModel

class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        when("challenge.model.DelayModel").predict(ANY).thenReturn([0])
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})

    def test_should_failed_unkown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        when("challenge.model.DelayModel").predict(ANY).thenReturn([0])
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)

    def test_should_failed_unkown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O",
                    "MES": 12 # changed 13 by 12 so the test can focus on TIPOVUELO
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)
    
    def test_should_failed_unkown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", # changed O by N so the test can focus on OPERA
                    "MES": 12 # changed 13 by 12 so the test can focus on OPERA
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)