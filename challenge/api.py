import fastapi
from typing import Literal
from pydantic import BaseModel, Field
import pandas as pd
# TODO: tests require to import from challenge.model
from model import DelayModel

app = fastapi.FastAPI()

model = DelayModel.load("challenge/delay_model.pkl")

class Flight(BaseModel):
    # TODO: derive OPERA possible values from DelayModel
    OPERA: Literal[
        "American Airlines",
        "Air Canada",
        "Air France",
        "Aeromexico",
        "Aerolineas Argentinas",
        "Austral",
        "Avianca",
        "Alitalia",
        "British Airways",
        "Copa Air",
        "Delta Air",
        "Gol Trans",
        "Iberia",
        "K.L.M.",
        "Qantas Airways",
        "United Airlines",
        "Grupo LATAM",
        "Sky Airline",
        "Latin American Wings",
        "Plus Ultra Lineas Aereas",
        "JetSmart SPA",
        "Oceanair Linhas Aereas",
        "Lacsa"
    ]
    TIPOVUELO: Literal["N", "I"]
    MES: int = Field(ge=1, le=12)


class Payload(BaseModel):
    flights: list[Flight]

    
class Prediction(BaseModel):
    predict: list[int]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

# TODO: fix the mapping of dict to a list of Flight Pydantic models;
# we want the dicts not the models, hence Pydantic is not helping here
@app.post("/predict", status_code=200, response_model=Prediction)
async def post_predict(payload: Payload) -> dict:
    df = pd.DataFrame(map(dict, payload.flights))

    preprocessed_df = model.preprocess(df)

    predictions = model.predict(preprocessed_df)
    
    return {"predict": predictions}