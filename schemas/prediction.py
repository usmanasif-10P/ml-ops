from pydantic import BaseModel


class GetPredictionRequest(BaseModel):
    features: list[int]