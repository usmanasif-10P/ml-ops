from pydantic import BaseModel


class GetPrediction(BaseModel):
    features: list[int]