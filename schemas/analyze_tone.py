from pydantic import BaseModel


class AnalyzeToneRequest(BaseModel):
    message: str


TONE_MAPPING = {
    1: "Appreciative",
    2: "Cautionary",
    3: "Diplomatic",
    4: "Direct",
    5: "Informative"
}