from pydantic import BaseModel


class AnalyzeToneRequest(BaseModel):
    message: str