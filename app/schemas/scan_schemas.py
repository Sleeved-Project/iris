from pydantic import BaseModel, Field


class ScanResponse(BaseModel):
    message: str = Field(
        description="Status message indicating number of cards detected"
    )
    cards_detected: bool = Field(description="True if at least one card was detected")
