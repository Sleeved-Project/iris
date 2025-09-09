# app/schemas/grading_schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field


class MatchedDefectDetail(BaseModel):
    card_id: str = Field(
        ..., description="ID of the matched card from the database."
    )
    card_name: Optional[str] = Field(
        None, description="Name of the matched card from the database (if available)."
    )
    similarity_percentage: float = Field(
        ..., description="Similarity percentage with this matched hash."
    )
    hamming_distance: int = Field(
        ..., description="Hamming distance to this matched hash."
    )


class GradedCard(BaseModel):
    card_hash: str = Field(..., description="Perceptual hash of the detected card.")
    card_index: int = Field(..., description="Index of the detected card in the image.")
    is_similar: bool = Field(
        False, description="True if a similar card is found in the database."
    )
    similarity_percentage: float = Field(
        ..., description="Similarity percentage with the best matching hash."
    )
    matched_card_id: Optional[str] = Field(
        None, description="ID of the best matching card."
    )
    matched_card_name: Optional[str] = Field(
        None, description="Name of the best matching card."
    )
    description: Optional[str] = Field(
        None, description="CSGO-style description of the card grade."
    )
    top_n_matches: List[MatchedDefectDetail] = Field(
        [], description="Top N closest matches from the database."
    )


class GradingResponse(BaseModel):
    message: str = Field(..., description="Status message of the analysis.")
    cards: List[GradedCard] = Field(..., description="List of graded cards.")
