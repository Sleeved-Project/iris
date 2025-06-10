# app/schemas/analysis_schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field


class MatchedCardDetail(BaseModel):
    """Details for a single matched card from the database."""

    card_id: str = Field(..., description="ID of the matched card from the database.")
    card_name: Optional[str] = Field(
        None, description="Name of the matched card from the database (if available)."
    )
    similarity_percentage: float = Field(
        ..., description="Similarity percentage with this matched hash."
    )
    hamming_distance: int = Field(
        ..., description="Hamming distance to this matched hash."
    )


class AnalyzedCard(BaseModel):
    card_hash: str = Field(..., description="Perceptual hash of the detected card.")
    card_index: int = Field(..., description="Index of the detected card in the image.")
    is_similar: bool = Field(
        False,
        description="""True if a similar card is
        found in the database based on threshold.""",
    )
    similarity_percentage: float = Field(
        ...,
        description="""Similarity percentage with the
        best matching hash in the database.""",
    )
    matched_card_id: Optional[str] = Field(
        None, description="ID of the best matching card from the database."
    )
    matched_card_name: Optional[str] = Field(
        None,
        description="Name of the best matching card from the database (if available).",
    )

    # New field to store the top N matches
    top_n_matches: List[MatchedCardDetail] = Field(
        [], description="Top N closest matches from the database, sorted by similarity."
    )


class AnalysisResponse(BaseModel):
    message: str = Field(..., description="Status message of the analysis.")
    cards: List[AnalyzedCard] = Field(
        ..., description="List of detected and analyzed cards."
    )
