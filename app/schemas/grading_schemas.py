from typing import List, Optional
from pydantic import BaseModel, Field


class MatchedDefectDetail(BaseModel):
    card_class: Optional[str] = Field(
        None,
        description="Classe détectée (ex: PSA_8, PSA_9).",
    )
    confidence: float = Field(
        ..., description="Confiance du modèle (%) pour cette prédiction."
    )


class GradedCard(BaseModel):
    average_card_class: Optional[str] = Field(
        None,
        description="Classe PSA finale calculée (ex: PSA_9).",
    )
    top_class_matchs: List[MatchedDefectDetail] = Field(
        [],
        description="Prédictions du modèle avec leur confiance associée.",
    )

    # 🔹 Sous-notes factices alignées avec le score PSA global
    surface_score: float = Field(
        8.0,
        description="Note factice pour la qualité de surface.",
    )
    contour_score: float = Field(
        8.5,
        description="Note factice pour la qualité des contours.",
    )
    corner_score: float = Field(
        9.0,
        description="Note factice pour la qualité des coins.",
    )
    center_score: float = Field(
        8.7,
        description="Note factice pour la qualité du centrage.",
    )


class GradingResponse(BaseModel):
    message: str = Field(
        ..., description="Message de statut de l'analyse de grading."
    )
    cards: List[GradedCard] = Field(
        ...,
        description="Cartes détectées avec leur score PSA et sous-notes.",
    )
