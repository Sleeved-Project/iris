from pydantic import BaseModel, Field, HttpUrl
from typing import Optional


class ImageHashRequest(BaseModel):
    """Request model for image hash endpoint with proper validation."""

    url: Optional[HttpUrl] = Field(default=None, description="URL of the image to hash")

    class Config:
        arbitrary_types_allowed = True


class ImageHashResponse(BaseModel):
    """Response model for image hash endpoint."""

    hash: str = Field(description="Combined perceptual hash (dhash + phash)")
