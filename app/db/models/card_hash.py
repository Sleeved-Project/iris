from sqlalchemy import Column, String, DateTime, Index, func
from app.db.session import Base


class CardHash(Base):
    """
    Model to store card identifiers and their corresponding perceptual hashes.
    Optimized for fast lookups by both card id and hash value.
    """

    __tablename__ = "Card_hash"

    id = Column(
        String(255),
        primary_key=True,
        nullable=False,
        unique=True,
        index=True,
        comment="Unique card identifier",
    )
    hash = Column(
        String(128),
        nullable=False,
        index=True,
        comment="Perceptual hash of the card image",
    )
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    __table_args__ = (
        Index(
            "idx_hash_prefix",
            "hash",
            mysql_length=64,
            postgresql_ops={"hash": "varchar_pattern_ops"},
        ),
    )
