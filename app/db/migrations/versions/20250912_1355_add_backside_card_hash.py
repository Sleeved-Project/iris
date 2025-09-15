"""add backside card hash

Revision ID: 92e5ba713d4c
Revises: e03299a3d603
Create Date: 2025-09-12 13:55:00

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "92e5ba713d4c"
down_revision: Union[str, None] = "e03299a3d603"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Insert the back-side card hash record
    op.execute(
        """
        INSERT INTO card_hash (id, hash, created_at)
        VALUES (
            'back-side',
            '6e04fa5ee46ee85cde0ef8c0f2d2e188f1c8f9d8cdf087c485a889589480e90bd115257f6c489ef863a572820ed6cf2e39ab789226aef3491af8c8352de760c2',
            '2025-06-11 15:20:00'
        )
        ON DUPLICATE KEY UPDATE id=id
        """
    )


def downgrade() -> None:
    # Remove the back-side card hash record if needed
    op.execute("DELETE FROM card_hash WHERE id = 'back-side'")
