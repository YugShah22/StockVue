"""add market bars

Revision ID: 8df8fbb0ba67
Revises: a4a086de3451
Create Date: 2026-10-01 23:14:36.498577

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8df8fbb0ba67"
down_revision: str | Sequence[str] | None = "a4a086de3451"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "market_bars",
        sa.Column("market_bar_id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["instrument_id"],
            ["instruments.instrument_id"],
        ),
        sa.PrimaryKeyConstraint("market_bar_id"),
        sa.UniqueConstraint(
            "instrument_id",
            "timestamp",
            name="uq_market_bar_instrument_timestamp",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("market_bars")
