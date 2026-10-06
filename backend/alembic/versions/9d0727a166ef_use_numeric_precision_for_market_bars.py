"""use numeric precision for market bars

Revision ID: 9d0727a166ef
Revises: 347812d99574
Create Date: 2026-10-06 00:27:00.142811

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9d0727a166ef"
down_revision: str | Sequence[str] | None = "347812d99574"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "market_bars",
        "open",
        existing_type=sa.Float(),
        type_=sa.Numeric(precision=18, scale=6),
        existing_nullable=False,
    )
    op.alter_column(
        "market_bars",
        "high",
        existing_type=sa.Float(),
        type_=sa.Numeric(precision=18, scale=6),
        existing_nullable=False,
    )
    op.alter_column(
        "market_bars",
        "low",
        existing_type=sa.Float(),
        type_=sa.Numeric(precision=18, scale=6),
        existing_nullable=False,
    )
    op.alter_column(
        "market_bars",
        "close",
        existing_type=sa.Float(),
        type_=sa.Numeric(precision=18, scale=6),
        existing_nullable=False,
    )
    op.alter_column(
        "market_bars",
        "volume",
        existing_type=sa.Float(),
        type_=sa.Numeric(precision=24, scale=6),
        existing_nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "market_bars",
        "volume",
        existing_type=sa.Numeric(precision=24, scale=6),
        type_=sa.Float(),
        existing_nullable=False,
    )
    op.alter_column(
        "market_bars",
        "close",
        existing_type=sa.Numeric(precision=18, scale=6),
        type_=sa.Float(),
        existing_nullable=False,
    )
    op.alter_column(
        "market_bars",
        "low",
        existing_type=sa.Numeric(precision=18, scale=6),
        type_=sa.Float(),
        existing_nullable=False,
    )
    op.alter_column(
        "market_bars",
        "high",
        existing_type=sa.Numeric(precision=18, scale=6),
        type_=sa.Float(),
        existing_nullable=False,
    )
    op.alter_column(
        "market_bars",
        "open",
        existing_type=sa.Numeric(precision=18, scale=6),
        type_=sa.Float(),
        existing_nullable=False,
    )
