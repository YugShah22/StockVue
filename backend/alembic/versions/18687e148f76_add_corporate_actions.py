"""add corporate actions

Revision ID: 18687e148f76
Revises: 9d0727a166ef
Create Date: 2026-10-07 00:10:55.112804

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "18687e148f76"
down_revision: str | Sequence[str] | None = "9d0727a166ef"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "corporate_actions",
        sa.Column("corporate_action_id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("action_type", sa.String(length=50), nullable=False),
        sa.Column("execution_date", sa.Date(), nullable=False),
        sa.Column("value", sa.Numeric(precision=24, scale=6), nullable=False),
        sa.Column("currency", sa.String(length=10), nullable=True),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["instrument_id"],
            ["instruments.instrument_id"],
        ),
        sa.PrimaryKeyConstraint("corporate_action_id"),
        sa.UniqueConstraint(
            "instrument_id",
            "execution_date",
            "action_type",
            name="uq_corporate_action_instrument_date_type",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("corporate_actions")
