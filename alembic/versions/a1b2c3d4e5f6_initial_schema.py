"""initial schema: questions / kb_chunks / attempts / conversations / metrics

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-08-30

"""
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    from storage.models import Base
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    from storage.models import Base
    Base.metadata.drop_all(bind=op.get_bind())
