"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    # 初始迁移：按 ORM 元数据建表（方言无关，MySQL/SQLite 皆可）
    from storage.models import Base
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    from storage.models import Base
    Base.metadata.drop_all(bind=op.get_bind())
