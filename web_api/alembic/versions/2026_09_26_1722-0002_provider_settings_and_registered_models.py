"""provider settings and registered models

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-26 17:22:48.193093

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('credential_models',
    sa.Column('credential_id', sa.Uuid(), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('base_model', sa.String(length=200), nullable=True),
    sa.Column('capabilities', postgresql.ARRAY(sa.String(length=16)), server_default='{}', nullable=False),
    sa.Column('embedding_dim', sa.Integer(), nullable=True),
    sa.Column('context_window', sa.Integer(), nullable=True),
    sa.Column('status', sa.Enum('untested', 'ok', 'failed', name='registered_model_status', native_enum=False, create_constraint=False, length=32), nullable=False),
    sa.Column('last_error', sa.Text(), nullable=True),
    sa.Column('last_checked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status IN ('untested', 'ok', 'failed')", name=op.f('ck_credential_models_registered_model_status')),
    sa.CheckConstraint('embedding_dim IS NULL OR embedding_dim > 0', name=op.f('ck_credential_models_embedding_dim_positive')),
    sa.ForeignKeyConstraint(['credential_id'], ['provider_credentials.id'], name=op.f('fk_credential_models_credential_id_provider_credentials'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_credential_models')),
    sa.UniqueConstraint('credential_id', 'name', name=op.f('uq_credential_models_credential_id_name'))
    )
    op.create_index(op.f('ix_credential_models_credential_id'), 'credential_models', ['credential_id'], unique=False)
    # the Ollama URL moved from a per-stage override to the connection's settings
    op.drop_column('project_stage_models', 'base_url')
    op.add_column('provider_credentials', sa.Column('settings', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('provider_credentials', 'settings')
    op.add_column('project_stage_models', sa.Column('base_url', sa.VARCHAR(length=500), autoincrement=False, nullable=True))
    op.drop_index(op.f('ix_credential_models_credential_id'), table_name='credential_models')
    op.drop_table('credential_models')
