import uuid
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.enums import AppRole, ProjectRole
from web_api.db.models import Project, ProjectMember, User
from web_api.deps.auth import CurrentUser, SessionDep
from web_api.errors import ProjectAccessDenied, ProjectNotFound


async def ensure_project_access(
    session: AsyncSession, user: User, project_id: uuid.UUID, roles: tuple[ProjectRole, ...] = ()
) -> Project:
    """Return the project if `user` may access it. Admins always may.

    `roles` empty = any member; otherwise the member's role must be in `roles`.
    Non-members get 404 (not 403) so project ids can't be probed.
    """
    project = await session.get(Project, project_id)
    if not project:
        raise ProjectNotFound()
    if user.app_role == AppRole.ADMIN:
        return project

    member = await session.get(ProjectMember, (project_id, user.id))
    if not member:
        raise ProjectNotFound()
    if roles and member.role not in roles:
        raise ProjectAccessDenied()
    return project


def _project_dependency(*roles: ProjectRole):
    async def dependency(project_id: uuid.UUID, user: CurrentUser, session: SessionDep) -> Project:
        return await ensure_project_access(session, user, project_id, roles)

    return dependency


# use on routes with a `{project_id}` path parameter
ProjectMemberAccess = Annotated[Project, Depends(_project_dependency())]
ProjectOwnerAccess = Annotated[Project, Depends(_project_dependency(ProjectRole.OWNER))]
