import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.DataModels import MemberResponse
from web_api.data_models.enums import ProjectRole
from web_api.db.models import Project, ProjectMember, User
from web_api.errors import ConflictError, MemberNotFound, UserNotFound
from web_api.services.UserService import UserService


class ProjectMembersHandlerService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_members(self, project: Project) -> list[MemberResponse]:
        rows = await self.session.execute(
            select(ProjectMember, User)
            .join(User, User.id == ProjectMember.user_id)
            .where(ProjectMember.project_id == project.id)
            .order_by(ProjectMember.added_at)
        )
        return [self._to_response(member, user) for member, user in rows]

    async def add_member(self, project: Project, email: str, role: ProjectRole, added_by: User) -> MemberResponse:
        user = await UserService(self.session).find_by_email(email)
        if not user:
            raise UserNotFound("No user with this email; invite them first")
        if await self.session.get(ProjectMember, (project.id, user.id)):
            raise ConflictError("User is already a member of this project")

        member = ProjectMember(project_id=project.id, user_id=user.id, role=role, added_by=added_by.id)
        self.session.add(member)
        await self.session.commit()
        return self._to_response(member, user)

    async def change_role(self, project: Project, user_id: uuid.UUID, role: ProjectRole) -> MemberResponse:
        member = await self._get_member(project, user_id)
        if member.role == ProjectRole.OWNER and role != ProjectRole.OWNER:
            await self._ensure_not_last_owner(project)
        member.role = role
        await self.session.commit()
        return self._to_response(member, await self.session.get(User, user_id))

    async def remove_member(self, project: Project, user_id: uuid.UUID) -> None:
        member = await self._get_member(project, user_id)
        if member.role == ProjectRole.OWNER:
            await self._ensure_not_last_owner(project)
        await self.session.delete(member)
        await self.session.commit()

    async def _get_member(self, project: Project, user_id: uuid.UUID) -> ProjectMember:
        member = await self.session.get(ProjectMember, (project.id, user_id))
        if not member:
            raise MemberNotFound()
        return member

    async def _ensure_not_last_owner(self, project: Project) -> None:
        owners = await self.session.scalar(
            select(func.count())
            .select_from(ProjectMember)
            .where(ProjectMember.project_id == project.id, ProjectMember.role == ProjectRole.OWNER)
        )
        if owners <= 1:
            raise ConflictError("A project must keep at least one owner")

    @staticmethod
    def _to_response(member: ProjectMember, user: User) -> MemberResponse:
        return MemberResponse(
            user_id=user.id,
            email=user.email,
            username=user.username,
            role=member.role,
            added_by=member.added_by,
            added_at=member.added_at,
        )
