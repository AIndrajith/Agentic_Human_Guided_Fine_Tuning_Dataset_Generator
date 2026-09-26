from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.DataModels import CreateProjectRequest, UpdateProjectRequest
from web_api.data_models.enums import AppRole, ProjectRole
from web_api.db.models import Document, Project, ProjectMember, User
from web_api.errors import ConflictError


class ProjectHandlerService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_project(self, request: CreateProjectRequest, creator: User) -> Project:
        project = Project(
            title=request.title,
            description=request.description,
            data_type=request.data_type,
            created_by=creator.id,
        )
        self.session.add(project)
        await self.session.flush()
        # the creator owns the project; same transaction, so no ownerless projects
        self.session.add(ProjectMember(
            project_id=project.id, user_id=creator.id, role=ProjectRole.OWNER, added_by=creator.id
        ))
        await self.session.commit()
        return project

    async def list_projects_for(self, user: User) -> list[Project]:
        query = select(Project).order_by(Project.created_at.desc())
        if user.app_role != AppRole.ADMIN:
            query = query.join(ProjectMember, ProjectMember.project_id == Project.id).where(
                ProjectMember.user_id == user.id
            )
        return list(await self.session.scalars(query))

    async def update_project(self, project: Project, request: UpdateProjectRequest) -> Project:
        changes = request.model_dump(exclude_unset=True, exclude_none=True)
        if "data_type" in changes and changes["data_type"] != project.data_type:
            has_documents = await self.session.scalar(
                select(Document.id).where(Document.project_id == project.id).limit(1)
            )
            if has_documents:
                raise ConflictError("Cannot change data type after documents were uploaded")
        for field, value in changes.items():
            setattr(project, field, value)
        await self.session.commit()
        return project

    async def delete_project(self, project: Project) -> None:
        """Deletes the row; members, documents, config, chunks cascade in the DB.
        Caller removes the project's MinIO objects afterwards."""
        await self.session.delete(project)
        await self.session.commit()
