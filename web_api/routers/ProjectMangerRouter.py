import uuid

from fastapi import APIRouter, status

from web_api.data_models.DataModels import (
    AddMemberRequest,
    CreateProjectRequest,
    MemberResponse,
    ProjectResponse,
    UpdateMemberRequest,
    UpdateProjectRequest,
)
from web_api.deps.auth import CurrentUser, SessionDep
from web_api.deps.projects import ProjectMemberAccess, ProjectOwnerAccess
from web_api.services.FileHandlerService import FileHandlerService
from web_api.services.ProjectHandlerService import ProjectHandlerService
from web_api.services.ProjectMembersHandlerService import ProjectMembersHandlerService

router = APIRouter(prefix="/projects", tags=["Project Management"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(request: CreateProjectRequest, user: CurrentUser, session: SessionDep):
    """Create a project; the caller becomes its owner."""
    return await ProjectHandlerService(session).create_project(request, user)


@router.get("", response_model=list[ProjectResponse])
async def list_projects(user: CurrentUser, session: SessionDep):
    """Projects the caller is a member of (admins: all projects)."""
    return await ProjectHandlerService(session).list_projects_for(user)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project: ProjectMemberAccess):
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(request: UpdateProjectRequest, project: ProjectOwnerAccess, session: SessionDep):
    return await ProjectHandlerService(session).update_project(project, request)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project: ProjectOwnerAccess, session: SessionDep):
    """Delete a project with all its documents, members and config."""
    project_id = project.id
    await ProjectHandlerService(session).delete_project(project)
    await FileHandlerService.delete_project_files(project_id)


# ---------- Members ----------

@router.get("/{project_id}/members", response_model=list[MemberResponse])
async def list_members(project: ProjectMemberAccess, session: SessionDep):
    return await ProjectMembersHandlerService(session).list_members(project)


@router.post("/{project_id}/members", response_model=MemberResponse, status_code=status.HTTP_201_CREATED)
async def add_member(request: AddMemberRequest, project: ProjectOwnerAccess, user: CurrentUser, session: SessionDep):
    return await ProjectMembersHandlerService(session).add_member(project, request.email, request.role, user)


@router.patch("/{project_id}/members/{user_id}", response_model=MemberResponse)
async def change_member_role(
    user_id: uuid.UUID, request: UpdateMemberRequest, project: ProjectOwnerAccess, session: SessionDep
):
    return await ProjectMembersHandlerService(session).change_role(project, user_id, request.role)


@router.delete("/{project_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(user_id: uuid.UUID, project: ProjectOwnerAccess, session: SessionDep):
    await ProjectMembersHandlerService(session).remove_member(project, user_id)
