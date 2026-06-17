from web_api.services.SecurityService import SecurityService
from beanie import PydanticObjectId
from web_api.data_models.UserModels import UserModel
from web_api.errors import UserNotFound

class UserService:
    def __init__(self,SecurityService: SecurityService):
        self.security_service = SecurityService

    async def find_email(self, email: str) -> UserModel | None:
        return await UserModel.find_one(UserModel.email == email)
    
    async def add_new_user(self, email: str, app_role: str, setup_token: str, setup_token_expiry) -> UserModel:
        new_user = UserModel(
            email=email,
            app_role=app_role,
            setup_token=setup_token,
            setup_token_expiry=setup_token_expiry
        )
        try:
            await new_user.insert()
        except Exception as e:
            raise RuntimeError("Failed to add new user") from e
        return new_user
    
    async def is_user_active(self, user_id: PydanticObjectId) -> bool:
        user = await UserModel.get(user_id)
        if not user:
            raise UserNotFound()
        return user.is_active
    
    async def is_must_change_password(self, user_id: PydanticObjectId) -> bool:
        user = await UserModel.get(user_id)
        if not user:
            raise UserNotFound()
        return user.must_change_password
    
    async def update_username(self, user_id: PydanticObjectId, new_username: str):
        user = await UserModel.get(user_id)
        if not user:
            raise UserNotFound()
        user.username = new_username
        try:
            await user.save()
        except Exception as e:
            raise RuntimeError("Failed to update username") from e
        
    async def update_password(self, user_id: PydanticObjectId, hashed_password: str):
        user = await UserModel.get(user_id)
        if not user:
            raise UserNotFound()
        user.hashed_password = hashed_password
        try:
            await user.save()
        except Exception as e:
            raise RuntimeError("Failed to update password") from e
        
    async def update_UserName_and_Password(self, user_id: PydanticObjectId, new_username: str, hashed_password: str):
        user = await UserModel.get(user_id)
        if not user:
            raise UserNotFound()
        user.username = new_username
        user.hashed_password = hashed_password
        try:
            await user.save()
            return user.id
        except Exception as e:
            raise RuntimeError("Failed to update username and password") from e
        
    async def find_email(self, email: str) -> UserModel | None:
        return await UserModel.find_one(UserModel.email == email)
    
    async def find_user_by_token(self, token: str) -> UserModel | None:
        return await UserModel.find_one(UserModel.setup_token == token)
    
    async def change_must_change_password(self, user_id: PydanticObjectId, must_change: bool):
        user = await UserModel.get(user_id)
        if not user:
            raise UserNotFound()
        user.must_change_password = must_change
        try:
            await user.save()
        except Exception as e:
            raise RuntimeError("Failed to update must_change_password") from e