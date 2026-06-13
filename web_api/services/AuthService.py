from web_api.services.UserService import UserService
from web_api.services.SecurityService import SecurityService
from web_api.data_models.UserModels import UserModel
from datetime import datetime, timedelta, timezone
from pydantic import EmailStr
from data_models.enums import AppRole
from web_api.services.EmailService import EmailService
from pydantic import EmailStr 
from beanie import PydanticObjectId


class AuthService:
    def __init__(self, userService: UserService, securityService: SecurityService):
        self.user_service = userService
        self.security_service = securityService

    async def add_user(self,email : EmailStr, app_role : AppRole , email_service: EmailService):
        user: UserModel|None = await self.user_service.find_email(email)
        if user:
            raise Exception("User already exists")
        setup_token = self.security_service.generate_secure_token()
        setup_token_expiry = datetime.now(timezone.utc) + timedelta(hours=24)
        new_user = await self.user_service.add_new_user(email, app_role, setup_token, setup_token_expiry)
        resend_id = await email_service.send_email(email, setup_token)
        await email_service.save_email_verification(new_user.id, resend_id)
        return new_user
    
    async def authenticate_initial_token(self, token: str) -> UserModel:
        user = await UserModel.find_one(UserModel.setup_token == token)
        if not user:
            raise Exception("Invalid token")
        if user.setup_token_expiry < datetime.now(timezone.utc):
            raise Exception("Token expired")
        return user
    
    async def set_credentials_first_login(self, user_id: PydanticObjectId, user_name: str, new_password: str):
        if not await self.user_service.is_user_active(user_id):
            raise Exception("User is not active")
        if not await self.user_service.is_must_change_password(user_id):
            raise Exception("Password cannot be changed at this time")

        self.security_service.validate_password(new_password)
        hashed_password = self.security_service.hash_password(new_password)
        user_id = await self.user_service.update_UserName_and_Password(user_id, user_name, hashed_password)
        return user_id
