from web_api.services.UserService import UserService
from web_api.services.SecurityService import SecurityService
from web_api.data_models.UserModels import UserModel
from datetime import datetime, timedelta, timezone
from pydantic import EmailStr
from web_api.data_models.enums import AppRole
from web_api.services.EmailService import EmailService
from pydantic import EmailStr
from beanie import PydanticObjectId
from pydantic import BaseModel
from web_api.services.JWTService import JWTService
from web_api.errors import (
    UserAlreadyExists,
    InvalidOrExpiredToken,
    AccountNotActive,
    InvalidCredentials,
    ConflictError,
)


class AuthPayload(BaseModel):
    email: EmailStr
    username: str
    user_id: str
    role: AppRole



class AuthService:
    def __init__(self, userService: UserService, securityService: SecurityService, jwtService: JWTService, emailService: EmailService):
        self.user_service = userService
        self.security_service = securityService
        self.jwt_service = jwtService
        self.email_service = emailService

    async def add_user(self,email : EmailStr, app_role : AppRole ):
        user: UserModel|None = await self.user_service.find_email(email)
        if user:
            raise UserAlreadyExists()
        setup_token = self.security_service.generate_secure_token()
        setup_token_expiry = datetime.now(timezone.utc) + timedelta(hours=24)
        new_user = await self.user_service.add_new_user(email, app_role, setup_token, setup_token_expiry)
        resend_id = await self.email_service.send_email(email, setup_token)
        await self.email_service.save_email_verification(new_user.id, resend_id)
        return new_user
    
    async def authenticate_initial_token(self, token: str) -> UserModel | None:
        user = await self.user_service.find_user_by_token(token)
        if not user:
            raise InvalidOrExpiredToken("Invalid token")
        if user.setup_token_expiry < datetime.now(timezone.utc):
            raise InvalidOrExpiredToken("Token expired")
        return user
    
    async def set_credentials_first_login(self, user_id: PydanticObjectId, user_name: str, new_password: str)-> PydanticObjectId:
        if not await self.user_service.is_user_active(user_id):
            raise AccountNotActive()
        if not await self.user_service.is_must_change_password(user_id):
            raise ConflictError("Password cannot be changed at this time")

        self.security_service.validate_password(new_password)
        hashed_password = self.security_service.hash_password(new_password)
        user_id = await self.user_service.update_UserName_and_Password(user_id, user_name, hashed_password)
        return user_id
    
    async def authenticate_user(self, email: EmailStr, password: str) -> str:
        user = await self.user_service.find_email(email)
        if not user:
            raise InvalidCredentials()
        if not user.is_active:
            raise AccountNotActive()
        if not self.security_service.verify_password(password, user.hashed_password):
            raise InvalidCredentials()
        
        auth_payload = AuthPayload(
            email=user.email,
            username=user.username,
            user_id=str(user.id),
            role=user.app_role
        )
    
        return self.jwt_service.create_token(auth_payload.model_dump())
    
    


def get_auth_service(user_service: UserService, security_service: SecurityService, jwt_service: JWTService, email_service: EmailService) -> AuthService:
    return AuthService(user_service, security_service, jwt_service, email_service)

        

