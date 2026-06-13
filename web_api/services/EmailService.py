import os
from beanie import PydanticObjectId
import resend
from pydantic import EmailStr
from web_api.data_models.UserModels import EmailVerificationModel


class EmailService:
    def __init__(self):
        api_key = os.getenv("RESEND_API_KEY")
        if not api_key:
            raise RuntimeError("RESEND_API_KEY not set")
        resend.api_key = api_key
        self.from_email = os.getenv("FROM_EMAIL")
        if not self.from_email:
            raise RuntimeError("FROM_EMAIL not set")

        self.subject = os.getenv("EMAIL_SUBJECT", "Welcome to Our App")
        self.html_template = os.getenv("EMAIL_HTML_TEMPLATE", "")
        self.host_url = os.getenv("HOST_URL", "http://localhost:3000")

    def send_email(self, to_email: EmailStr, token: str):
        try:
            link = f"{self.host_url}/setup-password?token={token}"

            html = self.html_template or """
                <h2>Welcome</h2>
                <p>Click below to set your password:</p>
                <a href="{{setup_link}}">Setup Password</a>
            """

            html = html.replace("{{setup_link}}", link)

            resend.Emails.send({
                "from": self.from_email,
                "to": to_email,
                "subject": self.subject,
                "html": html
            })

        except Exception as e:
            raise RuntimeError("Failed to send email") from e
        
    async def save_email_verification(self, user_id: PydanticObjectId, resend_id: str):
        
        email_verification = EmailVerificationModel(
            user_id=user_id,
            resend_id=resend_id,
        )
        try:
            await email_verification.insert()
        except Exception as e:
            raise RuntimeError("Failed to save email verification") from e