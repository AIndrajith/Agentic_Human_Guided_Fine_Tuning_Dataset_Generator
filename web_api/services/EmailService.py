import asyncio
import logging

import resend

from web_api.core.config import Settings
from web_api.errors import InternalError

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE = """
    <h2>Welcome</h2>
    <p>Click below to set your password:</p>
    <a href="{{setup_link}}">Setup Password</a>
"""


class EmailService:
    def __init__(self, settings: Settings):
        self.api_key = settings.RESEND_API_KEY.get_secret_value() if settings.RESEND_API_KEY else None
        self.from_email = settings.FROM_EMAIL
        self.subject = settings.EMAIL_SUBJECT
        self.html_template = settings.EMAIL_HTML_TEMPLATE or _DEFAULT_TEMPLATE
        self.host_url = settings.HOST_URL.rstrip("/")
        if self.api_key:
            resend.api_key = self.api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.from_email)

    def setup_link(self, token: str) -> str:
        return f"{self.host_url}/setup-password?token={token}"

    async def send_invite(self, to_email: str, token: str) -> str | None:
        """Send the setup-password email. Returns the Resend message id.

        Dev mode (RESEND_API_KEY or FROM_EMAIL unset): logs the link and returns None.
        """
        link = self.setup_link(token)
        if not self.enabled:
            logger.warning("Email not configured; setup link for %s: %s", to_email, link)
            return None

        html = self.html_template.replace("{{setup_link}}", link)
        try:
            # resend's SDK is blocking; keep it off the event loop
            response = await asyncio.to_thread(
                resend.Emails.send,
                {"from": self.from_email, "to": to_email, "subject": self.subject, "html": html},
            )
        except Exception as e:
            logger.exception("Failed to send invite email to %s", to_email)
            raise InternalError("Failed to send invite email") from e
        return response["id"]
