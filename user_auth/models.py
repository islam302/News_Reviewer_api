from django.contrib.auth.models import AbstractUser
from django.db import models
import uuid
import secrets
from django.conf import settings


class CustomUser(AbstractUser):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organization = models.CharField(max_length=255, blank=True, null=True)

    ROLES = (
        ('admin', 'Admin'),
        ('user', 'User')
    )
    role = models.CharField(max_length=20, choices=ROLES, default='user')

    # API Key for external API access
    api_key = models.CharField(max_length=64, unique=True, blank=True, null=True)
    api_key_created_at = models.DateTimeField(null=True, blank=True)

    def generate_api_key(self):
        """Generate a new API key for the user."""
        from django.utils import timezone
        self.api_key = f"nra_{secrets.token_hex(28)}"  # nra = news reviewer api
        self.api_key_created_at = timezone.now()
        self.save(update_fields=['api_key', 'api_key_created_at'])
        return self.api_key

    def revoke_api_key(self):
        """Revoke the current API key."""
        self.api_key = None
        self.api_key_created_at = None
        self.save(update_fields=['api_key', 'api_key_created_at'])

    @property
    def is_admin_dynamic(self):
        return self.role == "admin"

    def __str__(self):
        return self.username

