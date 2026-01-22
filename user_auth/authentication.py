from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from .models import CustomUser


class APIKeyAuthentication(BaseAuthentication):
    """
    Custom authentication using API key in header.

    Usage:
        Header: X-API-Key: nra_xxxxxxxxxxxx
    """
    keyword = 'X-API-Key'

    def authenticate(self, request):
        api_key = request.headers.get(self.keyword)

        if not api_key:
            return None  # No API key provided, try other auth methods

        try:
            user = CustomUser.objects.get(api_key=api_key, is_active=True)
        except CustomUser.DoesNotExist:
            raise AuthenticationFailed('Invalid API key')

        return (user, None)

    def authenticate_header(self, request):
        return self.keyword
