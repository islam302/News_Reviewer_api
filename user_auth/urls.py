from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    CustomTokenObtainPairView,
    CustomTokenRefreshView,
    RegisterView,
    UserProfileView,
    APIKeyView,
    UserViewSet,
    PasswordResetRequestView,
    PasswordResetConfirmAPIView,
)

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')

urlpatterns = [
    # Registration
    path('register/', RegisterView.as_view(), name='register'),

    # Token authentication
    path('token/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', CustomTokenRefreshView.as_view(), name='token_refresh'),

    # User profile
    path('profile/', UserProfileView.as_view(), name='user_profile'),

    # API Key management
    path('api-key/', APIKeyView.as_view(), name='api_key'),

    # Password reset
    path('password-reset/', PasswordResetRequestView.as_view(), name='password_reset'),
    path('password-reset-confirm/<token>/<uidb64>/', PasswordResetConfirmAPIView.as_view(),
         name='password_reset_confirm_api'),

    # User management (admin)
    path('', include(router.urls)),
]