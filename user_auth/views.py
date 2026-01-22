from rest_framework import generics, viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from django.contrib.auth import get_user_model
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_decode
from django.utils.timezone import localtime

from .models import CustomUser
from .serializers import (
    UserSerializer,
    CustomTokenObtainPairSerializer,
    CustomTokenRefreshSerializer,
    UserRegistrationSerializer,
    APIKeySerializer,
    UserProfileSerializer,
)
from .permissions import IsSuperUser
from .utils import send_reset_password_email


User = get_user_model()


# =============================================================================
# AUTH VIEWS
# =============================================================================

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


class CustomTokenRefreshView(TokenRefreshView):
    serializer_class = CustomTokenRefreshSerializer


class RegisterView(APIView):
    """Register a new user account."""
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({
                'message': 'تم إنشاء الحساب بنجاح',
                'user': {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'organization': user.organization,
                },
                'api_key': user.api_key,
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# =============================================================================
# USER PROFILE VIEWS
# =============================================================================

class UserProfileView(APIView):
    """Get or update current user's profile."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserProfileSerializer(request.user)
        return Response(serializer.data)

    def patch(self, request):
        serializer = UserProfileSerializer(
            request.user,
            data=request.data,
            partial=True
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# =============================================================================
# API KEY MANAGEMENT VIEWS
# =============================================================================

class APIKeyView(APIView):
    """Get, generate, or revoke API key."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get current API key (if exists)."""
        if request.user.api_key:
            return Response({
                'api_key': request.user.api_key,
                'created_at': request.user.api_key_created_at,
            })
        return Response({
            'api_key': None,
            'message': 'لا يوجد مفتاح API. استخدم POST لإنشاء مفتاح جديد.'
        })

    def post(self, request):
        """Generate a new API key (replaces existing)."""
        api_key = request.user.generate_api_key()
        return Response({
            'api_key': api_key,
            'created_at': request.user.api_key_created_at,
            'message': 'تم إنشاء مفتاح API جديد بنجاح',
        }, status=status.HTTP_201_CREATED)

    def delete(self, request):
        """Revoke the current API key."""
        if not request.user.api_key:
            return Response({
                'message': 'لا يوجد مفتاح API لإلغائه'
            }, status=status.HTTP_400_BAD_REQUEST)

        request.user.revoke_api_key()
        return Response({
            'message': 'تم إلغاء مفتاح API بنجاح'
        })


# =============================================================================
# USER MANAGEMENT VIEWS (Admin)
# =============================================================================

class UserViewSet(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsSuperUser]

    def get_queryset(self):
        return CustomUser.objects.filter(role="user")


# =============================================================================
# PASSWORD RESET VIEWS
# =============================================================================

class PasswordResetRequestView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get("email")
        if not email:
            return Response({"error": "يرجى إدخال البريد الإلكتروني"}, status=400)
        try:
            user = User.objects.get(email=email)
            success = send_reset_password_email(user)
            if success:
                return Response({"message": "تم إرسال بريد إعادة تعيين كلمة المرور بنجاح."})
            else:
                return Response({"error": "فشل الارسال يرجي التحقق من الايميل"}, status=500)

        except User.DoesNotExist:
            return Response({"error": "هذا المستخدم غير موجود"}, status=404)


class PasswordResetConfirmAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, uidb64, token):
        try:
            uid = urlsafe_base64_decode(uidb64).decode()
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            return Response({"error": "Invalid UID"}, status=400)

        if not default_token_generator.check_token(user, token):
            return Response({"error": "Invalid or expired token"}, status=400)

        password = request.data.get("password")
        if not password:
            return Response({"error": "Password is required"}, status=400)

        user.set_password(password)
        user.save()

        return Response({"message": "Password has been reset successfully."})