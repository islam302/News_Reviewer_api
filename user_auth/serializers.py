from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from django.contrib.auth.password_validation import validate_password
from .models import CustomUser
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer, TokenRefreshSerializer


class UserSerializer(serializers.ModelSerializer):

    class Meta:
        model = CustomUser
        fields = [
            'id',
            'username',
            'email',
            'role',
            'organization',
            'password',
        ]
        read_only_fields = ['id']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data['password'])
        return super().create(validated_data)

    def update(self, instance, validated_data):
        if 'password' in validated_data:
            validated_data['password'] = make_password(validated_data['password'])
        return super().update(instance, validated_data)


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serializer for user registration."""
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password', 'password_confirm', 'organization']

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError({
                'password_confirm': 'كلمات المرور غير متطابقة'
            })
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = CustomUser.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
            organization=validated_data.get('organization', ''),
        )
        # Generate API key for new user
        user.generate_api_key()
        return user


class APIKeySerializer(serializers.Serializer):
    """Serializer for API key response."""
    api_key = serializers.CharField(read_only=True)
    created_at = serializers.DateTimeField(source='api_key_created_at', read_only=True)


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for user profile."""
    has_api_key = serializers.SerializerMethodField()
    instructions_count = serializers.SerializerMethodField()

    class Meta:
        model = CustomUser
        fields = [
            'id',
            'username',
            'email',
            'organization',
            'role',
            'has_api_key',
            'instructions_count',
            'date_joined',
        ]
        read_only_fields = ['id', 'date_joined', 'has_api_key', 'instructions_count']

    def get_has_api_key(self, obj):
        return bool(obj.api_key)

    def get_instructions_count(self, obj):
        return obj.instructions.count()

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['username'] = user.username
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        user = self.user
        data['username'] = user.username
        data['organization'] = getattr(user, 'organization', '')
        data['email'] = getattr(user, 'email', '')
        data['role'] = getattr(user, 'role', '')

        return data

class CustomTokenRefreshSerializer(TokenRefreshSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)

        # ✅ أضف بيانات المستخدم
        refresh = self.token_class(attrs['refresh'])
        user_id = refresh['user_id']

        user = CustomUser.objects.get(id=user_id)
        data['username'] = user.username
        data['organization'] = getattr(user, 'organization', '')
        data['email'] = getattr(user, 'email', '')
        data['role'] = getattr(user, 'role', '')

        return data
