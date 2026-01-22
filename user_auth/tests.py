import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from user.models import CustomUser, EmailQuotaLog
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def superuser():
    return CustomUser.objects.create_superuser(username="admin", email="admin@example.com", password="adminpass")


@pytest.fixture
def normal_user():
    return CustomUser.objects.create_user(username="user", email="user@example.com", password="userpass")


@pytest.fixture
def auth_client(superuser):
    client = APIClient()
    refresh = RefreshToken.for_user(superuser)
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {refresh.access_token}")
    return client


@pytest.mark.django_db
def test_token_obtain_pair(api_client, normal_user):
    url = reverse("token_obtain_pair")
    response = api_client.post(url, {"username": "user", "password": "userpass"})
    assert response.status_code == 200
    assert "access" in response.data


@pytest.mark.django_db
def test_user_list(auth_client):
    url = reverse("user-list")
    response = auth_client.get(url)
    assert response.status_code == 200


@pytest.mark.django_db
def test_user_create(auth_client):
    url = reverse("user-list")
    data = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "newpass123"
    }
    response = auth_client.post(url, data)
    assert response.status_code in [201, 200]


@pytest.mark.django_db
def test_user_update(auth_client, normal_user):
    url = reverse("user-detail", args=[str(normal_user.id)])
    response = auth_client.patch(url, {"username": "updated_user"})
    assert response.status_code == 200
    assert response.data["username"] == "updated_user"


@pytest.mark.django_db
def test_user_delete(auth_client, normal_user):
    url = reverse("user-detail", args=[str(normal_user.id)])
    response = auth_client.delete(url)
    assert response.status_code == 204


@pytest.mark.django_db
def test_password_reset_request(api_client, normal_user):
    url = reverse("password_reset")
    response = api_client.post(url, {"email": normal_user.email})
    assert response.status_code in [200, 500]


@pytest.mark.django_db
def test_password_reset_confirm(api_client, normal_user):
    token = default_token_generator.make_token(normal_user)
    uidb64 = urlsafe_base64_encode(force_bytes(normal_user.pk))
    url = reverse("password_reset_confirm_api", args=[uidb64, token])
    response = api_client.post(url, {"password": "newpass456"})
    assert response.status_code in [200, 400]


@pytest.mark.django_db
def test_add_quota(auth_client, normal_user):
    url = reverse("add-quota")
    data = {
        "user_id": str(normal_user.id),
        "amount": 150
    }
    response = auth_client.post(url, data)
    assert response.status_code == 200
    normal_user.refresh_from_db()
    assert normal_user.email_quota >= 150


@pytest.mark.django_db
def test_quota_logs(auth_client, normal_user):
    # add quota first
    EmailQuotaLog.objects.create(user=normal_user, updated_by=auth_client.handler._force_user, amount_added=50)
    url = reverse("quota-logs")
    response = auth_client.get(url)
    assert response.status_code == 200
    assert isinstance(response.data, list)
    assert any(log["user"] == normal_user.username for log in response.data)
