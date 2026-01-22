import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from django.conf import settings
import os
from dotenv import load_dotenv
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes

load_dotenv()

BREVO_API_KEY = os.getenv("BREVO_API_KEY")
EMAIL_SENDER = "dev@una-oic.org"

def generate_reset_link(user):
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = default_token_generator.make_token(user)
    reset_link = f"https://mailer-delta-one.vercel.app/auth/reset-password?token={token}&uid={uid}"
    return reset_link

def send_reset_password_email(user):
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = BREVO_API_KEY
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

    reset_link = generate_reset_link(user)
    email_content = f"""
    <html>
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Tajawal&display=swap" rel="stylesheet">
      </head>
      <body style="font-family: 'Tajawal', sans-serif; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
          <h2 style="color: #4CAF50; text-align: center;">🔐 طلب إعادة تعيين كلمة المرور</h2>
          <p style="font-size: 18px; color: #333;">مرحباً <strong>{user.username}</strong>،</p>
          <p style="font-size: 16px; color: #555;">لقد تلقينا طلباً لإعادة تعيين كلمة المرور الخاصة بك.</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_link}" 
               style="background-color: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; font-size: 18px; border-radius: 5px;">
               إعادة تعيين كلمة المرور
            </a>
          </div>
          <p style="font-size: 14px; color: #777;">إذا لم تطلب هذا الإجراء، يمكنك تجاهل هذه الرسالة ولن يتم تغيير أي شيء في حسابك.</p>
          <hr style="margin: 30px 0;">
          <p style="font-size: 14px; text-align: center; color: #aaa;">فريق دعم UNA Email System</p>
        </div>
      </body>
    </html>
    """

    send_email = {
        "sender": {"email": EMAIL_SENDER, "name": "UNA Email System"},
        "to": [{"email": user.email, "name": user.username}],
        "subject": "🔒 رابط إعادة تعيين كلمة المرور الخاصة بك",
        "htmlContent": email_content,
        "trackClicks": False,
        "trackOpens": False
    }

    try:
        api_instance.send_transac_email(send_email)
        print(f"📨 تم إرسال رسالة إعادة تعيين كلمة المرور إلى {user.email}")
        return True
    except ApiException as e:
        print(f"❌ خطأ أثناء إرسال رسالة إعادة التعيين: {e}")
        return False
