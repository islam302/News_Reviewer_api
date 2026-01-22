# Generated manually for cleanup

import django.db.models.deletion
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviewer', '0003_fileuploadbatch_uploadedfile_instruction'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # Remove legacy models (order matters - remove dependent models first)
        migrations.DeleteModel(
            name='UploadedFile',
        ),
        migrations.DeleteModel(
            name='FileUploadBatch',
        ),
        migrations.DeleteModel(
            name='DocumentChunk',
        ),
        # Add NewsExample model
        migrations.CreateModel(
            name='NewsExample',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=255)),
                ('original_text', models.TextField(help_text='النص الأصلي للخبر قبل المعالجة')),
                ('processed_text', models.TextField(help_text='النص بعد المعالجة (المخرج المتوقع)')),
                ('notes', models.TextField(blank=True, help_text='ملاحظات إضافية حول هذا المثال')),
                ('is_active', models.BooleanField(default=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='news_examples', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
                'unique_together': {('user', 'title')},
            },
        ),
    ]
