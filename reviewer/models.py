import uuid
from django.db import models
from django.conf import settings


class Instruction(models.Model):
    """
    User-specific instructions for news review.
    Each user can have multiple instructions that will be applied when reviewing news.

    - content: Original instruction text (Arabic) - shown to user
    - processed_content: Processed/translated instruction (English) - used by AI model
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='instructions'
    )
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    content = models.TextField(help_text="التعليمات الأصلية بالعربي")
    processed_content = models.TextField(
        blank=True,
        help_text="التعليمات المعالجة والمترجمة للإنجليزية - تُستخدم بواسطة الموديل"
    )
    is_active = models.BooleanField(default=True)
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['order', 'created_at']
        unique_together = ['user', 'title']

    def __str__(self):
        return f"{self.title} - {self.user.username}"


class NewsExample(models.Model):
    """
    User-specific examples of processed news.
    These examples help the AI understand how to apply the instructions.
    Each example contains the original news text and the expected output after processing.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='news_examples'
    )
    title = models.CharField(max_length=255)
    original_text = models.TextField(help_text="النص الأصلي للخبر قبل المعالجة")
    processed_text = models.TextField(help_text="النص بعد المعالجة (المخرج المتوقع)")
    notes = models.TextField(blank=True, help_text="ملاحظات إضافية حول هذا المثال")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['user', 'title']

    def __str__(self):
        return f"{self.title} - {self.user.username}"
