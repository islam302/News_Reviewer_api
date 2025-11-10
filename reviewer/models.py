from django.db import models


class DocumentChunk(models.Model):
    class DocumentType(models.TextChoices):
        GUIDELINE = 'guideline', 'Guideline'
        EXAMPLE = 'example', 'Example'

    document_type = models.CharField(max_length=16, choices=DocumentType.choices)
    title = models.CharField(max_length=255)
    source_name = models.CharField(max_length=255, blank=True)
    order = models.PositiveIntegerField(default=0)
    text = models.TextField()
    embedding = models.JSONField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['document_type', 'title', 'order', 'id']

    def __str__(self):
        return f"{self.document_type}:{self.title}#{self.order}"
