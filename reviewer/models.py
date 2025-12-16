import uuid
from django.db import models


class FileUploadBatch(models.Model):
    """Tracks batch uploads of multiple files."""

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document_type = models.CharField(max_length=16)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    total_files = models.PositiveIntegerField(default=0)
    processed_files = models.PositiveIntegerField(default=0)
    total_chunks_created = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Batch {self.id} - {self.document_type} ({self.status})"


class UploadedFile(models.Model):
    """Tracks individual files within a batch upload."""

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    batch = models.ForeignKey(FileUploadBatch, on_delete=models.CASCADE, related_name='files')
    filename = models.CharField(max_length=255)
    title = models.CharField(max_length=255, blank=True)
    file_size = models.PositiveIntegerField(default=0)  # Size in bytes
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    chunks_created = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.filename} ({self.status})"


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
