from rest_framework import serializers


class BaseDocxUploadSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, allow_blank=True)
    file = serializers.FileField()


class InstructionUploadSerializer(BaseDocxUploadSerializer):
    pass


class ExampleUploadSerializer(BaseDocxUploadSerializer):
    pass


class MultiFileField(serializers.ListField):
    """Custom field to handle multiple file uploads."""
    child = serializers.FileField()

    def to_internal_value(self, data):
        # Handle both list and single file scenarios
        if not isinstance(data, list):
            data = [data]
        return super().to_internal_value(data)


class BulkDocxUploadSerializer(serializers.Serializer):
    """Serializer for uploading multiple DOCX files at once.

    Supports large files (10+ pages each) with async processing.
    Maximum 20 files per batch, each file can be up to 50MB.
    """
    files = MultiFileField(
        min_length=1,
        max_length=20,  # Maximum 20 files per batch
        help_text="Upload up to 20 DOCX files at once (max 50MB each)"
    )
    replace_existing = serializers.BooleanField(
        required=False,
        default=False,
        help_text="Replace existing documents with same titles"
    )

    def validate_files(self, files):
        """Validate file types and sizes."""
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
        ALLOWED_EXTENSIONS = ['.docx', '.doc']

        for file in files:
            # Check file extension
            filename = getattr(file, 'name', '')
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if f'.{ext}' not in ALLOWED_EXTENSIONS:
                raise serializers.ValidationError(
                    f"الملف '{filename}' غير مدعوم. الملفات المدعومة: {', '.join(ALLOWED_EXTENSIONS)}"
                )

            # Check file size
            if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
                raise serializers.ValidationError(
                    f"حجم الملف '{filename}' يتجاوز الحد الأقصى (50 ميجابايت)"
                )

        return files


class BulkGuidelinesUploadSerializer(BulkDocxUploadSerializer):
    """Serializer for bulk guidelines upload."""
    pass


class BulkExamplesUploadSerializer(BulkDocxUploadSerializer):
    """Serializer for bulk examples upload."""
    pass


class BatchStatusSerializer(serializers.Serializer):
    """Serializer for batch upload status response."""
    batch_id = serializers.UUIDField()
    status = serializers.CharField()
    total_files = serializers.IntegerField()
    processed_files = serializers.IntegerField()
    total_chunks_created = serializers.IntegerField()
    files = serializers.ListField(child=serializers.DictField())
    progress_percentage = serializers.FloatField(required=False)


class ReviewRequestSerializer(serializers.Serializer):
    news_text = serializers.CharField()


# =============================================================================
# DOCUMENT RETRIEVAL SERIALIZERS
# =============================================================================

class DocumentChunkSerializer(serializers.Serializer):
    """Serializer for individual document chunks."""
    id = serializers.IntegerField()
    document_type = serializers.CharField()
    title = serializers.CharField()
    source_name = serializers.CharField()
    order = serializers.IntegerField()
    text = serializers.CharField()
    metadata = serializers.DictField()
    created_at = serializers.DateTimeField()


class DocumentSummarySerializer(serializers.Serializer):
    """Serializer for document summary (grouped by title)."""
    title = serializers.CharField()
    source_name = serializers.CharField()
    document_type = serializers.CharField()
    total_chunks = serializers.IntegerField()
    created_at = serializers.DateTimeField()


class DocumentListResponseSerializer(serializers.Serializer):
    """Response serializer for listing documents."""
    total_documents = serializers.IntegerField()
    total_chunks = serializers.IntegerField()
    documents = DocumentSummarySerializer(many=True)


class DocumentDetailResponseSerializer(serializers.Serializer):
    """Response serializer for document detail with all chunks."""
    title = serializers.CharField()
    source_name = serializers.CharField()
    document_type = serializers.CharField()
    total_chunks = serializers.IntegerField()
    created_at = serializers.DateTimeField()
    chunks = DocumentChunkSerializer(many=True)


class UploadedFileSerializer(serializers.Serializer):
    """Serializer for uploaded file records."""
    id = serializers.IntegerField()
    filename = serializers.CharField()
    title = serializers.CharField()
    file_size = serializers.IntegerField()
    status = serializers.CharField()
    chunks_created = serializers.IntegerField()
    error_message = serializers.CharField(allow_blank=True)
    created_at = serializers.DateTimeField()


class BatchListSerializer(serializers.Serializer):
    """Serializer for batch upload list."""
    batch_id = serializers.UUIDField(source='id')
    document_type = serializers.CharField()
    status = serializers.CharField()
    total_files = serializers.IntegerField()
    processed_files = serializers.IntegerField()
    total_chunks_created = serializers.IntegerField()
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()

