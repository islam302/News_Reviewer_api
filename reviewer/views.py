from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DocumentChunk
from .serializers import (
    ExampleUploadSerializer,
    InstructionUploadSerializer,
    ReviewRequestSerializer,
)
from .services import (
    delete_document,
    generate_review,
    get_document_detail,
    ingest_docx,
    list_documents,
    retrieve_similar_chunks,
)


class InstructionUploadView(APIView):
    """Upload a single guidelines DOCX file."""

    def post(self, request, *args, **kwargs):
        serializer = InstructionUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            chunks = ingest_docx(
                file_obj=serializer.validated_data["file"],
                title=serializer.validated_data.get("title"),
                document_type=DocumentChunk.DocumentType.GUIDELINE,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            {
                "detail": "تم تحديث التعليمات بنجاح.",
                "chunks_created": len(chunks),
                "title": chunks[0].title if chunks else serializer.validated_data.get("title"),
            },
            status=status.HTTP_201_CREATED,
        )


class ExampleUploadView(APIView):
    """Upload a single examples DOCX file."""

    def post(self, request, *args, **kwargs):
        serializer = ExampleUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            chunks = ingest_docx(
                file_obj=serializer.validated_data["file"],
                title=serializer.validated_data.get("title"),
                document_type=DocumentChunk.DocumentType.EXAMPLE,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            {
                "detail": "تم تحديث الأمثلة بنجاح.",
                "chunks_created": len(chunks),
                "title": chunks[0].title if chunks else serializer.validated_data.get("title"),
            },
            status=status.HTTP_201_CREATED,
        )


class ReviewNewsView(APIView):
    """Review news using uploaded guidelines and examples."""

    def post(self, request, *args, **kwargs):
        serializer = ReviewRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        news_text = serializer.validated_data["news_text"]

        try:
            # Retrieve ALL chunks (no limit) to use full file content
            guideline_chunks = retrieve_similar_chunks(
                query_text=news_text,
                document_type=DocumentChunk.DocumentType.GUIDELINE,
                limit=None,  # Use ALL guidelines
            )
            example_chunks = retrieve_similar_chunks(
                query_text=news_text,
                document_type=DocumentChunk.DocumentType.EXAMPLE,
                limit=None,  # Use ALL examples
            )
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if not guideline_chunks:
            return Response(
                {"detail": "الرجاء رفع ملف التعليمات قبل محاولة مراجعة الأخبار."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            review_text = generate_review(
                news_text=news_text,
                guideline_chunks=guideline_chunks,
                example_chunks=example_chunks,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"review": review_text}, status=status.HTTP_200_OK)


# =============================================================================
# DOCUMENT RETRIEVAL VIEWS
# =============================================================================

class GuidelinesListView(APIView):
    """List or delete all uploaded guidelines documents."""

    def get(self, request):
        result = list_documents(document_type=DocumentChunk.DocumentType.GUIDELINE)
        return Response(result, status=status.HTTP_200_OK)

    def delete(self, request):
        """Delete ALL guidelines documents."""
        deleted_count = DocumentChunk.objects.filter(
            document_type=DocumentChunk.DocumentType.GUIDELINE
        ).delete()[0]
        return Response(
            {
                "detail": "تم حذف جميع التعليمات بنجاح.",
                "chunks_deleted": deleted_count,
            },
            status=status.HTTP_200_OK
        )


class ExamplesListView(APIView):
    """List or delete all uploaded examples documents."""

    def get(self, request):
        result = list_documents(document_type=DocumentChunk.DocumentType.EXAMPLE)
        return Response(result, status=status.HTTP_200_OK)

    def delete(self, request):
        """Delete ALL examples documents."""
        deleted_count = DocumentChunk.objects.filter(
            document_type=DocumentChunk.DocumentType.EXAMPLE
        ).delete()[0]
        return Response(
            {
                "detail": "تم حذف جميع الأمثلة بنجاح.",
                "chunks_deleted": deleted_count,
            },
            status=status.HTTP_200_OK
        )


class DocumentDetailView(APIView):
    """Get or delete a specific document by title."""

    def get(self, request, title):
        doc_type_param = request.query_params.get('type')
        document_type = None
        if doc_type_param == 'guideline':
            document_type = DocumentChunk.DocumentType.GUIDELINE
        elif doc_type_param == 'example':
            document_type = DocumentChunk.DocumentType.EXAMPLE

        result = get_document_detail(title=title, document_type=document_type)

        if result is None:
            return Response(
                {"detail": f"المستند '{title}' غير موجود."},
                status=status.HTTP_404_NOT_FOUND
            )

        return Response(result, status=status.HTTP_200_OK)

    def delete(self, request, title):
        doc_type_param = request.query_params.get('type')
        document_type = None
        if doc_type_param == 'guideline':
            document_type = DocumentChunk.DocumentType.GUIDELINE
        elif doc_type_param == 'example':
            document_type = DocumentChunk.DocumentType.EXAMPLE

        deleted_count = delete_document(title=title, document_type=document_type)

        if deleted_count == 0:
            return Response(
                {"detail": f"المستند '{title}' غير موجود."},
                status=status.HTTP_404_NOT_FOUND
            )

        return Response(
            {
                "detail": f"تم حذف المستند '{title}' بنجاح.",
                "chunks_deleted": deleted_count,
            },
            status=status.HTTP_200_OK
        )
