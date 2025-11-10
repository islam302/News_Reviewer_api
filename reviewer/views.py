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
    generate_review,
    ingest_docx,
    retrieve_similar_chunks,
)


class InstructionUploadView(APIView):
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
    def post(self, request, *args, **kwargs):
        serializer = ReviewRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        news_text = serializer.validated_data["news_text"]
        top_guidelines = serializer.validated_data.get("top_guidelines")
        top_examples = serializer.validated_data.get("top_examples")

        try:
            guideline_chunks = retrieve_similar_chunks(
                query_text=news_text,
                document_type=DocumentChunk.DocumentType.GUIDELINE,
                limit=top_guidelines,
            )
            example_chunks = retrieve_similar_chunks(
                query_text=news_text,
                document_type=DocumentChunk.DocumentType.EXAMPLE,
                limit=top_examples,
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
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        def _format_chunk(item):
            return {
                "id": item.chunk.id,
                "title": item.chunk.title,
                "order": item.chunk.order,
                "similarity": round(item.similarity, 4),
            }

        return Response(
            {
                "review": review_text,
            },
            status=status.HTTP_200_OK,
        )
