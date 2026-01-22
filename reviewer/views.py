from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

from django.db import transaction
from django.shortcuts import get_object_or_404

from .models import Instruction, NewsExample
from .serializers import (
    InstructionSerializer,
    InstructionCreateSerializer,
    InstructionUpdateSerializer,
    InstructionReorderSerializer,
    NewsExampleSerializer,
    NewsExampleCreateSerializer,
    ReviewRequestSerializer,
)
from .services import generate_review


# =============================================================================
# REVIEW ENDPOINT
# =============================================================================

class ReviewNewsView(APIView):
    """Review news using user's instructions and examples."""
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = ReviewRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        news_text = serializer.validated_data["news_text"]

        # Get user's active instructions
        user_instructions = Instruction.objects.filter(
            user=request.user,
            is_active=True
        ).order_by('order')

        if not user_instructions.exists():
            return Response(
                {"detail": "الرجاء إضافة تعليمات من لوحة التحكم قبل محاولة مراجعة الأخبار."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            review_text = generate_review(
                news_text=news_text,
                user=request.user,
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"review": review_text}, status=status.HTTP_200_OK)


# =============================================================================
# INSTRUCTION CRUD VIEWS (Dashboard)
# =============================================================================

class InstructionListCreateView(APIView):
    """List all instructions or create a new one."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """List all instructions for the current user."""
        instructions = Instruction.objects.filter(user=request.user)
        serializer = InstructionSerializer(instructions, many=True)
        return Response({
            'count': instructions.count(),
            'instructions': serializer.data
        })

    def post(self, request):
        """Create a new instruction."""
        serializer = InstructionCreateSerializer(
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            instruction = serializer.save()
            return Response(
                InstructionSerializer(instruction).data,
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class InstructionDetailView(APIView):
    """Get, update, or delete a specific instruction."""
    permission_classes = [IsAuthenticated]

    def get_object(self, pk, user):
        return get_object_or_404(Instruction, pk=pk, user=user)

    def get(self, request, pk):
        """Get instruction details."""
        instruction = self.get_object(pk, request.user)
        serializer = InstructionSerializer(instruction)
        return Response(serializer.data)

    def put(self, request, pk):
        """Update an instruction (re-processes if content changes)."""
        instruction = self.get_object(pk, request.user)
        serializer = InstructionUpdateSerializer(instruction, data=request.data, partial=True)
        if serializer.is_valid():
            updated = serializer.save()
            return Response(InstructionSerializer(updated).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """Delete an instruction."""
        instruction = self.get_object(pk, request.user)
        instruction.delete()
        return Response(
            {'detail': 'تم حذف التعليمة بنجاح'},
            status=status.HTTP_200_OK
        )


class InstructionReorderView(APIView):
    """Reorder instructions."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Reorder instructions by updating their order values."""
        serializer = InstructionReorderSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        instructions_data = serializer.validated_data['instructions']

        with transaction.atomic():
            for item in instructions_data:
                try:
                    instruction = Instruction.objects.get(
                        pk=item['id'],
                        user=request.user
                    )
                    instruction.order = int(item['order'])
                    instruction.save(update_fields=['order'])
                except Instruction.DoesNotExist:
                    return Response(
                        {'detail': f"التعليمة {item['id']} غير موجودة"},
                        status=status.HTTP_404_NOT_FOUND
                    )

        # Return updated list
        instructions = Instruction.objects.filter(user=request.user)
        return Response({
            'detail': 'تم تحديث ترتيب التعليمات بنجاح',
            'instructions': InstructionSerializer(instructions, many=True).data
        })


class InstructionToggleActiveView(APIView):
    """Toggle instruction active status."""
    permission_classes = [IsAuthenticated]

    def post(self, request, pk):
        """Toggle is_active status for an instruction."""
        instruction = get_object_or_404(Instruction, pk=pk, user=request.user)
        instruction.is_active = not instruction.is_active
        instruction.save(update_fields=['is_active'])
        return Response({
            'detail': 'تم تحديث حالة التعليمة بنجاح',
            'is_active': instruction.is_active
        })


# =============================================================================
# NEWS EXAMPLES CRUD VIEWS (Dashboard)
# =============================================================================

class NewsExampleListCreateView(APIView):
    """List all news examples or create a new one."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """List all news examples for the current user."""
        examples = NewsExample.objects.filter(user=request.user)
        serializer = NewsExampleSerializer(examples, many=True)
        return Response({
            'count': examples.count(),
            'examples': serializer.data
        })

    def post(self, request):
        """Create a new news example."""
        serializer = NewsExampleCreateSerializer(
            data=request.data,
            context={'request': request}
        )
        if serializer.is_valid():
            example = serializer.save()
            return Response(
                NewsExampleSerializer(example).data,
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class NewsExampleDetailView(APIView):
    """Get, update, or delete a specific news example."""
    permission_classes = [IsAuthenticated]

    def get_object(self, pk, user):
        return get_object_or_404(NewsExample, pk=pk, user=user)

    def get(self, request, pk):
        """Get news example details."""
        example = self.get_object(pk, request.user)
        serializer = NewsExampleSerializer(example)
        return Response(serializer.data)

    def put(self, request, pk):
        """Update a news example."""
        example = self.get_object(pk, request.user)
        serializer = NewsExampleSerializer(example, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """Delete a news example."""
        example = self.get_object(pk, request.user)
        example.delete()
        return Response(
            {'detail': 'تم حذف المثال بنجاح'},
            status=status.HTTP_200_OK
        )


class NewsExampleToggleActiveView(APIView):
    """Toggle news example active status."""
    permission_classes = [IsAuthenticated]

    def post(self, request, pk):
        """Toggle is_active status for a news example."""
        example = get_object_or_404(NewsExample, pk=pk, user=request.user)
        example.is_active = not example.is_active
        example.save(update_fields=['is_active'])
        return Response({
            'detail': 'تم تحديث حالة المثال بنجاح',
            'is_active': example.is_active
        })
