from rest_framework import serializers

from .models import Instruction, NewsExample
from .services import process_instruction


# =============================================================================
# INSTRUCTION SERIALIZERS (Dashboard)
# =============================================================================

class InstructionSerializer(serializers.ModelSerializer):
    """Serializer for user instructions CRUD operations."""

    class Meta:
        model = Instruction
        fields = [
            'id',
            'title',
            'description',
            'content',
            'processed_content',  # English version for AI model
            'is_active',
            'order',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'processed_content', 'created_at', 'updated_at']


class InstructionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating new instructions."""

    class Meta:
        model = Instruction
        fields = ['title', 'description', 'content', 'is_active', 'order']

    def create(self, validated_data):
        user = self.context['request'].user
        validated_data['user'] = user

        # Auto-set order if not provided
        if 'order' not in validated_data or validated_data['order'] == 0:
            max_order = Instruction.objects.filter(user=user).aggregate(
                max_order=serializers.models.Max('order')
            )['max_order'] or 0
            validated_data['order'] = max_order + 1

        # Process instruction content (translate/summarize to English)
        content = validated_data.get('content', '')
        title = validated_data.get('title', '')
        if content:
            try:
                processed = process_instruction(content, title)
                validated_data['processed_content'] = processed
            except Exception:
                # If processing fails, leave processed_content empty
                pass

        return super().create(validated_data)


class InstructionUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating instructions - re-processes if content changes."""

    class Meta:
        model = Instruction
        fields = ['title', 'description', 'content', 'is_active', 'order']

    def update(self, instance, validated_data):
        # Check if content or title changed - need to re-process
        content_changed = 'content' in validated_data and validated_data['content'] != instance.content
        title_changed = 'title' in validated_data and validated_data['title'] != instance.title

        if content_changed or title_changed:
            content = validated_data.get('content', instance.content)
            title = validated_data.get('title', instance.title)
            try:
                processed = process_instruction(content, title)
                validated_data['processed_content'] = processed
            except Exception:
                # If processing fails, keep old processed_content
                pass

        return super().update(instance, validated_data)


class InstructionReorderSerializer(serializers.Serializer):
    """Serializer for reordering instructions."""
    instructions = serializers.ListField(
        child=serializers.DictField(
            child=serializers.CharField()
        ),
        help_text="List of {id, order} objects"
    )

    def validate_instructions(self, value):
        for item in value:
            if 'id' not in item or 'order' not in item:
                raise serializers.ValidationError(
                    "Each item must have 'id' and 'order' fields"
                )
            try:
                int(item['order'])
            except (ValueError, TypeError):
                raise serializers.ValidationError(
                    f"Invalid order value: {item['order']}"
                )
        return value


# =============================================================================
# NEWS EXAMPLE SERIALIZERS (Dashboard)
# =============================================================================

class NewsExampleSerializer(serializers.ModelSerializer):
    """Serializer for news example CRUD operations."""

    class Meta:
        model = NewsExample
        fields = [
            'id',
            'title',
            'original_text',
            'processed_text',
            'notes',
            'is_active',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class NewsExampleCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating new news examples."""

    class Meta:
        model = NewsExample
        fields = ['title', 'original_text', 'processed_text', 'notes', 'is_active']

    def create(self, validated_data):
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)


# =============================================================================
# REVIEW REQUEST SERIALIZER
# =============================================================================

class ReviewRequestSerializer(serializers.Serializer):
    """Serializer for news review requests."""
    news_text = serializers.CharField()
