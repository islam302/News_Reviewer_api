from rest_framework import serializers


class BaseDocxUploadSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, allow_blank=True)
    file = serializers.FileField()


class InstructionUploadSerializer(BaseDocxUploadSerializer):
    pass


class ExampleUploadSerializer(BaseDocxUploadSerializer):
    pass


class ReviewRequestSerializer(serializers.Serializer):
    news_text = serializers.CharField()
    top_guidelines = serializers.IntegerField(required=False, min_value=1, max_value=100)
    top_examples = serializers.IntegerField(required=False, min_value=1, max_value=100)

