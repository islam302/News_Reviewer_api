from django.urls import path

from .views import ExampleUploadView, InstructionUploadView, ReviewNewsView

app_name = "reviewer"

urlpatterns = [
    path("guidelines/upload/", InstructionUploadView.as_view(), name="upload-guidelines"),
    path("examples/upload/", ExampleUploadView.as_view(), name="upload-examples"),
    path("review/", ReviewNewsView.as_view(), name="review-news"),
]

