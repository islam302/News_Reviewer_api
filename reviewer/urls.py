from django.urls import path

from .views import (
    # Upload views
    ExampleUploadView,
    InstructionUploadView,
    # Document retrieval views
    GuidelinesListView,
    ExamplesListView,
    DocumentDetailView,
    # Review view
    ReviewNewsView,
)

app_name = "reviewer"

urlpatterns = [
    # Upload endpoints
    path("guidelines/upload/", InstructionUploadView.as_view(), name="upload-guidelines"),
    path("examples/upload/", ExampleUploadView.as_view(), name="upload-examples"),

    # List uploaded documents
    path("guidelines/", GuidelinesListView.as_view(), name="list-guidelines"),
    path("examples/", ExamplesListView.as_view(), name="list-examples"),

    # Document detail (get/delete by title)
    path("documents/<str:title>/", DocumentDetailView.as_view(), name="document-detail"),

    # Review endpoint
    path("review/", ReviewNewsView.as_view(), name="review-news"),
]
