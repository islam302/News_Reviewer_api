from django.urls import path

from .views import (
    # Review view
    ReviewNewsView,
    # Instructions Dashboard views
    InstructionListCreateView,
    InstructionDetailView,
    InstructionReorderView,
    InstructionToggleActiveView,
    # News Examples Dashboard views
    NewsExampleListCreateView,
    NewsExampleDetailView,
    NewsExampleToggleActiveView,
)

app_name = "reviewer"

urlpatterns = [
    # ==========================================================================
    # INSTRUCTIONS DASHBOARD
    # ==========================================================================
    path("instructions/", InstructionListCreateView.as_view(), name="instruction-list-create"),
    path("instructions/<uuid:pk>/", InstructionDetailView.as_view(), name="instruction-detail"),
    path("instructions/reorder/", InstructionReorderView.as_view(), name="instruction-reorder"),
    path("instructions/<uuid:pk>/toggle/", InstructionToggleActiveView.as_view(), name="instruction-toggle"),

    # ==========================================================================
    # NEWS EXAMPLES DASHBOARD
    # ==========================================================================
    path("examples/", NewsExampleListCreateView.as_view(), name="example-list-create"),
    path("examples/<uuid:pk>/", NewsExampleDetailView.as_view(), name="example-detail"),
    path("examples/<uuid:pk>/toggle/", NewsExampleToggleActiveView.as_view(), name="example-toggle"),

    # ==========================================================================
    # REVIEW ENDPOINT (Uses user's instructions and examples)
    # ==========================================================================
    path("review/", ReviewNewsView.as_view(), name="review-news"),
]
