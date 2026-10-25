from django.urls import path

from . import views

app_name = "audit"
urlpatterns = [
    path("", views.overview, name="overview"),
    path("queue/", views.queue, name="queue"),
    path("reports/<int:pk>/", views.report_detail, name="report"),
    path("reports/<int:pk>/decide/", views.decide, name="decide"),
    path("model/", views.model_performance, name="model"),
    path("feedback/", views.feedback_loop, name="feedback"),
]
