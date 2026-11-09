from django.urls import path

from . import batch_views, views

app_name = "audit"
urlpatterns = [
    path("", views.overview, name="overview"),
    path("queue/", views.queue, name="queue"),
    path("reports/<int:pk>/", views.report_detail, name="report"),
    path("reports/<int:pk>/decide/", views.decide, name="decide"),
    path("model/", views.model_performance, name="model"),
    path("feedback/", views.feedback_loop, name="feedback"),
    path("batches/", batch_views.batch_list, name="batches"),
    path("batches/upload/", batch_views.upload, name="upload"),
    path("batches/template.csv", batch_views.upload_template, name="upload_template"),
    path("batches/<int:pk>/", batch_views.batch_detail, name="batch"),
    path("batches/<int:pk>/process/", batch_views.start_job, name="start_job"),
    path("batches/<int:pk>/summary.md", batch_views.batch_summary_markdown, name="batch_summary"),
    path("batches/<int:pk>/reports.csv", batch_views.batch_reports_csv, name="batch_reports"),
    path("jobs/<int:pk>/progress/", batch_views.job_progress, name="job_progress"),
]
