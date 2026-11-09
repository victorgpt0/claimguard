from django.urls import path

from . import views

app_name = "ingestion"
urlpatterns = [
    path("", views.pipeline, name="pipeline"),
    path("live/", views.pipeline_live, name="live"),
    path("sync/", views.sync_now, name="sync"),
    path("periods/<int:pk>/close/", views.close_now, name="close"),
    path("feed/toggle/", views.toggle_feed, name="toggle"),
    path("clock/", views.clock, name="clock"),
]
