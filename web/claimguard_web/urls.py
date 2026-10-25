from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="audit:overview"), name="home"),
    path("audit/", include("audit.urls")),
    path("accounts/", include("allauth.urls")),
    path("admin/", admin.site.urls),
]
