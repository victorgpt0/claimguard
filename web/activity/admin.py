"""The Administrator's activity log view (#56): filter, search, read-only, CSV download."""
import csv

from django.contrib import admin
from django.http import HttpResponse

from .models import ActivityLog


@admin.register(ActivityLog)
class ActivityLogAdmin(admin.ModelAdmin):
    list_display = ("at", "user", "action", "detail")
    list_filter = ("action", "user")
    search_fields = ("detail", "user__username")
    date_hierarchy = "at"
    actions = ["download_csv"]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    @admin.action(description="Download selected entries as CSV")
    def download_csv(self, request, queryset):
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="activity_log.csv"'
        writer = csv.writer(response)
        writer.writerow(["at", "username", "action", "detail"])
        for entry in queryset.select_related("user"):
            writer.writerow([entry.at.isoformat(), entry.user.username if entry.user else "", entry.action, entry.detail])
        ActivityLog.record(ActivityLog.Action.DOWNLOAD, request.user, f"activity log, {queryset.count()} entries")
        return response
