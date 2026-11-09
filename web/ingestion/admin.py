from django.contrib import admin

from .models import IngestionRun, RejectedClaim


@admin.register(IngestionRun)
class IngestionRunAdmin(admin.ModelAdmin):
    list_display = ("started_at", "status", "fetched", "accepted", "rejected", "duplicates", "triggered_by")
    list_filter = ("status",)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(RejectedClaim)
class RejectedClaimAdmin(admin.ModelAdmin):
    list_display = ("claim_ref", "paid_at", "run", "created_at")
    search_fields = ("claim_ref",)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
