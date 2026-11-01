from django.contrib import admin

from .models import AuditReport, Batch, Claim, ClassifierVerdict, Job


@admin.register(Batch)
class BatchAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "claims_count", "uploaded_by", "uploaded_at", "processed_at")
    list_filter = ("status",)


@admin.register(Claim)
class ClaimAdmin(admin.ModelAdmin):
    list_display = ("claim_ref", "batch", "provider_id", "icd11_code", "claim_amount", "service_date")
    list_filter = ("batch",)
    search_fields = ("claim_ref", "provider_id", "icd11_code")


@admin.register(AuditReport)
class AuditReportAdmin(admin.ModelAdmin):
    list_display = ("id", "claim", "status", "reviewed_by", "reviewed_at", "created_at")
    list_filter = ("status",)
    readonly_fields = ("claim", "verdict", "shap_attribution", "llm_narrative", "flag_sources", "reviewed_by",
                       "reviewed_at", "created_at")


admin.site.register(ClassifierVerdict)


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("id", "batch", "queue", "status", "stage", "done", "total", "worker", "created_by", "created_at")
    list_filter = ("status", "queue")
admin.site.site_header = "ClaimGuard administration"
