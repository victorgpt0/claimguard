"""User management for the Administrator (#56): Django's admin site, restricted to the
fields this project uses. Deleting is disabled: accounts are deactivated."""
from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin
from django.core.exceptions import ValidationError

from activity.models import ActivityLog

from .models import User


@admin.register(User)
class ClaimGuardUserAdmin(UserAdmin):
    list_display = ("username", "role", "email", "is_active", "last_login", "date_joined")
    list_filter = ("role", "is_active")
    search_fields = ("username", "email")
    ordering = ("username",)
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        ("Role and Google sign-in", {"fields": ("role", "email", "is_active")}),
        ("History", {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = ((None, {"classes": ("wide",),
                             "fields": ("username", "role", "email", "password1", "password2")}),)
    readonly_fields = ("last_login", "date_joined")

    def has_delete_permission(self, request, obj=None):
        return False

    def save_model(self, request, obj, form, change):
        before = User.objects.filter(pk=obj.pk).first() if change else None
        if change and obj.pk == request.user.pk and not obj.is_active:
            messages.error(request, "You cannot deactivate your own account.")
            return
        try:
            obj.full_clean(exclude=["password"])
            super().save_model(request, obj, form, change)
        except ValidationError as exc:
            messages.error(request, "; ".join(exc.messages))
            return
        if before is None:
            ActivityLog.record(ActivityLog.Action.USER_CREATED, request.user, f"{obj.username} ({obj.role})")
            return
        if before.is_active != obj.is_active:
            action = ActivityLog.Action.USER_REACTIVATED if obj.is_active else ActivityLog.Action.USER_DEACTIVATED
            ActivityLog.record(action, request.user, obj.username)
        if before.email != obj.email:
            ActivityLog.record(ActivityLog.Action.EMAIL_LINKED, request.user, f"{obj.username}: {obj.email or 'unlinked'}")
        if before.role != obj.role:
            ActivityLog.record(ActivityLog.Action.ROLE_CHANGED, request.user, f"{obj.username}: {obj.role}")
