"""Role checks for views (IR-03, NFR-03): a signed-out visitor is sent to sign in; a signed-in
user of the wrong role gets 403. The role, not the URL, decides what a user can reach."""
from functools import wraps

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied


def role_required(*roles):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapper(request, *args, **kwargs):
            if request.user.role not in roles:
                raise PermissionDenied("Your role cannot open this page.")
            return view(request, *args, **kwargs)
        return wrapper
    return decorator


auditor_required = role_required("Auditor")
administrator_required = role_required("Administrator")
any_role_required = role_required("Auditor", "Administrator")
