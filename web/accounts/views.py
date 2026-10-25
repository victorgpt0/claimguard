from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required
def home(request):
    """Placeholder landing page until the screens of phase 2 replace it."""
    return render(request, "home.html")
