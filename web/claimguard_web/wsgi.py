import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # the repository, for src/
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "claimguard_web.settings")

from django.core.wsgi import get_wsgi_application  # noqa: E402

application = get_wsgi_application()
