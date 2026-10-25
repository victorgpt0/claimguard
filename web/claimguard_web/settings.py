"""Settings for the ClaimGuard web application (docs/platform_plan.md).

Everything that differs between a laptop, the Docker stack and a hosted deployment comes
from environment variables (a `.env` at the repository root is read if present), so the
code is the same everywhere. Without any variables it runs locally on SQLite with DEBUG on.
"""
from __future__ import annotations

import os
from pathlib import Path

import dj_database_url
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent          # web/
REPO_ROOT = BASE_DIR.parent                                # the repository, with src/
load_dotenv(REPO_ROOT / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes", "on"}


DEBUG = env_bool("DJANGO_DEBUG", True)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or ("dev-only-insecure-key" if DEBUG else "")
if not SECRET_KEY:
    raise RuntimeError("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is off")
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
CSRF_TRUSTED_ORIGINS = [o for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
    "accounts",
    "audit",
    "activity",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
]

if not DEBUG:  # runserver serves static files in development; WhiteNoise in deployment
    MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")

ROOT_URLCONF = "claimguard_web.urls"
WSGI_APPLICATION = "claimguard_web.wsgi.application"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
    ]},
}]

# PostgreSQL in Docker and in deployment (DATABASE_URL); SQLite for a quick local run and tests.
DATABASES = {"default": dj_database_url.config(
    default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}", conn_max_age=60)}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Accounts (NFR-03) ------------------------------------------------------------------------
AUTH_USER_MODEL = "accounts.User"
AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]
# Django's PBKDF2-SHA256 for new passwords; the legacy hasher only verifies passwords imported
# from the Streamlit-era SQLite database, and Django re-hashes them at the next sign-in.
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "accounts.hashers.LegacyClaimGuardHasher",
]
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 10}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]
LOGIN_URL = "account_login"
LOGIN_REDIRECT_URL = "home"
ACCOUNT_LOGOUT_REDIRECT_URL = "account_login"

# Sessions end after 30 idle minutes (the Streamlit dashboard's rule, #55).
SESSION_COOKIE_AGE = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "30")) * 60
SESSION_SAVE_EVERY_REQUEST = True
SESSION_EXPIRE_AT_BROWSER_CLOSE = True

# allauth: username/password sign-in, no self sign-up. Google proves identity only: a Google
# account signs in as the existing user whose verified email matches it, never as a new user
# (accounts.adapters, #61).
ACCOUNT_LOGIN_METHODS = {"username"}
ACCOUNT_SIGNUP_FIELDS = ["username*", "password1*", "password2*"]
ACCOUNT_ADAPTER = "accounts.adapters.NoSignupAccountAdapter"
SOCIALACCOUNT_ADAPTER = "accounts.adapters.LinkedAccountsOnlyAdapter"
SOCIALACCOUNT_EMAIL_AUTHENTICATION = True
SOCIALACCOUNT_EMAIL_AUTHENTICATION_AUTO_CONNECT = True
SOCIALACCOUNT_LOGIN_ON_GET = False
ACCOUNT_EMAIL_VERIFICATION = "none"
SOCIALACCOUNT_PROVIDERS = {"google": {
    "SCOPE": ["openid", "email"],
    "APP": {"client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
            "secret": os.environ.get("GOOGLE_CLIENT_SECRET", "")},
}}
GOOGLE_SIGN_IN_ENABLED = bool(os.environ.get("GOOGLE_CLIENT_ID"))

# --- Static files, time --------------------------------------------------------------------------
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"
                    if not DEBUG else "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", REPO_ROOT / "media"))
MEDIA_URL = "media/"
LANGUAGE_CODE = "en-gb"
TIME_ZONE = "Africa/Nairobi"
USE_I18N = True
USE_TZ = True

# --- Jobs (phase 3) -------------------------------------------------------------------------------
CELERY_BROKER_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = None   # progress lives in the Job table, not in Celery results
CELERY_TASK_ALWAYS_EAGER = env_bool("CELERY_TASK_ALWAYS_EAGER", False)

# The ML pipeline's paths and thresholds (FR-10): the same config.yaml the scripts read.
CLAIMGUARD_CONFIG = Path(os.environ.get("CLAIMGUARD_CONFIG", REPO_ROOT / "config" / "config.yaml"))

if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = True
