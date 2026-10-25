from django.apps import AppConfig


class AccountsConfig(AppConfig):
    name = "accounts"

    def ready(self):
        from activity import signals  # noqa: F401  (sign-in and sign-out logging)
