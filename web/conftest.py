import pytest


@pytest.fixture
def make_user(db):
    from accounts.models import User

    def make(username="auditor", role="Auditor", password="auditor-pass-1", **fields):
        user = User(username=username, role=role, **fields)
        user.set_password(password)
        user.save()
        return user
    return make
