# The ClaimGuard web application (docs/platform_plan.md). Dependencies come from uv.lock, so
# the image installs exactly the versions tested locally. The CPU worker (phase 3) extends
# this image with the ml group.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group web --group sha
COPY . .
RUN DJANGO_DEBUG=false DJANGO_SECRET_KEY=build-only python web/manage.py collectstatic --noinput

EXPOSE 8000
CMD ["sh", "-c", "python web/manage.py migrate --noinput && gunicorn --chdir web claimguard_web.wsgi --bind 0.0.0.0:8000 --workers 3"]
