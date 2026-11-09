# The ClaimGuard images (docs/platform_plan.md). Dependencies come from uv.lock, so each image
# installs exactly the versions tested locally. Two targets share one base:
#   worker  the CPU worker: the web group plus the ml group (torch, transformers, peft, shap)
#   web     the web application and the claims platform (the default: it is the last stage)
# Both build on x86_64 and on ARM (the Oracle Ampere VM, deploy/README.md): uv.lock carries
# CPU-only torch wheels for both.
FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
# A slow connection otherwise fails on the larger wheels (numpy, pyarrow) at uv's 30 s default.
ENV UV_HTTP_TIMEOUT=300
WORKDIR /app
COPY pyproject.toml uv.lock ./

FROM base AS worker
# The cache mount keeps downloaded wheels between builds, so a failed build resumes rather
# than downloading everything again.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-default-groups --group web --group ml
COPY . .
CMD ["celery", "--workdir", "web", "-A", "claimguard_web", "worker", "-Q", "cpu", "-c", "1", "-n", "cpu@%h", "-l", "INFO"]

FROM base AS web
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-default-groups --group web --group sha
COPY . .
# src/ is the Tailwind source (built into css/app.css): not served, and its @import would fail
# the manifest's post-processing.
RUN DJANGO_DEBUG=false DJANGO_SECRET_KEY=build-only python web/manage.py collectstatic --noinput --ignore src

EXPOSE 8000
CMD ["sh", "-c", "python web/manage.py migrate --noinput && gunicorn --chdir web claimguard_web.wsgi --bind 0.0.0.0:8000 --workers 3"]
