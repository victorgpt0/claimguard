"""The `lucide` template tag, decorative by default.

Replaces the lucide package's tag (TEMPLATES OPTIONS "libraries" in settings.py) so every
icon is hidden from screen readers and skipped by keyboard focus unless it is given an
accessible name: `{% lucide "x" aria_label="Close" %}` renders a labelled image instead.
Every icon in the interface sits next to visible text or inside a labelled control, so
announcing it would only repeat or add noise (WCAG 1.1.1).
"""
from __future__ import annotations

from django import template
from lucide.templatetags.lucide import _render_icon

register = template.Library()


@register.simple_tag
def lucide(name: str, *, size: int | None = 24, **kwargs: object) -> str:
    if "aria_label" in kwargs:
        kwargs.setdefault("role", "img")
    else:
        kwargs.setdefault("aria_hidden", "true")
    kwargs.setdefault("focusable", "false")
    return _render_icon(name, size, **kwargs)
