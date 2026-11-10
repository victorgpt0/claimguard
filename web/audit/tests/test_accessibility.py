"""Accessibility checks (WCAG 2.2 AA) that can be made without a browser: icons hidden from
screen readers, the skip link's target, the current page in the navigation, error messages
tied to their fields, filters that do not submit on change, and AA contrast for the theme
colours that carry text or mark a control. Keyboard order, focus visibility and layout at
360 px were checked in Chromium (10 Nov 2026)."""
import re
from pathlib import Path

from django.core.files.uploadedfile import SimpleUploadedFile
from django.template import Context, Template
from django.test import Client
from django.urls import reverse

CSS = Path(__file__).resolve().parents[2] / "static" / "src" / "app.css"


def test_icons_are_decorative_unless_labelled():
    hidden = Template('{% load lucide %}{% lucide "x" size=16 %}').render(Context())
    assert 'aria-hidden="true"' in hidden and 'focusable="false"' in hidden
    named = Template('{% load lucide %}{% lucide "x" aria_label="Close" %}').render(Context())
    assert 'aria-label="Close"' in named and 'role="img"' in named and "aria-hidden" not in named


def test_skip_link_target_and_current_page(auditor, reports, pipeline_files, db):
    for html in (auditor.get(reverse("audit:queue")).content.decode(),
                 Client().get(reverse("account_login")).content.decode()):
        assert 'href="#main"' in html and 'id="main"' in html and html.count("<main") == 1
    html = auditor.get(reverse("audit:queue")).content.decode()
    assert re.search(r'aria-current="page"\s+href="/audit/queue/"', html)
    # Closed on small screens, the drawer is invisible as well as off screen, so Tab skips it.
    assert re.search(r'<aside id="sidebar" class="invisible [^"]*lg:visible', html)
    assert 'role="menu"' not in html  # disclosures, not ARIA menus that promise arrow keys


def test_sign_in_errors_are_tied_to_the_fields(db):
    html = Client().post(reverse("account_login"), {"login": "nobody", "password": "wrong"}).content.decode()
    assert 'id="login-error"' in html
    assert len(re.findall(r'aria-invalid="true" aria-describedby="login-error"', html)) == 2


def test_upload_errors_are_tied_to_the_fields(admin, pipeline_files):
    html = admin.post(reverse("audit:upload"), {"file": SimpleUploadedFile("x.txt", b"hello")}).content.decode()
    for field in ("file", "synthetic"):
        assert f'aria-describedby="id_{field}_error"' in html and f'id="id_{field}_error"' in html


def test_queue_filters_submit_on_a_button(auditor, reports, pipeline_files):
    html = auditor.get(reverse("audit:queue")).content.decode()
    assert "onchange" not in html and ">Apply</button>" in html
    assert "textinconsistent" not in html


def _tokens(block):
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-f]{6})", block))


def _contrast(a, b):
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_theme_colours_meet_aa_contrast():
    css = CSS.read_text()
    light = _tokens(css[css.index(":root {"):css.index("[data-theme=dark] {")])
    dark = _tokens(css[css.index("[data-theme=dark] {"):css.index("@theme inline")])
    text = [("ink", bg) for bg in ("surface", "canvas")] + [("muted", bg) for bg in ("surface", "canvas", "subtle", "hover")] + [
        ("red", "red-soft"), ("amber", "amber-soft"), ("green", "green-soft"), ("blue", "blue-soft"),
        ("red", "surface"), ("green", "surface"), ("amber", "surface"), ("blue", "surface"),
        ("on-primary", "primary"), ("accent-ink", "surface"), ("accent-ink", "accent-soft")]
    ui = [("control", bg) for bg in ("surface", "subtle", "canvas")] + [("accent-ink", bg) for bg in ("canvas", "hover")]
    for name, theme in (("light", light), ("dark", dark)):
        for fg, bg in text:
            assert _contrast(theme[fg], theme[bg]) >= 4.5, (name, fg, bg)
        for fg, bg in ui:
            assert _contrast(theme[fg], theme[bg]) >= 3, (name, fg, bg)
    assert ":focus-visible { outline: 2px solid var(--accent-ink)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
