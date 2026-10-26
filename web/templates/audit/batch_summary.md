{% autoescape off %}# Batch summary: #{{ batch.pk }} {{ batch.name }}

Generated {{ generated|date:"Y-m-d H:i" }} by {{ user.username }}. Synthetic data only (DR-06).

| | |
|---|---|
| Status | {{ batch.get_status_display }} |
| Uploaded | {{ batch.uploaded_at|date:"Y-m-d H:i" }}{% if batch.uploaded_by %} by {{ batch.uploaded_by.username }}{% endif %} |
| Processed | {% if batch.processed_at %}{{ batch.processed_at|date:"Y-m-d H:i" }}{% else %}-{% endif %} |
{% if batch.source_sha256 %}| File SHA-256 | `{{ batch.source_sha256 }}` |
{% endif %}
## Claims through the pipeline

| Stage | Claims |
|---|---:|
| Received | {{ s.claims }} |
| Flagged at stage 1 | {{ s.flagged }} |
{% if s.flagged_by %}| - by Isolation Forest | {{ s.flagged_by.isolation_forest }} |
| - by the autoencoder | {{ s.flagged_by.autoencoder }} |
| - by the text-consistency screen | {{ s.flagged_by.text_screen }} |
{% endif %}| Cleared by the classifier (logged only) | {{ s.cleared }} |
| Queued: text inconsistent | {{ s.lanes.text_inconsistent }} |
| Queued: structural anomaly | {{ s.lanes.structural_anomaly }} |
| Narratives generated | {{ s.narratives }} |
{% if s.narrative_failures is not None %}| Narratives failed | {{ s.narrative_failures }} |
{% endif %}
Amount claimed: {{ amount_claimed }}. Amount under audit (queued claims): {{ amount_under_audit }}.

## Review progress

| Pending | Approved | Rejected | Escalated |
|---:|---:|---:|---:|
| {{ s.review.pending }} | {{ s.review.approved }} | {{ s.review.rejected }} | {{ s.review.escalated }} |
{% if s.reviewers %}
| Reviewer | Approved | Rejected | Escalated |
|---|---:|---:|---:|
{% for name, d in s.reviewers.items %}| {{ name }} | {{ d.approved }} | {{ d.rejected }} | {{ d.escalated }} |
{% endfor %}{% endif %}
## Queued claims by provider (top {{ top }})
{% if s.top_providers %}
| Provider | Claims | Amount (KES) |
|---|---:|---:|
{% for p in s.top_providers %}| {{ p.claim__provider_id }} | {{ p.claims }} | {{ p.amount|floatformat:"0g" }} |
{% endfor %}{% else %}
None.
{% endif %}
## Queued claims by ICD-11 code (top {{ top }})
{% if s.top_codes %}
| Code | Claims | Amount (KES) |
|---|---:|---:|
{% for c in s.top_codes %}| {{ c.claim__icd11_code }} | {{ c.claims }} | {{ c.amount|floatformat:"0g" }} |
{% endfor %}{% else %}
None.
{% endif %}
## Models and settings

- Classifier version: {{ s.model_versions|join:", "|default:"-" }}
{% if s.settings.queue %}- Worker queue: {{ s.settings.queue }}{% if s.settings.job %} (job #{{ s.settings.job }}){% endif %}
{% endif %}{% if s.settings.dataset %}- Stage 1 fitted on: {{ s.settings.dataset }} training split
{% endif %}{% for name, value in s.settings.thresholds.items %}- {{ name }}: {{ value }}
{% endfor %}{% if not s.settings %}- No run log: {% if batch.status == "imported" %}imported from the earlier SQLite database, which kept only flagged claims, so "received" equals "flagged at stage 1"{% else %}the batch has not been processed{% endif %}.
{% endif %}{% endautoescape %}
