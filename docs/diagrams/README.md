# Design diagrams for the Django platform (#60)

PlantUML sources and rendered PNGs for the Chapter 4 figures that the Django platform
changed or added (26 Oct 2026). The paste instructions and the text around each figure are
in `compare/changes.txt`, section D, outside this repository.

| File | Figure in the document | Change |
|---|---|---|
| `fig4_1_use_case` | 4.1 Use Case Diagram | Replaces: sign-in, upload, batch processing, batch summary, password change, user management, activity log; Google and the processing worker as secondary actors |
| `fig4_3_architecture` | 4.3 System Architecture Diagram | Replaces: Django, PostgreSQL (Neon), Redis (Upstash), CPU and Colab GPU workers instead of Streamlit and SQLite |
| `fig4_4_class` | 4.4 Class Diagram | Replaces: adds User, ActivityLog, Batch, Job, UploadValidator, BatchSummary, the process_batch task and DjangoAuditRepository |
| `fig4_7_sequence_upload` | new 4.7 Sequence Diagram: Batch Upload and Processing | New, after Figure 4.6 |
| `fig4_8_schema` | 4.8 Logical Database Schema (was 4.7) | Replaces: PostgreSQL tables from the Django migrations |
| `fig4_9_state_batch_job` | new 4.9 State Machine Diagram: Batch and Job | New, in Section 4.4.5 |

Render (PlantUML 1.2020.2 and Graphviz are enough):

    java -jar plantuml.jar -tpng docs/diagrams/*.puml

A jar can be obtained with `apt-get download plantuml && dpkg-deb -x plantuml*.deb /tmp/plantuml_extract`
(no root needed); it lands at `/tmp/plantuml_extract/usr/share/plantuml/plantuml.jar`.
