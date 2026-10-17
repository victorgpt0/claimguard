-- Logical Database Schema (chapter4_system_analysis_design.txt Section 4.4.4, Figure 4.7).
-- Training data, model artifacts, and the Feedback Dataset are deliberately EXCLUDED from
-- this schema (DR-05, DR-08) — they are file-based artifacts in Google Drive, not part of
-- this operational database.

CREATE TABLE IF NOT EXISTS User (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('Auditor', 'Administrator'))
);

CREATE TABLE IF NOT EXISTS Claim (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_ref       TEXT,   -- the submitter's own claim id (the dataset's claim_id)
    provider_id     TEXT NOT NULL,
    patient_ref     TEXT NOT NULL,
    icd11_code      TEXT NOT NULL,
    claim_amount    REAL NOT NULL,
    service_date    DATE NOT NULL,
    discharge_date  DATE,
    clinical_notes  TEXT NOT NULL,
    anomaly_score   REAL
);

CREATE TABLE IF NOT EXISTS ClassifierVerdict (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id      INTEGER NOT NULL REFERENCES Claim(id),
    verdict       TEXT NOT NULL CHECK (verdict IN ('consistent', 'inconsistent')),
    confidence    REAL NOT NULL,
    model_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS AuditReport (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id         INTEGER NOT NULL REFERENCES Claim(id),
    verdict_id       INTEGER REFERENCES ClassifierVerdict(id),
    shap_attribution TEXT,   -- JSON-encoded SHAP attribution dictionary
    llm_narrative    TEXT,   -- NULL when verdict is 'consistent' (structural-anomaly lane)
    flag_sources     TEXT,   -- JSON list of the stage-1 components that flagged the claim
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'approved', 'rejected', 'escalated')),
    reviewed_by      INTEGER REFERENCES User(id),
    reviewed_at      DATETIME,
    created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_auditreport_status ON AuditReport(status);
CREATE INDEX IF NOT EXISTS idx_claim_provider ON Claim(provider_id);
