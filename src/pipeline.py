"""Audit pipeline orchestration (Figure 4.2 architecture, Figure 4.4 sequence): runs a batch
of claims through every stage and saves an AuditReport per flagged claim.

    claims batch -> ClaimsPreprocessor -> Isolation Forest + Autoencoder (stage 1 funnel)
      -> flagged claims only: SHAP attribution -> Semantic Consistency Classifier
      -> inconsistent only: ICD-11 reference text -> LLM narrative
      -> AuditReport saved to SQLite (pending review on the dashboard)

The detectors are fitted on the training split at start-up (seconds on a CPU). Claims the
detectors do not flag never reach the classifier: that is the funnel design, and its cost
is recorded in docs/sprint_plan.md ("Later improvements").

Each run also writes a JSON run log mapping database ids to dataset claim ids, with every
stage's output, so the stored reports can be evaluated against the ground-truth labels.
Labels are never read to make a decision.

    python -m src.pipeline --data data/processed/v0.5-5k --batch deployment_pool \\
        --cache artifacts/icd11_cache.json --classifier <adapter dir> --db audit.sqlite
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from src.classification.export import reference_text
from src.data.preprocessing import FEATURE_COLUMNS, ClaimsPreprocessor
from src.detection.autoencoder import AutoencoderDetector
from src.detection.isolation_forest import IsolationForestDetector
from src.explainability.shap_explainer import SHAPExplainer
from src.icd11.cache import ICD11Cache
from src.reporting.report_generator import AuditReportGenerator
from src.reporting.repository import AuditReportRepository

logger = logging.getLogger(__name__)

# Provisional until Sprint 6 tuning: the contamination Isolation Forest's validation
# tuning selected on every seed (TC-023).
DEFAULT_CONTAMINATION = 0.05
NARRATIVE_ATTEMPTS = 3  # NFR-02: fixed retry count, then the failure is logged


@dataclass
class Stage1:
    """The funnel's first stage: both detectors fitted on the training split."""
    preprocessor: ClaimsPreprocessor
    isolation_forest: IsolationForestDetector
    autoencoder: AutoencoderDetector
    autoencoder_threshold: float
    explainer: SHAPExplainer


def fit_stage1(train: pd.DataFrame, contamination: float = DEFAULT_CONTAMINATION,
               seed: int = 42, background_size: int = 20) -> Stage1:
    """Fits the preprocessing and both detectors on the training split. Isolation Forest's
    contamination is a quantile of its training scores; the autoencoder's threshold is
    set at the same quantile of its own training reconstruction errors, so both flag the
    same share of training claims."""
    preprocessor = ClaimsPreprocessor()
    train_x = preprocessor.feature_matrix(preprocessor.align_schema(train), fit=True)
    forest = IsolationForestDetector(contamination=contamination, random_state=seed).fit(train_x)
    torch.manual_seed(seed)
    autoencoder = AutoencoderDetector(input_dim=train_x.shape[1]).fit(train_x)
    threshold = float(np.quantile(autoencoder.score(train_x), 1 - contamination))
    # A k-means summary of the training data keeps KernelSHAP's cost per claim small.
    background = shap.kmeans(train_x, background_size)
    explainer = SHAPExplainer(forest, FEATURE_COLUMNS, background)
    return Stage1(preprocessor, forest, autoencoder, threshold, explainer)


@dataclass
class RunLog:
    started_at: str
    settings: dict
    claims: list[dict] = field(default_factory=list)
    narrative_failures: int = 0

    def summary(self) -> dict:
        flagged = [c for c in self.claims if c["flagged"]]
        inconsistent = [c for c in flagged if c.get("verdict") == "inconsistent"]
        return {
            "claims": len(self.claims),
            "flagged_stage1": len(flagged),
            "inconsistent": len(inconsistent),
            "narratives": sum(1 for c in inconsistent if c.get("narrative_ok")),
            "narrative_failures": self.narrative_failures,
        }


class AuditPipeline:
    def __init__(self, stage1: Stage1, classifier, cache: ICD11Cache, repository: AuditReportRepository,
                 narrator=None):
        """`classifier` needs .predict(claim_id, clinical_notes, icd11_definition) and
        `narrator` .generate(clinical_notes, verdict, icd11_definition) — the
        SemanticConsistencyClassifier and AuditNarrativeGenerator. With no narrator,
        inconsistent claims are reported without a narrative (e.g. no LLM access)."""
        self.stage1 = stage1
        self.classifier = classifier
        self.cache = cache
        self.repository = repository
        self.reports = AuditReportGenerator(repository)
        self.narrator = narrator

    def run(self, claims: pd.DataFrame, settings: dict | None = None) -> RunLog:
        log = RunLog(started_at=datetime.now(timezone.utc).isoformat(), settings=settings or {})
        s1 = self.stage1
        aligned = s1.preprocessor.align_schema(claims)
        features = s1.preprocessor.feature_matrix(aligned)
        forest_flag = s1.isolation_forest.model.predict(features) == -1
        forest_score = s1.isolation_forest.score(features)
        autoencoder_score = s1.autoencoder.score(features)
        flagged = forest_flag | (autoencoder_score >= s1.autoencoder_threshold)

        flagged_rows = np.flatnonzero(flagged)
        attributions = s1.explainer.explain(features[flagged_rows]) if len(flagged_rows) else []
        attribution_by_row = dict(zip(flagged_rows.tolist(), attributions))

        for i, claim in enumerate(aligned.itertuples(index=False)):
            entry = {
                "claim_ref": claim.claim_id,
                "isolation_forest_score": round(float(forest_score[i]), 4),
                "autoencoder_score": round(float(autoencoder_score[i]), 4),
                "flagged": bool(flagged[i]),
            }
            log.claims.append(entry)
            if flagged[i]:
                entry.update(self._audit(claim, attribution_by_row[i], float(forest_score[i]), log))
        return log

    def _audit(self, claim, attribution: dict, anomaly_score: float, log: RunLog) -> dict:
        """Stage 2 for one flagged claim: classifier verdict, then a narrative only if
        the verdict is inconsistent. Every flagged claim gets a report; a consistent
        verdict is a logged false positive with no narrative."""
        definition = self.cache.get(claim.icd11_code)
        reference = reference_text(definition)
        claim_id = self.repository.save_claim({
            "provider_id": claim.provider_id,
            "patient_ref": claim.member_id,
            "icd11_code": claim.icd11_code,
            "claim_amount": claim.claim_amount,
            "service_date": claim.service_date.date().isoformat(),
            "discharge_date": None if pd.isna(claim.discharge_date) else claim.discharge_date.date().isoformat(),
            "clinical_notes": claim.clinical_notes,
            "anomaly_score": anomaly_score,
        })
        verdict = self.classifier.predict(str(claim_id), claim.clinical_notes, reference)
        verdict_id = self.repository.save_verdict(claim_id, verdict.verdict, verdict.confidence,
                                                  verdict.model_version)
        narrative = None
        if verdict.verdict == "inconsistent" and self.narrator is not None:
            narrative = self._narrate(claim.clinical_notes, verdict, reference, log)
        report_id = self.reports.assemble_and_save(claim_id, verdict_id, attribution, narrative)
        return {"db_claim_id": claim_id, "report_id": report_id, "verdict": verdict.verdict,
                "confidence": round(verdict.confidence, 4), "shap": attribution,
                "narrative_ok": narrative is not None}

    def _narrate(self, notes: str, verdict, reference: str, log: RunLog) -> str | None:
        for attempt in range(1, NARRATIVE_ATTEMPTS + 1):
            try:
                return self.narrator.generate(notes, verdict, reference)
            except Exception as exc:  # provider errors vary by backend
                logger.warning("Narrative attempt %d/%d failed: %s", attempt, NARRATIVE_ATTEMPTS, exc)
                if attempt < NARRATIVE_ATTEMPTS:
                    time.sleep(2 ** attempt)
        log.narrative_failures += 1
        return None


def evaluate_run(log: RunLog, claims: pd.DataFrame) -> dict:
    """Scores a run against the batch's ground truth (offline evaluation only)."""
    truth = claims.set_index("claim_id")
    rows = pd.DataFrame(log.claims).set_index("claim_ref").join(truth[["is_fraud", "label", "fraud_typology"]])
    flagged = rows["flagged"]
    caught = flagged & (rows.get("verdict") == "inconsistent")
    return {
        "fraud_claims": int(rows["is_fraud"].sum()),
        "stage1_recall": round(float(flagged[rows["is_fraud"] == 1].mean()), 4),
        "stage1_flagged_clean": int((flagged & (rows["is_fraud"] == 0)).sum()),
        "inconsistent_caught": int((caught & (rows["label"] == "inconsistent")).sum()),
        "inconsistent_total": int((rows["label"] == "inconsistent").sum()),
        "flagged_by_typology": flagged.groupby(rows["fraud_typology"]).agg(["size", "sum"])
                                      .astype(int).to_dict(orient="index"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="processed dataset directory")
    parser.add_argument("--batch", default="deployment_pool",
                        help="split to audit (default: the DR-08 Deployment/Demonstration Pool)")
    parser.add_argument("--limit", type=int, help="audit only the first N claims of the batch")
    parser.add_argument("--cache", type=Path, required=True, help="ICD-11 cache JSON")
    parser.add_argument("--classifier", type=Path, help="LoRA adapter directory from train.py")
    parser.add_argument("--base-model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--allow-untrained", action="store_true",
                        help="run with the untrained base model (wiring checks only)")
    parser.add_argument("--db", required=True, help="SQLite database the dashboard reads")
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    parser.add_argument("--llm-provider", default="vllm", choices=["vllm", "huggingface", "openai"])
    parser.add_argument("--llm-model", help="e.g. hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4; "
                                            "omit to skip narratives")
    parser.add_argument("--log-dir", type=Path, default=Path("experiments/pipeline_runs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.classification.classifier import SemanticConsistencyClassifier
    from src.config import REPO_ROOT
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    if args.classifier is None and not args.allow_untrained:
        parser.error("--classifier is required (or --allow-untrained for a wiring check)")
    train = pd.read_parquet(args.data / "train.parquet")
    batch = pd.read_parquet(args.data / f"{args.batch}.parquet")
    if args.limit:
        batch = batch.head(args.limit)

    print(f"Fitting stage 1 on {len(train)} training claims...")
    stage1 = fit_stage1(train, args.contamination, args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = SemanticConsistencyClassifier(str(args.classifier or "untrained"), args.base_model, device)
    narrator = None
    if args.llm_model:
        from src.narrative.narrative_generator import AuditNarrativeGenerator
        narrator = AuditNarrativeGenerator(args.llm_provider, args.llm_model)

    pipeline = AuditPipeline(stage1, classifier, ICD11Cache(args.cache),
                             AuditReportRepository(args.db), narrator)
    settings = {k: str(v) for k, v in vars(args).items()} | {"classifier_version": classifier.model_version}
    log = pipeline.run(batch, settings)

    record = {"started_at": log.started_at, "settings": log.settings, "summary": log.summary(),
              "evaluation": evaluate_run(log, batch), "claims": log.claims}
    args.log_dir.mkdir(parents=True, exist_ok=True)
    out = args.log_dir / f"{args.data.name}_{args.batch}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"summary": record["summary"], "evaluation": record["evaluation"]}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
