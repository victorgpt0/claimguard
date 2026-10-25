"""Audit pipeline orchestration (Figure 4.2 architecture, Figure 4.4 sequence): runs a batch
of claims through every stage and saves an AuditReport per flagged claim.

    claims batch -> ClaimsPreprocessor -> Isolation Forest + Autoencoder + text screen
      (stage 1 funnel) -> flagged claims only: SHAP attribution -> Semantic Consistency
      Classifier -> inconsistent only: ICD-11 reference text -> LLM narrative
      -> AuditReport saved to SQLite (pending review on the dashboard)

The detectors and the screen are fitted on the training split at start-up (seconds on a
CPU). Claims stage 1 does not flag never reach the classifier: that is the funnel design.
The text-consistency screen (#50) exists so that fraud visible only in the notes still
reaches the classifier.

A flagged claim's report goes to one of two review lanes (#46): "text_inconsistent" when
the classifier finds the notes inconsistent with the billed code (with a narrative), and
"structural_anomaly" when the notes are consistent but a structured-data detector flagged
the claim (conversion and dependent fraud have consistent notes). A claim flagged only by
the text screen whose notes the classifier finds consistent is a false positive of the
screen: its verdict is saved, but no report is queued (FR-05).

Each run also writes a JSON run log mapping database ids to dataset claim ids, with every
stage's output, so the stored reports can be evaluated against the ground-truth labels.
Labels are never read to make a decision.

    python -m src.pipeline --data data/processed/v0.5-5k --batch deployment_pool \\
        --cache artifacts/icd11_cache.json --classifier <adapter dir> --db audit.sqlite \\
        --config config/config.yaml

Stage-1 settings come from config.yaml's `thresholds:` block (src.detection.tune_stage1
writes the values); the command-line options override it for experiments.
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
from src.detection.text_screen import TextConsistencyScreen
from src.explainability.shap_explainer import SHAPExplainer
from src.icd11.cache import ICD11Cache
from src.reporting.report_generator import AuditReportGenerator
from src.reporting.repository import AuditReportRepository

logger = logging.getLogger(__name__)

# Used only when neither config.yaml nor the command line sets a value: the contamination
# Isolation Forest's F1 tuning selected on every seed (TC-023).
DEFAULT_CONTAMINATION = 0.05
STRUCTURED_SOURCES = {"isolation_forest", "autoencoder"}
LANES = ("text_inconsistent", "structural_anomaly")
NARRATIVE_ATTEMPTS = 3  # NFR-02: fixed retry count, then the failure is logged


@dataclass
class Stage1:
    """The funnel's first stage: both detectors fitted on the training split."""
    preprocessor: ClaimsPreprocessor
    isolation_forest: IsolationForestDetector
    autoencoder: AutoencoderDetector
    autoencoder_threshold: float
    explainer: SHAPExplainer
    screen: TextConsistencyScreen | None = None


def fit_stage1(train: pd.DataFrame, contamination: float = DEFAULT_CONTAMINATION,
               seed: int = 42, background_size: int = 20, autoencoder_quantile: float | None = None,
               screen: TextConsistencyScreen | None = None) -> Stage1:
    """Fits the preprocessing and both detectors on the training split. Isolation Forest's
    contamination is a quantile of its training scores; the autoencoder's threshold is
    the `autoencoder_quantile` of its own training reconstruction errors (by default
    1 - contamination, so both flag the same share of training claims). `screen` is an
    already fitted TextConsistencyScreen with a threshold, or None for no screen."""
    if autoencoder_quantile is None:
        autoencoder_quantile = 1 - contamination
    preprocessor = ClaimsPreprocessor()
    train_x = preprocessor.feature_matrix(preprocessor.align_schema(train), fit=True)
    forest = IsolationForestDetector(contamination=contamination, random_state=seed).fit(train_x)
    torch.manual_seed(seed)
    autoencoder = AutoencoderDetector(input_dim=train_x.shape[1]).fit(train_x)
    threshold = float(np.quantile(autoencoder.score(train_x), autoencoder_quantile))
    # A k-means summary of the training data keeps KernelSHAP's cost per claim small.
    background = shap.kmeans(train_x, background_size)
    explainer = SHAPExplainer(forest, FEATURE_COLUMNS, background)
    return Stage1(preprocessor, forest, autoencoder, threshold, explainer, screen)


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
            "flagged_by": {source: sum(source in c["flag_sources"] for c in flagged)
                           for source in ("isolation_forest", "autoencoder", "text_screen")},
            "reported": {lane: sum(c.get("lane") == lane for c in flagged) for lane in LANES},
            "screen_false_positives": sum(1 for c in flagged if not c.get("reported")),
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
        autoencoder_flag = autoencoder_score >= s1.autoencoder_threshold
        screen_flag = s1.screen.flag(aligned) if s1.screen is not None else np.zeros(len(aligned), bool)
        flagged = forest_flag | autoencoder_flag | screen_flag

        flagged_rows = np.flatnonzero(flagged)
        attributions = s1.explainer.explain(features[flagged_rows]) if len(flagged_rows) else []
        attribution_by_row = dict(zip(flagged_rows.tolist(), attributions))

        for i, claim in enumerate(aligned.itertuples(index=False)):
            entry = {
                "claim_ref": claim.claim_id,
                "isolation_forest_score": round(float(forest_score[i]), 4),
                "autoencoder_score": round(float(autoencoder_score[i]), 4),
                "flagged": bool(flagged[i]),
                "flag_sources": [name for name, hit in (("isolation_forest", forest_flag[i]),
                                                        ("autoencoder", autoencoder_flag[i]),
                                                        ("text_screen", screen_flag[i])) if hit],
            }
            log.claims.append(entry)
            if flagged[i]:
                entry.update(self._audit(claim, attribution_by_row[i], float(forest_score[i]),
                                         entry["flag_sources"], log))
        return log

    def _audit(self, claim, attribution: dict, anomaly_score: float, sources: list[str],
               log: RunLog) -> dict:
        """Stage 2 for one flagged claim: classifier verdict, then a narrative only if
        the verdict is inconsistent, and a report in the lane the verdict and the flag
        sources imply (module docstring). A screen-only flag with consistent notes is a
        logged false positive: verdict saved, no report."""
        definition = self.cache.get(claim.icd11_code)
        reference = reference_text(definition)
        claim_id = self.repository.save_claim({
            "claim_ref": str(claim.claim_id),
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
        result = {"db_claim_id": claim_id, "verdict": verdict.verdict,
                  "confidence": round(verdict.confidence, 4), "shap": attribution}
        if verdict.verdict == "inconsistent":
            lane = "text_inconsistent"
        elif STRUCTURED_SOURCES.intersection(sources):
            lane = "structural_anomaly"
        else:
            return result | {"reported": False, "lane": None, "report_id": None, "narrative_ok": False}
        narrative = None
        if lane == "text_inconsistent" and self.narrator is not None:
            narrative = self._narrate(claim.clinical_notes, verdict, reference, log)
        report_id = self.reports.assemble_and_save(claim_id, verdict_id, attribution, narrative, sources)
        return result | {"reported": True, "lane": lane, "report_id": report_id,
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
    reported = rows["reported"].fillna(False).astype(bool) if "reported" in rows else flagged & False
    fraud = rows["is_fraud"] == 1
    return {
        "fraud_claims": int(fraud.sum()),
        "stage1_recall": round(float(flagged[fraud].mean()), 4),
        "stage1_flagged_clean": int((flagged & ~fraud).sum()),
        "end_to_end_recall": round(float(reported[fraud].mean()), 4),
        "reported_clean": int((reported & ~fraud).sum()),
        "inconsistent_caught": int((caught & (rows["label"] == "inconsistent")).sum()),
        "inconsistent_total": int((rows["label"] == "inconsistent").sum()),
        "flagged_by_typology": flagged.groupby(rows["fraud_typology"]).agg(["size", "sum"])
                                      .astype(int).to_dict(orient="index"),
        "reported_by_typology": reported.groupby(rows["fraud_typology"]).agg(["size", "sum"])
                                        .astype(int).to_dict(orient="index"),
    }


def run_predictions(log: RunLog) -> dict[str, pd.DataFrame]:
    """The run in the shared prediction format (src.evaluation.predictions): "pipeline"
    flags every claim that reached an auditor lane; "pipeline_text_lane" only the
    text-inconsistent lane, scored by the classifier's probability of "inconsistent"
    (0 for claims the classifier never saw). "pipeline_stage1" (any stage-1 flag) and
    "pipeline_stage1_structural" (an Isolation Forest or autoencoder flag) let another
    stage-2 model be put in the classifier's place (src.baselines.zero_shot_llm)."""
    rows = pd.DataFrame(log.claims)
    reported = rows["reported"].fillna(False).astype(bool) if "reported" in rows else rows["flagged"] & False
    verdict = rows.get("verdict", pd.Series(index=rows.index, dtype=object))
    confidence = rows.get("confidence", pd.Series(index=rows.index, dtype=float))
    p_inconsistent = np.where(verdict == "inconsistent", confidence,
                              np.where(verdict == "consistent", 1 - confidence, 0.0))
    claim_id = rows["claim_ref"].astype(str)
    structural = rows["flag_sources"].map(lambda sources: bool({"isolation_forest", "autoencoder"} & set(sources)))
    return {
        "pipeline_stage1": pd.DataFrame({"claim_id": claim_id, "flagged": rows["flagged"].astype(int)}),
        "pipeline_stage1_structural": pd.DataFrame({"claim_id": claim_id, "flagged": structural.astype(int)}),
        "pipeline": pd.DataFrame({"claim_id": claim_id, "flagged": reported.astype(int)}),
        "pipeline_text_lane": pd.DataFrame({"claim_id": claim_id,
                                            "flagged": (verdict == "inconsistent").astype(int),
                                            "score": p_inconsistent}),
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
    parser.add_argument("--config", type=Path, help="config.yaml whose thresholds: block sets stage 1")
    parser.add_argument("--contamination", type=float, help="overrides the config value")
    parser.add_argument("--autoencoder-quantile", type=float, help="overrides the config value")
    parser.add_argument("--screen-threshold", type=float, help="overrides the config value")
    parser.add_argument("--no-text-screen", action="store_true",
                        help="stage 1 without the text screen (the original funnel)")
    parser.add_argument("--predictions", type=Path, nargs="?", const=Path("experiments/predictions"),
                        help="also write the run in the shared prediction format under this root")
    parser.add_argument("--llm-provider", default="vllm", choices=["vllm", "huggingface", "openai"])
    parser.add_argument("--llm-model", help="e.g. hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4; "
                                            "omit to skip narratives")
    parser.add_argument("--log-dir", type=Path, default=Path("experiments/pipeline_runs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # INFO for this module only: shap logs every KernelSHAP iteration at INFO.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logger.setLevel(logging.INFO)

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

    thresholds = {}
    if args.config:
        from src.config import load_config
        thresholds = load_config(args.config).get("thresholds") or {}
    contamination = first_set(args.contamination, thresholds.get("isolation_forest_contamination"),
                              DEFAULT_CONTAMINATION)
    quantile = first_set(args.autoencoder_quantile, thresholds.get("autoencoder_train_quantile"))
    screen_threshold = first_set(args.screen_threshold, thresholds.get("text_screen_threshold"))
    cache = ICD11Cache(args.cache)
    screen = None
    if not args.no_text_screen:
        if screen_threshold is None:
            parser.error("no text_screen_threshold in --config or --screen-threshold "
                         "(run src.detection.tune_stage1), or pass --no-text-screen")
        screen = TextConsistencyScreen(cache, screen_threshold).fit(train)

    print(f"Fitting stage 1 on {len(train)} training claims...")
    stage1 = fit_stage1(train, contamination, args.seed, autoencoder_quantile=quantile, screen=screen)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = SemanticConsistencyClassifier(str(args.classifier or "untrained"), args.base_model, device)
    narrator = None
    if args.llm_model:
        from src.narrative.narrative_generator import AuditNarrativeGenerator
        narrator = AuditNarrativeGenerator(args.llm_provider, args.llm_model)

    pipeline = AuditPipeline(stage1, classifier, cache, AuditReportRepository(args.db), narrator)
    settings = {k: str(v) for k, v in vars(args).items()} | {
        "classifier_version": classifier.model_version, "contamination": contamination,
        "autoencoder_threshold": stage1.autoencoder_threshold, "screen_threshold": screen_threshold}
    log = pipeline.run(batch, settings)

    record = {"started_at": log.started_at, "settings": log.settings, "summary": log.summary(),
              "evaluation": evaluate_run(log, batch), "claims": log.claims}
    args.log_dir.mkdir(parents=True, exist_ok=True)
    out = args.log_dir / f"{args.data.name}_{args.batch}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"summary": record["summary"], "evaluation": record["evaluation"]}, indent=2))
    print(f"-> {out}")
    if args.predictions:
        from src.evaluation.predictions import write_predictions
        for system, frame in run_predictions(log).items():
            print(f"-> {write_predictions(system, args.batch, frame, args.predictions)}")


def first_set(*values):
    return next((v for v in values if v is not None), None)


if __name__ == "__main__":
    main()
