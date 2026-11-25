# notebooks/

Colab experimentation only — nothing here is the source of truth. Once a notebook cell
works, promote the logic into `src/` as a proper module and import it, rather than letting
the notebook itself become the implementation. Each notebook starts with the same Colab
bootstrap cell (mount Drive, clone or pull the repo, install requirements, add the repo to
the path); copy it from an existing notebook when adding one.

| Notebook | Runs on | What it does |
|---|---|---|
| `01_generate_dataset.ipynb` | Colab GPU | Builds the synthetic dataset (notes by an LLM) |
| `02_train_classifier.ipynb` | Colab GPU | Fine-tunes the classifier, 5 seeds |
| `03_run_pipeline.ipynb` | Colab GPU | Runs a split through the pipeline with narratives; `WRITE_PREDICTIONS = True` also writes `pipeline`, `pipeline_text_lane`, `pipeline_stage1` and `pipeline_stage1_structural` predictions |
| `04_baselines.ipynb` | Colab GPU | Structured-only baseline (export, 5 seeds, predict, #30) and zero-shot LLM baseline (#52) |
| `05_gpu_worker.ipynb` | Colab GPU | Celery worker on the web application's `gpu` queue: processes uploaded batches with narratives (connects over TLS to the PostgreSQL and Redis on the deployed VM; see `deploy/README.md`) |
| `06_generate_live_feed.ipynb` | Colab GPU | The live claims feed (#64): a fresh seed, Oct 2025 – Mar 2026, the v0.6 providers, `L`-prefixed ids, one unsplit `live.parquet` for the simulated claims platform |
