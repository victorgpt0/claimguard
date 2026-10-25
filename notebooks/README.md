# notebooks/

Colab experimentation only — nothing here is the source of truth. Once a notebook cell
works, promote the logic into `src/` as a proper module and import it, rather than letting
the notebook itself become the implementation. See `implementation_roadmap.md` Section 2
for the standard Colab bootstrap cell (mount Drive, clone/pull repo, install
requirements, add `src/` to path) — paste it at the top of every notebook here.

| Notebook | Runs on | What it does |
|---|---|---|
| `01_generate_dataset.ipynb` | Colab GPU | Builds the synthetic dataset (notes by an LLM) |
| `02_train_classifier.ipynb` | Colab GPU | Fine-tunes the classifier, 5 seeds |
| `03_run_pipeline.ipynb` | Colab GPU | Runs a split through the pipeline with narratives; `WRITE_PREDICTIONS = True` also writes `pipeline`, `pipeline_text_lane`, `pipeline_stage1` and `pipeline_stage1_structural` predictions |
| `04_baselines.ipynb` | Colab GPU | Structured-only baseline (export, 5 seeds, predict, #30) and zero-shot LLM baseline (#52) |
