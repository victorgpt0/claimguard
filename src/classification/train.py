"""LoRA/PEFT fine-tuning script for the Semantic Consistency Classifier (Section 3.2.3,
Section 3.8.6). Fine-tunes across 5 random seeds (config.yaml training.random_seeds) per
the rigor-improvements patch (Block K): a single training run cannot distinguish a
genuinely better classifier from a lucky seed (Dodge et al., 2020).

For each seed, writes experiments/<run_id>/config.json (hyperparameters, seed, base
model, LoRA config, final validation/test metrics) per the implementation roadmap's
Section 2a — this is what Chapter 5 (research_writing.txt 5.3.2) means by "retain
configuration files and experiment logs so the reported run can be reproduced."
"""


def train_one_seed(seed: int, config: dict) -> dict:
    """Fine-tune the classifier for a single seed. Returns the run's metrics dict, which
    the caller writes to experiments/<run_id>/config.json alongside the hyperparameters."""
    raise NotImplementedError


def train_all_seeds(config: dict) -> list[dict]:
    """Runs train_one_seed for every seed in config['training']['random_seeds'];
    returns per-seed metrics so mean +/- std can be computed and reported (Section 3.2.4)."""
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError("Wire this up once config loading and the dataset are ready.")
