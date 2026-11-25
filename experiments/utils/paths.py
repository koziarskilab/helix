from pathlib import Path

RUNS_DIR        = Path("runs")
CHECKPOINTS_DIR = Path("bo_checkpoints")
RESULTS_DIR     = Path("experiments") / "results"
DATA_DIR        = Path("data")
HITS_DIR        = Path("experiments") / "data" / "initial_hits"


def _scope(experiment: str, family: str, oracle: str, variant: str | None = None) -> Path:
    """Common subpath: experiment/family/oracle[/variant]."""
    base = Path(experiment) / family / oracle
    return base / variant if variant else base


def run_name(experiment: str, family: str, oracle: str, hit: int, variant: str | None = None) -> str:
    """Run identifier used for WandB run name and run directory."""
    return f"{_scope(experiment, family, oracle, variant)}/hit_{hit}"


def runs_dir(experiment: str, family: str, oracle: str, hit: int, variant: str | None = None) -> Path:
    """Trainer output directory (synthesis paths, WandB artifacts)."""
    return RUNS_DIR / run_name(experiment, family, oracle, hit, variant=variant)


def checkpoint_path(experiment: str, family: str, oracle: str, hit: int, variant: str | None = None) -> Path:
    """BO History checkpoint path."""
    return CHECKPOINTS_DIR / _scope(experiment, family, oracle, variant) / f"hit_{hit}.pt"


def hits_path(oracle: str) -> Path:
    """Path to the initial hits file for an oracle."""
    return HITS_DIR / f"{oracle}_hits.csv"


def results_path(experiment: str, family: str, oracle: str, hit: int, variant: str | None = None) -> Path:
    return RESULTS_DIR / _scope(experiment, family, oracle, variant) / f"hit_{hit}.csv"


def metrics_path(experiment: str, family: str, oracle: str, hit: int, variant: str | None = None) -> Path:
    return RESULTS_DIR / _scope(experiment, family, oracle, variant) / f"hit_{hit}_metrics.csv"


def aggregated_path(experiment: str, family: str, oracle: str, variant: str | None = None) -> Path:
    return RESULTS_DIR / _scope(experiment, family, oracle, variant) / "aggregated.csv"