import argparse
from pathlib import Path

import pandas as pd
import torch

from experiments.utils import paths
from experiments.utils.factory import make_oracle
from experiments.utils.metrics import MODE_THRESHOLDS, compute_trajectory_metrics


SIM_THRESHOLDS = [0.0, 0.4, 0.5, 0.6]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--oracle", choices=["seh", "jnk3", "gsk"], required=True)
    parser.add_argument("--variant", type=str, default=None,
                        help="Variant subdirectory under oracle (e.g. 'noise_0.5', 'concat_4k').")
    parser.add_argument("--hit", type=int, default=None,
                        help="If set, evaluate only this hit index.")
    parser.add_argument("--budget", type=int, default=None,
                        help="If set, truncate trajectories to this many oracle calls.")
    parser.add_argument("--noisy", action="store_true",
                        help="Recompute clean oracle scores (use when reading noisy_proxy runs).")
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def load_initial_hit(oracle: str, hit_idx: int) -> str:
    df = pd.read_csv(paths.hits_path(oracle))
    return df["molecule"].iloc[hit_idx]


def evaluate_hit(experiment: str, model: str, oracle: str, hit_idx: int,
                 budget: int | None, noisy: bool, variant: str | None, batch_size: int | None) -> Path:
    df = pd.read_csv(paths.results_path(experiment, model, oracle, hit_idx, variant=variant))
    if budget is not None:
        df = df.iloc[:budget]
    smiles = df["smiles"].to_numpy()

    if noisy:
        scores = make_oracle(oracle)(smiles.tolist()).squeeze(-1)
    else:
        scores = torch.from_numpy(df["oracle_score"].to_numpy())

    initial_hit = load_initial_hit(oracle, hit_idx)
    metrics_df = compute_trajectory_metrics(
        smiles_list=smiles, scores_list=scores, initial_hit=initial_hit,
        mode_thresholds=MODE_THRESHOLDS[oracle], sim_thresholds=SIM_THRESHOLDS,
        batch_size=batch_size,    # NEW
    )
    out_path = paths.metrics_path(experiment, model, oracle, hit_idx, variant=variant)
    metrics_df.to_csv(out_path, index=False)
    return out_path


def aggregate(metrics_paths: list[Path]) -> pd.DataFrame:
    """Mean + std across hits, grouped by oracle_calls."""
    dfs = [pd.read_csv(p) for p in metrics_paths]
    concat = pd.concat(dfs, ignore_index=True)
    agg = concat.groupby("oracle_calls").agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def main():
    args = parse_args()

    results_dir = paths.RESULTS_DIR / args.experiment / args.model / args.oracle
    if args.variant:
        results_dir = results_dir / args.variant

    hit_files = sorted(
        f for f in results_dir.glob("hit_*.csv") if "_metrics" not in f.stem
    )
    if args.hit is not None:
        hit_files = [f for f in hit_files if int(f.stem.split("_")[1]) == args.hit]

    metrics_paths = []
    for f in hit_files:
        hit_idx = int(f.stem.split("_")[1])
        print(f"Evaluating {f.name}...")
        metrics_paths.append(
            evaluate_hit(args.experiment, args.model, args.oracle, hit_idx,
                         args.budget, args.noisy, args.variant, args.batch_size)
        )

    if len(metrics_paths) > 1:
        agg = aggregate(metrics_paths)
        agg_path = paths.aggregated_path(args.experiment, args.model, args.oracle, variant=args.variant)
        agg.to_csv(agg_path, index=False)
        print(f"Aggregated → {agg_path}")


if __name__ == "__main__":
    main()