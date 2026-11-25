"""
No-BO HELiX runner for paper experiments.

Trains an RGFN policy under a gin config and writes the oracle-call history
to results/{experiment}/{model}/{oracle}/[variant/]hit_{hit}.csv.

Run from repo root, e.g.:
    python experiments/train_helix.py \
        --cfg configs/experiments/bo_ablations/helix.gin \
        --oracle seh --hit 0
"""
import argparse
from pathlib import Path

import gin
import pandas as pd

from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateEarlyTerminal
from rgfn.trainer.trainer import Trainer
from rgfn.utils.helpers import seed_everything

from experiments.utils import paths


MODEL_NAMES = {"helix", "helix_bo"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--oracle", choices=["seh", "jnk3", "gsk"], default="seh")
    parser.add_argument("--hit", type=int, default=0)
    parser.add_argument("--max-oracle-calls", type=int, default=10000)
    return parser.parse_args()


def derive_cfg_meta(cfg_path: str):
    """Map a cfg path to (experiment, model, variant). Mirrors run_helix.py."""
    cfg = Path(cfg_path)
    if cfg.parent.name == "experiments":
        return cfg.stem, "helix", None
    experiment = cfg.parent.name
    if cfg.stem in MODEL_NAMES:
        return experiment, cfg.stem, None
    return experiment, "helix", cfg.stem


def extract_results(trainer, max_calls: int):
    """Pull (smiles, score) pairs from the oracle proxy's cache, capped."""
    oracle_proxy = trainer.train_forward_sampler.reward.proxy
    smiles, scores = [], []
    for state, score_dict in oracle_proxy.cache.items():
        if isinstance(state, ReactionStateEarlyTerminal):
            continue
        score = score_dict if isinstance(score_dict, float) else score_dict.get("value", score_dict)
        smiles.append(state.molecule.smiles)
        scores.append(score)
        if len(smiles) >= max_calls:
            break
    return smiles, scores


def main():
    args = parse_args()
    seed_everything(args.hit)

    experiment, model, variant = derive_cfg_meta(args.cfg)
    name = paths.run_name(experiment, model, args.oracle, args.hit, variant=variant)

    gin.parse_config_files_and_bindings(
        [args.cfg],
        bindings=[
            f'hits_filepath = "{paths.hits_path(args.oracle)}"',
            f'load_starting_molecules.hit = {args.hit}',
            f'run_name = "{name}"',
        ],
    )

    paths.runs_dir(experiment, model, args.oracle, args.hit, variant=variant).mkdir(
        parents=True, exist_ok=True
    )

    trainer = Trainer()
    trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
    trainer.logger.log_to_file(gin.config_str(), "config")

    trainer.train()

    smiles, scores = extract_results(trainer, args.max_oracle_calls)

    results_path = paths.results_path(experiment, model, args.oracle, args.hit, variant=variant)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": smiles, "oracle_score": scores}).to_csv(results_path, index=False)

    trainer.logger.close()


if __name__ == "__main__":
    main()
