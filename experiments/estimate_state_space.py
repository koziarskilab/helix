"""
Estimates the chemical space size accessible by HELiX as a function of
composition depth, across multiple fragment libraries.

Run from the repo root:
    python experiments/estimate_state_space.py --libraries small concat_2k

Outputs one JSON per library at:
    experiments/results/space_estimation/{library_name}.json

Note: `scent.json` in the results directory is a frozen artifact from an older
version of this script that ran a SCENT-style global space estimate (forward
sampling + backward trajectory enumeration). The schema differs (single record
per library with `traj_n` and `traj_counts`, no `hit_idx`). The HELiX local
estimate computed here uses forward sampling only.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionActionExit,
    ReactionActionSpaceEarlyTerminate,
    ReactionActionX,
    ReactionStateA,
    ReactionStateB,
    ReactionStateEarlyTerminal,
    ReactionStateH,
    ReactionStateTerminal,
    ReactionStateX,
    ReactionStateY,
    ReactionStateZ,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory
from rgfn.gfns.reaction_gfn.reaction_env import ReactionEnv

from experiments.utils import paths

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = paths.RESULTS_DIR / "space_estimation"

LIBRARY_NAMES = ["small", "concat_2k", "concat_4k", "concat_8k", "concat_16k"]

N_HITS                 = 3      # number of sEH hits to average over
BUDGET                 = 1
MAX_NUM_DECOMPOSITIONS = 2
MAX_NUM_REACTIONS      = 4
N_SAMPLES              = 1000   # forward trajectories per hit
SEED                   = 42


def get_library_cfg(name: str) -> dict:
    """Map a library name to its data files and yield strategy."""
    if name == "small":
        lib_dir = paths.DATA_DIR / "small"
        return {
            "reaction_path": str(lib_dir / "templates.txt"),
            "fragment_path": str(lib_dir / "fragments.txt"),
            "cost_path":     str(lib_dir / "fragment_to_real_cost.json"),
            "yield_path":    str(lib_dir / "yields.csv"),
            "yield_value":   None,
        }
    if name.startswith("concat_"):
        size = name.split("_", 1)[1]   # e.g. "2k"
        lib_dir = paths.DATA_DIR / "concat"
        return {
            "reaction_path": str(lib_dir / "templates.txt"),
            "fragment_path": str(lib_dir / f"fragments_{size}.txt"),
            "cost_path":     str(lib_dir / "fragment_to_real_cost.json"),
            "yield_path":    None,
            "yield_value":   0.75,
        }
    raise ValueError(f"Unknown library: {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hit(hit_idx: int, budget: int = BUDGET) -> Tuple[str, int]:
    df = pd.read_csv(paths.hits_path("seh"))
    smiles = df["molecule"].iloc[hit_idx]
    return (smiles, budget)


def make_env(library_cfg: dict, hit: Tuple[str, int]) -> ReactionEnv:
    factory = ReactionDataFactory(
        reaction_path=library_cfg["reaction_path"],
        fragment_path=library_cfg["fragment_path"],
        cost_path=library_cfg["cost_path"],
        yield_path=library_cfg["yield_path"],
        yield_value=library_cfg["yield_value"],
    )
    env = ReactionEnv(
        data_factory=factory,
        max_num_reactions=MAX_NUM_REACTIONS,
        max_num_decompositions=MAX_NUM_DECOMPOSITIONS,
        starting_smiles_with_budget=[hit],
        use_recurrence=False,
        use_cache=True,
    )
    return env


# ---------------------------------------------------------------------------
# Forward sampling
# ---------------------------------------------------------------------------

def sample_trajectories(env: ReactionEnv, n_samples: int = N_SAMPLES):
    """Sample random forward trajectories; force max_num_decompositions decomp steps."""
    branching_factors = defaultdict(lambda: {"react": [], "frag": []})
    terminal_states = []
    stats = {"n_early_exits": 0, "n_dead_ends": 0, "n_invalid_intermediates": 0, "n_success": 0}

    for _ in range(n_samples):
        # step 1: StateH -> StateX
        state = ReactionStateH()
        action_space = env.get_forward_action_spaces([state])[0]
        possible = action_space.get_possible_actions_indices()
        if not possible:
            stats["n_early_exits"] += 1
            continue
        idx = random.choice(possible)
        state = env.apply_forward_actions([state], [action_space.get_action_at_idx(idx)])[0]

        # step 2: decomposition
        decomp_steps = 0
        early_exit = False
        while isinstance(state, (ReactionStateX, ReactionStateY, ReactionStateZ)):
            action_space = env.get_forward_action_spaces([state])[0]
            possible = action_space.get_possible_actions_indices()
            if isinstance(state, ReactionStateX):
                if decomp_steps < env.max_num_decompositions:
                    possible = [i for i in possible
                                if not isinstance(action_space.get_action_at_idx(i), ReactionActionExit)]
                else:
                    possible = [i for i in possible
                                if isinstance(action_space.get_action_at_idx(i), ReactionActionExit)]
            elif isinstance(state, (ReactionStateY, ReactionStateZ)):
                possible = [i for i in possible
                            if not isinstance(action_space.get_action_at_idx(i), ReactionActionExit)]
            if not possible:
                early_exit = True
                break
            idx = random.choice(possible)
            action = action_space.get_action_at_idx(idx)
            state = env.apply_forward_actions([state], [action])[0]
            if isinstance(action, ReactionActionX):
                decomp_steps += 1

        if early_exit:
            stats["n_early_exits"] += 1
            continue

        # step 3: composition
        composition_depth = 0
        while not isinstance(state, (ReactionStateTerminal, ReactionStateEarlyTerminal)):
            action_space = env.get_forward_action_spaces([state])[0]
            possible = action_space.get_possible_actions_indices()
            if isinstance(action_space, ReactionActionSpaceEarlyTerminate):
                stats["n_dead_ends"] += 1
                break
            if not possible:
                stats["n_invalid_intermediates"] += 1
                break
            if isinstance(state, ReactionStateA):
                all_actions = [action_space.get_action_at_idx(i) for i in possible]
                reaction_actions = [a for a in all_actions if a.anchored_reaction is not None]
                if state.budget > 0 and len(reaction_actions) > 0:
                    branching_factors[composition_depth]["react"].append(len(reaction_actions))
                    action = random.choice(reaction_actions)
                else:
                    if state.budget > 0 and len(reaction_actions) == 0:
                        stats["n_dead_ends"] += 1
                    action = random.choice(all_actions)
            elif isinstance(state, ReactionStateB):
                branching_factors[composition_depth]["frag"].append(len(possible))
                action = action_space.get_action_at_idx(random.choice(possible))
            else:  # StateC
                action = action_space.get_action_at_idx(random.choice(possible))
            state = env.apply_forward_actions([state], [action])[0]
            if isinstance(state, ReactionStateA):
                composition_depth += 1

        if isinstance(state, ReactionStateTerminal):
            terminal_states.append(state)
            stats["n_success"] += 1

    return branching_factors, terminal_states, stats


# ---------------------------------------------------------------------------
# Space size estimator
# ---------------------------------------------------------------------------

def estimate_space_sizes(branching_factors: Dict, max_depth: int) -> Dict[int, float]:
    """Apply RGFN paper eq. 9 to estimate space size at each composition depth."""
    sizes = {}
    product = 1.0
    for depth in range(max_depth):
        react = branching_factors[depth]["react"]
        frag  = branching_factors[depth]["frag"]
        mean_react = np.mean(react) if react else 0.0
        # unimolecular reactions have no StateB - treat as no branching
        mean_frag  = np.mean(frag) if frag else 1.0
        product *= mean_react * mean_frag
        sizes[depth + 1] = product
    return sizes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_estimation(library_name: str, library_cfg: dict, hit_idx: int, results_path: Path) -> dict:
    print(f"\n  hit {hit_idx} ...")
    hit = load_hit(hit_idx)
    env = make_env(library_cfg, hit)
    print(f"    fragments={len(env.fragments)}, anchored_reactions={len(env.anchored_reactions)}")

    branching_factors, terminal_states, stats = sample_trajectories(env, n_samples=N_SAMPLES)
    print(f"    stats: {stats}")
    print(f"    terminal states: {len(terminal_states)}")

    max_depth = MAX_NUM_DECOMPOSITIONS + BUDGET
    sizes = estimate_space_sizes(branching_factors, max_depth)
    print(f"    space sizes: {sizes}")

    result = {
        "library":     library_name,
        "hit_idx":     hit_idx,
        "n_fragments": len(env.fragments),
        "stats":       stats,
        "space_sizes": {str(k): v for k, v in sizes.items()},
        "mean_react":  {str(d): float(np.mean(branching_factors[d]["react"]))
                        if branching_factors[d]["react"] else 0.0
                        for d in range(max_depth)},
        "mean_frag":   {str(d): float(np.mean(branching_factors[d]["frag"]))
                        if branching_factors[d]["frag"] else 1.0
                        for d in range(max_depth)},
        "raw_react":   {str(d): branching_factors[d]["react"]
                        for d in range(max_depth)},
        "raw_frag":    {str(d): branching_factors[d]["frag"]
                        for d in range(max_depth)},
    }

    existing = json.loads(results_path.read_text()) if results_path.exists() else []
    existing.append(result)
    results_path.write_text(json.dumps(existing, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libraries", nargs="+", default=LIBRARY_NAMES)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    N_SAMPLES = args.n_samples

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for lib_name in args.libraries:
        lib_cfg = get_library_cfg(lib_name)
        print(f"\n{'='*60}")
        print(f"Library: {lib_name}")
        print(f"{'='*60}")
        results_path = OUT_DIR / f"{lib_name}.json"
        for hit_idx in range(N_HITS):
            run_estimation(lib_name, lib_cfg, hit_idx, results_path)

    print("\nDone.")