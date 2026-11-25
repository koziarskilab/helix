from typing import Callable, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import DataStructs

from molbo.bo import History


MODE_THRESHOLDS = {
    "seh": [7.2, 8.0],
    "jnk3": [0.75, 0.8],
    "gsk": [0.75, 0.8],
}


# ============================================================================
# Helper Functions
# ============================================================================


# Compute ECFP fingerprints:
def smiles_to_morgan_fp(smiles: str, radius: int = 2, fp_size: int = 2048, as_numpy: bool = True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    # Return fingerprint as numpy array
    if as_numpy:
        return fp_gen.GetCountFingerprintAsNumPy(mol)
    return fp_gen.GetCountFingerprint(mol)


def get_centroid_indices(
    smiles_list: List[str], scores_list: torch.Tensor, tanimoto_threshold: float = 0.7
):

    fps = np.array([smiles_to_morgan_fp(smi, as_numpy=False) for smi in smiles_list])

    sorted_indices = torch.argsort(scores_list, descending=True)
    centroids = []

    for fp, score, idx in zip(fps[sorted_indices], scores_list[sorted_indices], sorted_indices):
        if len(centroids) == 0:
            centroids.append((fp, score, idx))
        else:
            is_close = False
            for centroid in centroids:
                if DataStructs.TanimotoSimilarity(fp, centroid[0]) > tanimoto_threshold:
                    is_close = True
            if not is_close:
                centroids.append((fp, score, idx))

    centroid_indices = [centroid[2].item() for centroid in centroids]
    return centroid_indices


def avg_tanimoto_similarity(fps: List[np.ndarray]):
    """Compute average pairwise Tanimoto similarity."""
    total_similarity = 0
    n = len(fps)
    for fp in fps:
        for other_fp in fps:
            sim = DataStructs.TanimotoSimilarity(fp, other_fp)
            total_similarity += sim
    return total_similarity / (n**2)


# ============================================================================
# Cheap online metrics
# ============================================================================

def best_observed(history: History) -> dict:
    return {"best_observed": history.best_observed}


def n_observed(history: History) -> dict:
    return {"n_observed": len(history.y_all)}


def top_k_mean(ks: List[int]) -> Callable[[History], dict]:
    def metric(history: History) -> dict:
        y = history.y_all.squeeze(-1)
        return {
            f"top_{k}_mean": y.topk(min(k, len(y))).values.mean()
            for k in ks
        }
    return metric


# ============================================================================
# Post-hoc metrics (computed offline from checkpoint)
# ============================================================================

def best_score(scores_list: torch.Tensor) -> float:
    return scores_list.max().item()


def topk_mean(scores_list: torch.Tensor, k: int = 10) -> float:
    if len(scores_list) < k:
        return scores_list.mean().item()
    return scores_list.topk(k).values.mean().item()

def diversity(smiles_list: np.ndarray):
    """Compute diversity as 1 - average Tanimoto similarity."""
    fps = [smiles_to_morgan_fp(smiles, as_numpy=False) for smiles in smiles_list]
    return 1 - avg_tanimoto_similarity(fps)


def topk_diversity(smiles_list: np.ndarray, scores_list: torch.Tensor, k: int = 10):
    """Compute diversity of top-k molecules by score."""
    if len(smiles_list) < k:
        return diversity(smiles_list)

    topk_indices = torch.argsort(scores_list, descending=True)[:k]
    topk_smiles = smiles_list[topk_indices]
    return diversity(topk_smiles)


def mode_metrics(smiles_list: np.ndarray, scores_list: torch.Tensor, k: int = 10, thresholds: List[float] = [7.2, 8]):
    """Compute mean score of top-k mode centroids and number of modes above given reward thresholds."""
    mode_metrics = {}

    centroid_indices = get_centroid_indices(list(smiles_list), scores_list)
    topk_indices = centroid_indices[: min(k, len(centroid_indices))]
    topk_mode_scores = scores_list[topk_indices]
    mode_metrics['topk_modes'] = topk_mode_scores.mean().item()

    for threshold in thresholds:
        num_modes = 0
        for idx in centroid_indices:
            if scores_list[idx] >= threshold:
                num_modes += 1
        mode_metrics[threshold] = num_modes
    
    return mode_metrics


def num_scaffolds(smiles_list: np.ndarray, scores_list: torch.Tensor, thresholds: List[float] = [7.2, 8]):
    scaffold_counts = {}

    for threshold in thresholds:
        scaffolds = set()
        for smiles, score in zip(smiles_list, scores_list):
            if score > threshold:
                try:
                    scaffold = MurckoScaffoldSmiles(smiles)
                    scaffolds.add(scaffold)
                except:
                    continue
        scaffold_counts[threshold] = len(scaffolds)
    
    return scaffold_counts


def auc_top10(scores_list: torch.Tensor, batch_size: int = 500) -> float:
    """
    Compute AUC Top-10 metric from PMO benchmark.
    
    This computes the area under the curve of the top-10 average scores
    over oracle calls, normalized to [0,1].
    
    Args:
        scores_list: All oracle scores seen so far
        batch_size: Number of oracle calls per BO iteration
    
    Returns:
        AUC Top-10 score normalized to [0,1]
    """
    n_oracle_calls = len(scores_list)
    acquisitions = list(range(batch_size, n_oracle_calls + 1, batch_size))
    
    top10_scores = [0]
    for batch in acquisitions:
        top10_mean = topk_mean(scores_list[:batch], k=10)
        top10_scores.append(top10_mean)
    
    auc = np.trapz(top10_scores, dx=batch_size)
    normalized_auc = auc / n_oracle_calls
    
    return normalized_auc


# ============================================================================
# Similarity-filtered metrics
# ============================================================================


def compute_similarity_filtered_metrics(
    smiles_list: np.ndarray,
    scores_list: torch.Tensor,
    initial_hit: str,
    mode_thresholds: List[float],
    similarity_threshold: float = 0.0,
) -> dict:
    """
    Metrics on molecules within `similarity_threshold` Tanimoto of `initial_hit`.
    Pass similarity_threshold=0 for unfiltered metrics (no `_sim*` suffix in keys).
    """
    suffix = f"_sim{similarity_threshold}" if similarity_threshold > 0 else ""

    hit_fp = smiles_to_morgan_fp(initial_hit, as_numpy=False)

    filtered_indices = []
    for i, smiles in enumerate(smiles_list):
        mol_fp = smiles_to_morgan_fp(smiles, as_numpy=False)
        sim = DataStructs.TanimotoSimilarity(mol_fp, hit_fp) if mol_fp is not None else -1
        if sim >= similarity_threshold:
            filtered_indices.append(i)

    if len(filtered_indices) == 0:
        empty = {
            f"best_score{suffix}": 0.0,
            f"oracle_mean{suffix}": 0.0,
            f"top10_mean{suffix}": 0.0,
            f"top10_diversity{suffix}": 0.0,
            f"top10_modes{suffix}": 0.0,
            f"count{suffix}": 0,
        }
        for t in mode_thresholds:
            empty[f"num_modes_{t}{suffix}"] = 0.0
            empty[f"num_scaffolds_{t}{suffix}"] = 0.0
        return empty

    filtered_smiles = smiles_list[filtered_indices]
    filtered_scores = scores_list[filtered_indices]

    mode_results = mode_metrics(filtered_smiles, filtered_scores, k=10, thresholds=mode_thresholds)
    scaffold_counts = num_scaffolds(filtered_smiles, filtered_scores, thresholds=mode_thresholds)

    metrics = {
        f"best_score{suffix}": best_score(filtered_scores),
        f"oracle_mean{suffix}": filtered_scores.mean().item(),
        f"top10_mean{suffix}": topk_mean(filtered_scores, k=10),
        f"top10_diversity{suffix}": topk_diversity(filtered_smiles, filtered_scores, k=10),
        f"top10_modes{suffix}": mode_results["topk_modes"],
        f"count{suffix}": len(filtered_smiles),
    }
    for t in mode_thresholds:
        metrics[f"num_modes_{t}{suffix}"] = mode_results[t]
        metrics[f"num_scaffolds_{t}{suffix}"] = scaffold_counts[t]
    return metrics

def compute_trajectory_metrics(
    smiles_list: np.ndarray,
    scores_list: torch.Tensor,
    initial_hit: str,
    mode_thresholds: List[float],
    sim_thresholds: List[float],
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Per-batch metrics across an acquisition trajectory.
    One row per batch (oracle_calls = batch_size, 2*batch_size, ...).
    AUC top-10 is a trajectory-level scalar attached to the final row;
    other rows have NaN in that column.
    """
    n_oracle_calls = len(scores_list)

    rows = []
    for n_calls in range(batch_size, n_oracle_calls + 1, batch_size):
        row = {"oracle_calls": n_calls}
        smiles_batch = smiles_list[:n_calls]
        scores_batch = scores_list[:n_calls]

        for sim in sim_thresholds:
            row.update(
                compute_similarity_filtered_metrics(
                    smiles_batch, scores_batch, initial_hit, mode_thresholds, sim
                )
            )
        rows.append(row)

    # Trajectory-level scalar on the final row; pandas fills NaN elsewhere.
    if rows:
        rows[-1]["auc_top10"] = auc_top10(scores_list, batch_size)

    return pd.DataFrame(rows)    
