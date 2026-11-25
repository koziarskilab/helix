from pathlib import Path
from typing import List, Optional, Tuple

import gin
import pandas as pd


@gin.configurable
def load_starting_molecules(
    filepath: str,
    hit: int = 0,
    budget: int = 1,
) -> List[Tuple[str, int]]:
    """
    Load a single starting molecule from the consolidated hits CSV.

    Args:
        filepath: Path to consolidated hits.csv - one hit per row, 'molecule' column
        hit: Row index of the hit to load
        budget: Initial reaction budget

    Returns:
        List containing a single (smiles, budget) tuple.
    """
    df = pd.read_csv(Path(filepath))
    smiles = df.iloc[hit]['molecule']
    return [(smiles, budget)]
