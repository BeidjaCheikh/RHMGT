import numpy as np

from hera_hgt.config import FeaturizerConfig
from hera_hgt.hierarchy import (
    REL_ATOM_TO_MOTIF,
    REL_GLOBAL_TO_MOTIF,
    REL_MOTIF_TO_GLOBAL,
    featurize_hierarchical_smiles,
)


def test_hierarchical_graph_has_real_cross_level_edges_and_global_node():
    x = featurize_hierarchical_smiles("CN(C)CCc1ccccc1", FeaturizerConfig(max_motifs=24))
    rel = x["relation"]
    assert x["node_x"].shape[1] == 39
    assert x["n_atoms"] > 0
    assert x["n_motifs"] > 0
    assert int((rel == REL_ATOM_TO_MOTIF).sum()) > 0
    assert int((rel == REL_MOTIF_TO_GLOBAL).sum()) == x["n_motifs"]
    assert int((rel == REL_GLOBAL_TO_MOTIF).sum()) == x["n_motifs"]
    assert x["global_index"] == rel.shape[0] - 1
    assert x["pair_bond10"].shape[-1] == 10
    assert np.any(x["pair_bond10"] != 0)
    assert x["desc_raw"].shape == (10,)
    assert x["fp_morgan"].shape == (1024,)
    assert "fp_rdkit" not in x
