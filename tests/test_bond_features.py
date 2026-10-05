import numpy as np
from rdkit import Chem

from hera_hgt.chemistry import bond_features_10


def test_bond_features_are_10d_and_semantic():
    mol = Chem.MolFromSmiles("CC=C")
    f0 = bond_features_10(mol.GetBondWithIdx(0))
    f1 = bond_features_10(mol.GetBondWithIdx(1))
    assert f0.shape == (10,)
    assert f1.shape == (10,)
    assert np.isclose(f0[0], 1.0)  # single
    assert np.isclose(f1[1], 1.0)  # double
    assert np.isclose(f0[6:10].sum(), 1.0)  # one stereo state
