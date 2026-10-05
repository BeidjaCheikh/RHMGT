from rdkit import Chem

from hera_hgt.chemistry import atom_feature_dim, atom_features_hergat


def test_hergat_atom_features_are_exactly_39d():
    mol = Chem.MolFromSmiles("C[C@H](N)Cl")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    feats = [atom_features_hergat(a) for a in mol.GetAtoms()]
    assert atom_feature_dim() == 39
    assert all(len(x) == 39 for x in feats)


def test_symbol_block_has_one_active_category():
    mol = Chem.MolFromSmiles("CCl")
    for atom in mol.GetAtoms():
        x = atom_features_hergat(atom)
        assert sum(x[:16]) == 1.0
