import numpy as np
from rdkit import Chem

from hera_hgt.chemistry import DESCRIPTOR_NAMES, descriptors_raw_10
from hera_hgt.data import DescriptorScaler


def test_descriptor_vector_is_10d_and_scalable():
    mol = Chem.MolFromSmiles("CCN(CC)CC")
    x = descriptors_raw_10(mol)
    assert len(DESCRIPTOR_NAMES) == 10
    assert x.shape == (10,)
    scaler = DescriptorScaler(mean=x.copy(), scale=np.ones(10, dtype=np.float32))
    assert np.allclose(scaler.transform(x), 0.0)
