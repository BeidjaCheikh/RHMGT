from hera_hgt.chemistry import pharmacophore_registry_issues

def test_registry_compiles():
    assert pharmacophore_registry_issues() == []
