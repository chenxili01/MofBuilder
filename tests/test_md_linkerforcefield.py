from mofbuilder.md.linkerforcefield import ForceFieldMapper
import pytest
import sys

try:
    import veloxchem as vlx
except ImportError:
    pass

try:
    import rdkit
except ImportError:
    pass


@pytest.mark.md
class LinkerForceFieldMappingTest:

    @pytest.mark.skipif("veloxchem" not in sys.modules, reason="veloxchem not available")
    @pytest.mark.skipif("rdkit" not in sys.modules, reason="rdkit not available")
    
    def test_linker_forcefield_mapping(self):
        import veloxchem as vlx
        import rdkit
                   
        oco_str = '''
        3

        O              1.036955000000        -0.586306000000         2.049074000000
        C              0.835228000000         0.631617000000         2.134616000000
        O              0.633501000000         1.849541000000         2.220158000000
        '''

        coo_str = '''
        3   

        C              0.835228000000         0.631617000000         2.134616000000
        O              1.036955000000        -0.586306000000         2.049074000000
        O              0.633501000000         1.849541000000         2.220158000000
        '''

        mapper = ForceFieldMapper()
        src_molecule = vlx.Molecule.read_xyz_string(oco_str)
        dest_molecule = vlx.Molecule.read_xyz_string(coo_str)

        mapping = mapper._get_mapping_between_two_molecules(    
            src_molecule,
            dest_molecule
        )
        expected_mapping = {1: 2, 2: 1, 3: 3}
        assert mapping == expected_mapping  



