import pytest
import sys


try:
    import veloxchem as vlx
except ImportError:
    pass

from mofbuilder.core.linker import FrameLinker


@pytest.mark.core
class TestFrameLinker:


    def test_linker_2conn(self):
        linker_test = FrameLinker()
        linker_test.linker_connectivity = 2
        linker_test.filename = "tests/testdata/testlinker.xyz"
        #linker_test.target_dir = "tests/testoutput"
        linker_test._debug = True
        linker_test.create()

    def test_linker_4conn(self):
        linker_test = FrameLinker()
        linker_test.linker_connectivity = 4
        linker_test.filename = "tests/testdata/testtetralinker.xyz"
        #linker_test.target_dir = "tests/testoutput"
        linker_test.create()

    def test_linker_3conn(self):
        linker_test = FrameLinker()
        linker_test.linker_connectivity = 3
        linker_test.filename = "tests/testdata/testtrilinker.xyz"
        #linker_test.target_dir = "tests/testoutput"
        linker_test.create()

    @pytest.mark.skipif("veloxchem" not in sys.modules, reason="veloxchem not available")
    def test_linker_molecule(self):
        import veloxchem as vlx
        linker_test = FrameLinker()
        linker_test.linker_connectivity = 2
        molecule = vlx.Molecule.read_smiles("C1=C(C=C(C=C1C(=O)[O-])C(=O)[O-])C(=O)[O-]")
        linker_test.create(molecule=molecule)






