
import pytest
import sys
from mofbuilder.core.net import FrameNet

@pytest.mark.core

class TestFrameNet:
    def test_create_net(self):
        cif_file = "tests/testdata/test.cif"
        netgraph = FrameNet()
        netgraph.create_net(cif_file=cif_file)